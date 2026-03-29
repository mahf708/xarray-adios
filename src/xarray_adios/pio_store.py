"""Low-level store for reading ADIOS BP files written by SCORPIO/PIO.

PIO writes BP files with a ``__pio__`` namespace:
  - ``/__pio__/dim/{name}``   — uint64 scalar per dimension
  - ``/__pio__/var/{name}``   — variable data stored as blocks (one per MPI rank per frame)
  - ``/__pio__/decomp/{ioid}`` — decomposition maps (1-based global indices)
  - ``/__pio__/track/``       — variable-to-decomposition associations

Reconstruction strategies:
  - **concat+reshape** (default): concatenate all blocks and reshape to the
    inferred dimensions.  Works for remapped / regular-grid output.
  - **decomposition scatter** (Phase 2): use the decomp map to scatter each
    rank's data into the correct global positions.  Required for unstructured /
    native-grid output (e.g. ne4pg2 spectral-element data before remapping).
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import adios2
import numpy as np

logger = logging.getLogger(__name__)

# Prefixes in the __pio__ namespace
_PIO_DIM_PREFIX = "/__pio__/dim/"
_PIO_VAR_PREFIX = "/__pio__/var/"
_PIO_DECOMP_PREFIX = "/__pio__/decomp/"
_PIO_TRACK_PREFIX = "/__pio__/track/"

# ADIOS type string to numpy dtype
_ADIOS_TYPE_MAP = {
    "float": np.float32,
    "double": np.float64,
    "int8_t": np.int8,
    "int16_t": np.int16,
    "int32_t": np.int32,
    "int64_t": np.int64,
    "uint8_t": np.uint8,
    "uint16_t": np.uint16,
    "uint32_t": np.uint32,
    "uint64_t": np.uint64,
    "string": str,
}


class VariableInfo:
    """Metadata about a PIO variable."""

    __slots__ = (
        "name",
        "pio_name",
        "dims",
        "shape",
        "dtype",
        "attrs",
        "nblocks",
        "block_counts",
        "decomp_id",
    )

    def __init__(
        self,
        name: str,
        pio_name: str,
        dims: tuple[str, ...],
        shape: tuple[int, ...],
        dtype: np.dtype,
        attrs: dict[str, Any],
        nblocks: int,
        block_counts: list[int],
        decomp_id: str | None = None,
    ):
        self.name = name
        self.pio_name = pio_name
        self.dims = dims
        self.shape = shape
        self.dtype = dtype
        self.attrs = attrs
        self.nblocks = nblocks
        self.block_counts = block_counts
        self.decomp_id = decomp_id


class PioStore:
    """Read PIO-formatted ADIOS BP files.

    Opens the file in ReadRandomAccess mode and provides methods to
    read dimensions, variables, and attributes.  When decomposition maps
    are present (``__pio__/decomp/*``), variables that reference a
    decomposition are reconstructed via index-scatter instead of plain
    concat+reshape.
    """

    def __init__(self, filename: str):
        self._filename = str(filename)
        self._adios = adios2.Adios()
        self._io = self._adios.declare_io("xarray_pio_reader")
        self._engine = self._io.open(self._filename, adios2.Mode.ReadRandomAccess)

        self._all_vars = self._io.available_variables()
        self._all_attrs = self._io.available_attributes()

        self._dimensions: dict[str, int] | None = None
        self._variable_info: dict[str, VariableInfo] | None = None
        self._decomp_maps: dict[str, list[np.ndarray]] | None = None
        self._var_decomp_map: dict[str, str] | None = None
        self._var_defs: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    def get_dimensions(self) -> dict[str, int]:
        """Read all dimension sizes from ``__pio__/dim/*``."""
        if self._dimensions is not None:
            return self._dimensions

        dims: dict[str, int] = {}
        for vname in self._all_vars:
            if vname.startswith(_PIO_DIM_PREFIX):
                dim_name = vname[len(_PIO_DIM_PREFIX) :]
                var = self._io.inquire_variable(vname)
                val = np.zeros(1, dtype=np.uint64)
                self._engine.get(var, val)
                self._engine.perform_gets()
                dims[dim_name] = int(val[0])

        self._dimensions = dims
        return dims

    # ------------------------------------------------------------------
    # Variable definitions (def/dims, def/decomp from SCORPIO metadata)
    # ------------------------------------------------------------------

    def _get_var_defs(self) -> dict[str, dict[str, Any]]:
        """Extract SCORPIO variable definitions from ``def/*`` attributes.

        Returns a dict mapping variable short name → {dims, decomp, ndims, ...}.
        """
        if self._var_defs is not None:
            return self._var_defs

        defs: dict[str, dict[str, Any]] = {}
        for aname, ainfo in self._all_attrs.items():
            if not aname.startswith(_PIO_VAR_PREFIX):
                continue
            rest = aname[len(_PIO_VAR_PREFIX) :]
            if "/def/" not in rest:
                continue
            var_name, _, def_key = rest.partition("/def/")
            val = _parse_attr_value(ainfo)
            defs.setdefault(var_name, {})[def_key] = val

        self._var_defs = defs
        return defs

    def _dims_from_def(
        self, def_dims_val: Any, total_elements: int, dims: dict[str, int]
    ) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
        """Compute dimension names and shape from a ``def/dims`` attribute value.

        Parameters
        ----------
        def_dims_val : str or array-like
            The raw value of the ``def/dims`` attribute, e.g. ``'{ "time", "ncol" }'``.
        total_elements : int
            Total number of data elements across all blocks.
        dims : dict
            Dimension name → size mapping from ``get_dimensions()``.

        Returns
        -------
        (dim_names, shape) or None if the attribute value cannot be parsed.
        """
        dim_names = _parse_string_array(def_dims_val)
        if not dim_names:
            return None

        shape: list[int] = []
        unknown_idx: int | None = None

        for i, d in enumerate(dim_names):
            size = dims.get(d, 0)
            if size > 0:
                shape.append(size)
            elif d == "time" or size == 0:
                # Unlimited / unknown — will compute from data
                if unknown_idx is not None:
                    return None  # Multiple unknown dims — can't resolve
                unknown_idx = i
                shape.append(0)  # placeholder
            else:
                return None

        if unknown_idx is not None:
            known_product = 1
            for s in shape:
                if s > 0:
                    known_product *= s
            if known_product > 0 and total_elements % known_product == 0:
                shape[unknown_idx] = total_elements // known_product
            else:
                return None

        return tuple(dim_names), tuple(shape)

    # ------------------------------------------------------------------
    # Variable catalog
    # ------------------------------------------------------------------

    def get_variables(self) -> dict[str, VariableInfo]:
        """Catalog all science variables under ``__pio__/var/*``."""
        if self._variable_info is not None:
            return self._variable_info

        dims = self.get_dimensions()
        var_defs = self._get_var_defs()
        var_decomps = self._get_var_decomp_mapping()
        variables: dict[str, VariableInfo] = {}

        for vname in sorted(self._all_vars):
            if not vname.startswith(_PIO_VAR_PREFIX):
                continue

            short_name = vname[len(_PIO_VAR_PREFIX) :]
            var = self._io.inquire_variable(vname)
            if var is None:
                continue

            dtype = _adios_dtype(var)
            if dtype is str:
                # String variables — read eagerly, store as attrs later
                continue

            # Read block info to determine shape
            try:
                block_info = self._engine.blocks_info(vname, 0)
            except UnicodeDecodeError:
                logger.debug(
                    "Skipping variable %s: blocks_info raised UnicodeDecodeError", short_name
                )
                continue
            nblocks = len(block_info)
            block_counts = [int(bi["Count"]) for bi in block_info]
            total_elements = sum(block_counts)

            # Detect scalars: SingleValue or ndims=0 with 0 data elements
            vdef = var_defs.get(short_name, {})
            vinfo_dict = self._all_vars[vname]
            is_scalar = vinfo_dict.get("SingleValue", "") == "true"
            if not is_scalar:
                ndims_val = vdef.get("ndims")
                if ndims_val is not None and int(ndims_val) == 0:
                    is_scalar = True

            if is_scalar:
                variables[short_name] = VariableInfo(
                    name=short_name,
                    pio_name=vname,
                    dims=(),
                    shape=(),
                    dtype=np.dtype(dtype),
                    attrs=self._read_var_attrs(vname, short_name),
                    nblocks=nblocks,
                    block_counts=block_counts,
                    decomp_id=None,
                )
                continue

            # Decomp ID: prefer def/decomp attribute, fall back to heuristic
            decomp_id = None
            def_decomp = vdef.get("decomp")
            if def_decomp is not None:
                decomp_id = str(def_decomp).strip('"')
            if decomp_id is None:
                decomp_id = var_decomps.get(short_name)

            # Dimension names and shape: prefer def/dims, fall back to heuristic
            var_dims: tuple[str, ...] | None = None
            var_shape: tuple[int, ...] | None = None
            def_dims = vdef.get("dims")
            if def_dims is not None:
                result = self._dims_from_def(def_dims, total_elements, dims)
                if result is not None:
                    var_dims, var_shape = result

            if var_dims is None or var_shape is None:
                var_dims, var_shape = self._infer_dims_and_shape(
                    short_name,
                    total_elements,
                    block_counts,
                    dims,
                    var,
                    decomp_id=decomp_id,
                )

            # Read variable attributes (exclude def/* internal metadata)
            attrs = self._read_var_attrs(vname, short_name)

            variables[short_name] = VariableInfo(
                name=short_name,
                pio_name=vname,
                dims=var_dims,
                shape=var_shape,
                dtype=np.dtype(dtype),
                attrs=attrs,
                nblocks=nblocks,
                block_counts=block_counts,
                decomp_id=decomp_id,
            )

        self._variable_info = variables
        return variables

    # ------------------------------------------------------------------
    # Reading variable data
    # ------------------------------------------------------------------

    def read_variable(self, name: str, key: tuple | None = None) -> np.ndarray:
        """Read a variable, reconstructing the global array from blocks.

        Parameters
        ----------
        name : str
            Short variable name (without ``__pio__/var/`` prefix).
        key : tuple of slices/ints, optional
            Indexing key for lazy loading. If None, read everything.

        Returns
        -------
        np.ndarray
        """
        info = self.get_variables()[name]

        # Scalar: read via ADIOS get (SingleValue)
        if info.shape == ():
            var = self._io.inquire_variable(info.pio_name)
            buf = np.zeros(1, dtype=info.dtype)
            self._engine.get(var, buf)
            self._engine.perform_gets()
            return buf.reshape(())

        if info.decomp_id is not None:
            full = self._read_blocks_decomp(info)
        else:
            full = self._read_blocks(info)

        if key is not None:
            full = full[key]

        return full

    def _read_raw_blocks(
        self, pio_name: str, nblocks: int, block_counts: list[int], dtype: np.dtype
    ) -> list[np.ndarray]:
        """Read all blocks for a variable and return them as a list."""
        var = self._io.inquire_variable(pio_name)
        var.set_step_selection((0, 1))

        blocks: list[np.ndarray] = []
        for bid in range(nblocks):
            var.set_block_selection(bid)
            count = block_counts[bid]
            block = np.zeros(count, dtype=dtype)
            self._engine.get(var, block)
            self._engine.perform_gets()
            blocks.append(block)

        return blocks

    def _read_blocks(self, info: VariableInfo) -> np.ndarray:
        """Read all blocks for a variable and reconstruct via concat+reshape."""
        blocks = self._read_raw_blocks(info.pio_name, info.nblocks, info.block_counts, info.dtype)
        flat = np.concatenate(blocks)

        # Reshape to the inferred shape
        try:
            return flat.reshape(info.shape)
        except ValueError:
            warnings.warn(
                f"Variable '{info.name}': cannot reshape {flat.shape} to {info.shape}, "
                f"returning flat array.",
                stacklevel=2,
            )
            return flat

    def _read_blocks_decomp(self, info: VariableInfo) -> np.ndarray:
        """Read blocks and reconstruct using decomposition map (scatter).

        Each rank wrote a subset of the global array.  The decomposition map
        for that rank contains 1-based global indices indicating where each
        element belongs.  We scatter the data into a pre-allocated global
        array.

        For multi-frame variables (e.g. ``time × ncol``), the decomp map
        applies to the spatial portion of each frame.  Frames are detected
        by comparing each rank's data count against its decomp count.
        """
        assert info.decomp_id is not None
        decomp_rank_blocks = self._get_decomp_rank_blocks(info.decomp_id)

        # Read data blocks
        data_blocks = self._read_raw_blocks(
            info.pio_name, info.nblocks, info.block_counts, info.dtype
        )

        # Determine spatial size from decomp map
        all_indices = np.concatenate(decomp_rank_blocks)
        spatial_size = int(np.max(all_indices))  # 1-based → max = global spatial size

        nranks = len(decomp_rank_blocks)
        ndata_blocks = len(data_blocks)

        # Detect frames: either ndata_blocks = nranks × nframes,
        # or each data block is nframes × decomp_count for that rank.
        nframes = self._detect_nframes(data_blocks, decomp_rank_blocks, nranks, ndata_blocks)

        if nframes > 1 and ndata_blocks == nranks:
            # Embedded frames: each data block has nframes × decomp_count elements
            result = np.zeros((nframes, spatial_size), dtype=info.dtype)
            for r in range(nranks):
                dmap = decomp_rank_blocks[r]
                data = data_blocks[r]
                stride = len(dmap)
                valid = dmap > 0
                for f in range(nframes):
                    frame_data = data[f * stride : (f + 1) * stride]
                    result[f, dmap[valid] - 1] = frame_data[valid]
        elif nframes > 1 and ndata_blocks == nranks * nframes:
            # Separate blocks per frame: blocks ordered [f0_r0, f0_r1, ..., f1_r0, ...]
            result = np.zeros((nframes, spatial_size), dtype=info.dtype)
            for f in range(nframes):
                for r in range(nranks):
                    bid = f * nranks + r
                    dmap = decomp_rank_blocks[r]
                    data = data_blocks[bid]
                    valid = dmap > 0
                    result[f, dmap[valid] - 1] = data[valid]
        else:
            # Single frame / spatial-only
            result = np.zeros(spatial_size, dtype=info.dtype)
            nranks_actual = min(nranks, ndata_blocks)
            for r in range(nranks_actual):
                dmap = decomp_rank_blocks[r]
                data = data_blocks[r]
                valid = dmap > 0
                result[dmap[valid] - 1] = data[valid]

        try:
            return result.reshape(info.shape)
        except ValueError:
            warnings.warn(
                f"Variable '{info.name}': cannot reshape decomp result "
                f"{result.shape} to {info.shape}, returning as-is.",
                stacklevel=2,
            )
            return result

    @staticmethod
    def _detect_nframes(
        data_blocks: list[np.ndarray],
        decomp_blocks: list[np.ndarray],
        nranks: int,
        ndata_blocks: int,
    ) -> int:
        """Detect how many frames (timesteps) a variable spans."""
        if ndata_blocks > nranks and ndata_blocks % nranks == 0:
            return ndata_blocks // nranks

        # Check if each data block is an integer multiple of its decomp block
        if ndata_blocks == nranks and nranks > 0:
            ratios = []
            for r in range(nranks):
                dlen = len(decomp_blocks[r])
                if dlen == 0:
                    continue
                blen = len(data_blocks[r])
                if blen % dlen != 0:
                    return 1
                ratios.append(blen // dlen)
            if ratios and len(set(ratios)) == 1 and ratios[0] > 1:
                return ratios[0]

        return 1

    # ------------------------------------------------------------------
    # Decomposition maps
    # ------------------------------------------------------------------

    def _get_decomp_rank_blocks(self, ioid: str) -> list[np.ndarray]:
        """Read the per-rank decomposition map blocks for *ioid*.

        Returns a list of 1-D int64 arrays, one per MPI rank, containing the
        1-based global indices assigned to that rank.
        """
        if self._decomp_maps is None:
            self._decomp_maps = {}

        if ioid not in self._decomp_maps:
            pio_name = f"{_PIO_DECOMP_PREFIX}{ioid}"
            var = self._io.inquire_variable(pio_name)
            if var is None:
                raise KeyError(f"Decomposition map not found: {pio_name}")

            try:
                bi = self._engine.blocks_info(pio_name, 0)
            except UnicodeDecodeError as exc:
                raise KeyError(
                    f"Cannot read decomposition map {pio_name}: "
                    "blocks_info raised UnicodeDecodeError"
                ) from exc
            nblocks = len(bi)
            counts = [int(b["Count"]) for b in bi]
            # Determine the underlying integer dtype
            dtype = _adios_dtype(var)
            if dtype is str:
                dtype = np.int64
            np_dtype = np.dtype(dtype)

            self._decomp_maps[ioid] = self._read_raw_blocks(pio_name, nblocks, counts, np_dtype)

        return self._decomp_maps[ioid]

    def get_decomp_ids(self) -> list[str]:
        """Return the ioid strings for all decomposition maps in the file."""
        return [
            vname[len(_PIO_DECOMP_PREFIX) :]
            for vname in self._all_vars
            if vname.startswith(_PIO_DECOMP_PREFIX)
        ]

    def has_decomp_maps(self) -> bool:
        """Return True if the file contains any decomposition maps."""
        return any(v.startswith(_PIO_DECOMP_PREFIX) for v in self._all_vars)

    def _get_var_decomp_mapping(self) -> dict[str, str]:
        """Build a mapping of variable short-name → decomp ioid.

        Discovery strategies (in priority order):

        1. **Attribute-based**: an attribute ``{pio_name}/_pio_decomp`` or
           ``{short_name}/_pio_decomp`` whose value is the ioid string.
        2. **Track-attribute-based**: attributes under ``__pio__/track/``
           of the form ``/__pio__/track/{varname} = ioid``.
        3. **Block-matching heuristic**: match each variable's per-rank block
           counts against the per-rank block counts of available decomposition
           maps.  A unique match is used.
        """
        if self._var_decomp_map is not None:
            return self._var_decomp_map

        mapping: dict[str, str] = {}

        # ------ strategy 1: attribute-based ------
        for aname, ainfo in self._all_attrs.items():
            if aname.endswith("/_pio_decomp"):
                val = _parse_attr_value(ainfo)
                # Extract variable short name
                prefix = aname[: -len("/_pio_decomp")]
                if prefix.startswith(_PIO_VAR_PREFIX):
                    short = prefix[len(_PIO_VAR_PREFIX) :]
                else:
                    short = prefix
                mapping[short] = str(val)

        # ------ strategy 2: track-based ------
        for aname, ainfo in self._all_attrs.items():
            if not aname.startswith(_PIO_TRACK_PREFIX):
                continue
            # Expected format: /__pio__/track/{varname} = ioid
            suffix = aname[len(_PIO_TRACK_PREFIX) :]
            if "/" not in suffix:
                val = _parse_attr_value(ainfo)
                if suffix not in mapping:
                    mapping[suffix] = str(val)

        # ------ strategy 3: block-count heuristic (for unmapped vars) ------
        decomp_ids = self.get_decomp_ids()
        if not decomp_ids:
            self._var_decomp_map = {}
            return {}

        # Build block-count signatures for each decomp
        decomp_sigs: dict[str, tuple[int, ...]] = {}
        for ioid in decomp_ids:
            dname = f"{_PIO_DECOMP_PREFIX}{ioid}"
            try:
                bi = self._engine.blocks_info(dname, 0)
            except UnicodeDecodeError:
                continue
            decomp_sigs[ioid] = tuple(int(b["Count"]) for b in bi)

        # For each variable, check if its block counts (or a sub-multiple for
        # multi-frame variables) match exactly one decomposition.
        for vname in sorted(self._all_vars):
            if not vname.startswith(_PIO_VAR_PREFIX):
                continue
            short = vname[len(_PIO_VAR_PREFIX) :]
            if short in mapping:
                continue

            try:
                bi = self._engine.blocks_info(vname, 0)
            except UnicodeDecodeError:
                continue
            var_counts = tuple(int(b["Count"]) for b in bi)
            nvar = len(var_counts)

            candidates: list[str] = []
            for ioid, dcounts in decomp_sigs.items():
                nd = len(dcounts)
                if nd == 0:
                    continue
                # Exact block-count match (single frame)
                if var_counts == dcounts:
                    candidates.append(ioid)
                    continue
                # Multi-frame: nvar blocks == nd blocks × nframes
                if nvar > nd and nvar % nd == 0:
                    nframes = nvar // nd
                    # Check if repeated decomp pattern matches
                    if all(
                        var_counts[f * nd + r] == dcounts[r]
                        for f in range(nframes)
                        for r in range(nd)
                    ):
                        candidates.append(ioid)
                        continue
                # Embedded frames: same nblocks, each var block is k × decomp block
                if nvar == nd:
                    ratios = []
                    ok = True
                    for r in range(nd):
                        if dcounts[r] == 0:
                            ok = False
                            break
                        if var_counts[r] % dcounts[r] != 0:
                            ok = False
                            break
                        ratios.append(var_counts[r] // dcounts[r])
                    if ok and ratios and len(set(ratios)) == 1 and ratios[0] > 1:
                        candidates.append(ioid)

            if len(candidates) == 1:
                mapping[short] = candidates[0]

        self._var_decomp_map = mapping
        return mapping

    # ------------------------------------------------------------------
    # Dimension inference
    # ------------------------------------------------------------------

    def _infer_dims_and_shape(
        self,
        var_name: str,
        total_elements: int,
        block_counts: list[int],
        dims: dict[str, int],
        adios_var,
        decomp_id: str | None = None,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Infer dimension names and shape for a variable.

        Strategy:
        1. If a decomposition map is associated, derive the spatial size from the
           map and detect frames from the data/decomp ratio.
        2. Try to match against known dimension sizes.
        3. Fall back to 1D.
        """
        if total_elements == 0:
            return ("_empty",), (0,)

        # Scalar
        if total_elements == 1 and len(block_counts) == 1:
            return (), ()

        # ---- decomp-aware inference ----
        if decomp_id is not None:
            return self._infer_dims_decomp(var_name, total_elements, block_counts, dims, decomp_id)

        # ---- original concat+reshape inference ----

        # Check if this is a coordinate variable (1D, matches a dim size)
        for dname, dsize in dims.items():
            if total_elements == dsize:
                return (dname,), (dsize,)

        # Try 2D: time × spatial
        if "time" in dims:
            ntime = dims["time"]
            if ntime > 0 and total_elements % ntime == 0:
                spatial = total_elements // ntime
                spatial_dims = self._match_spatial(spatial, dims, exclude={"time"})
                if spatial_dims is not None:
                    dim_names, dim_shape = spatial_dims
                    return ("time", *dim_names), (ntime, *dim_shape)

        # Try common spatial patterns without time
        spatial_dims = self._match_spatial(total_elements, dims)
        if spatial_dims is not None:
            return spatial_dims

        # Try matching total against ntime × nlev × ncol (3D fields)
        if "time" in dims and "lev" in dims:
            ntime = dims["time"]
            nlev = dims["lev"]
            if ntime > 0 and nlev > 0 and total_elements % (ntime * nlev) == 0:
                ncol = total_elements // (ntime * nlev)
                col_dim = self._find_dim_by_size(ncol, dims, exclude={"time", "lev"})
                if col_dim:
                    return ("time", "lev", col_dim), (ntime, nlev, ncol)
                elif ncol > 1:
                    return ("time", "lev", f"ncol_{ncol}"), (ntime, nlev, ncol)

        # Try time × ilev for interface-level variables
        if "time" in dims and "ilev" in dims:
            ntime = dims["time"]
            nilev = dims["ilev"]
            if ntime > 0 and nilev > 0 and total_elements % (ntime * nilev) == 0:
                ncol = total_elements // (ntime * nilev)
                col_dim = self._find_dim_by_size(ncol, dims, exclude={"time", "ilev"})
                if col_dim:
                    return ("time", "ilev", col_dim), (ntime, nilev, ncol)

        # Fallback: 1D with generated dim name
        return (f"dim_{var_name}_{total_elements}",), (total_elements,)

    def _infer_dims_decomp(
        self,
        var_name: str,
        total_elements: int,
        block_counts: list[int],
        dims: dict[str, int],
        decomp_id: str,
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Infer dims/shape for a variable that uses a decomposition map."""
        decomp_blocks = self._get_decomp_rank_blocks(decomp_id)
        all_indices = np.concatenate(decomp_blocks)
        spatial_size = int(np.max(all_indices))  # 1-based

        decomp_total = sum(len(b) for b in decomp_blocks)

        # Find the spatial dimension name
        spatial_dim = self._find_dim_by_size(spatial_size, dims) or f"ncol_{spatial_size}"

        # Detect frames
        if (
            decomp_total > 0
            and total_elements > decomp_total
            and total_elements % decomp_total == 0
        ):
            nframes = total_elements // decomp_total
            # Prefer an explicit "time" dimension when it matches nframes,
            # and only fall back to a generic size-based lookup otherwise.
            if "time" in dims and dims["time"] == nframes:
                time_dim: str | None = "time"
            else:
                time_dim = self._find_dim_by_size(nframes, dims)
            if time_dim:
                return (time_dim, spatial_dim), (nframes, spatial_size)
            return (f"frame_{nframes}", spatial_dim), (nframes, spatial_size)

        return (spatial_dim,), (spatial_size,)

    def _match_spatial(
        self,
        n: int,
        dims: dict[str, int],
        exclude: set[str] | None = None,
    ) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
        """Try to match n elements to known spatial dimension combinations."""
        exclude = exclude or set()
        avail = {k: v for k, v in dims.items() if k not in exclude and v > 0}

        # Single dimension match
        for dname, dsize in avail.items():
            if n == dsize:
                return (dname,), (dsize,)

        # 2D: try lat × lon, ncol, etc.
        if "lat" in avail and "lon" in avail and n == avail["lat"] * avail["lon"]:
            return ("lat", "lon"), (avail["lat"], avail["lon"])

        # lev × ncol or lev × (lat × lon)
        if "lev" in avail:
            nlev = avail["lev"]
            if nlev > 0 and n % nlev == 0:
                remaining = n // nlev
                for dname, dsize in avail.items():
                    if dname != "lev" and remaining == dsize:
                        return ("lev", dname), (nlev, dsize)
                if "lat" in avail and "lon" in avail and remaining == avail["lat"] * avail["lon"]:
                    return ("lev", "lat", "lon"), (nlev, avail["lat"], avail["lon"])

        return None

    def _find_dim_by_size(
        self, size: int, dims: dict[str, int], exclude: set[str] | None = None
    ) -> str | None:
        """Find a dimension name matching the given size."""
        exclude = exclude or set()
        for dname, dsize in dims.items():
            if dname not in exclude and dsize == size:
                return dname
        return None

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    def _ensure_attr_index(self) -> None:
        """Build an index of attributes by variable name/prefix (lazy).

        Avoids repeatedly scanning ``self._all_attrs`` in
        :meth:`_read_var_attrs`, which can become expensive for large
        files with many variables and attributes.
        """
        if getattr(self, "_attr_index_built", False):
            return

        self._attrs_by_pio: dict[str, dict[str, Any]] = {}
        self._attrs_by_short: dict[str, dict[str, Any]] = {}

        for aname, ainfo in self._all_attrs.items():
            if aname.startswith(_PIO_VAR_PREFIX):
                var_path, _, attr_name = aname.rpartition("/")
                if not attr_name:
                    continue
                self._attrs_by_pio.setdefault(var_path, {})[attr_name] = _parse_attr_value(ainfo)
            else:
                if "/" not in aname:
                    continue
                short_name, _, attr_name = aname.partition("/")
                if not attr_name:
                    continue
                self._attrs_by_short.setdefault(short_name, {})[attr_name] = _parse_attr_value(
                    ainfo
                )

        self._attr_index_built = True

    def _read_var_attrs(self, pio_name: str, short_name: str) -> dict[str, Any]:
        """Read attributes for a variable.

        PIO stores variable attributes as ADIOS attributes, typically
        associated with the ``__pio__/var/{name}`` path.
        """
        self._ensure_attr_index()

        attrs: dict[str, Any] = {}
        pio_attrs = self._attrs_by_pio.get(pio_name)
        if pio_attrs:
            attrs.update(pio_attrs)
        short_attrs = self._attrs_by_short.get(short_name)
        if short_attrs:
            attrs.update(short_attrs)

        return attrs

    def get_global_attrs(self) -> dict[str, Any]:
        """Read global (file-level) attributes."""
        attrs: dict[str, Any] = {}
        var_names = set(self._all_vars.keys())
        # Also collect short variable names so that attributes stored as
        # "{short_name}/{attr_name}" are correctly recognised as variable-scoped.
        short_var_names = {
            v[len(_PIO_VAR_PREFIX) :] for v in self._all_vars if v.startswith(_PIO_VAR_PREFIX)
        }

        for aname, ainfo in self._all_attrs.items():
            # Skip attributes that belong to variables in the PIO namespace
            if aname.startswith("/__pio__/"):
                continue
            # Skip if this looks like a variable attribute (contains / after first segment)
            parts = aname.split("/")
            if len(parts) > 1 and (parts[0] in var_names or parts[0] in short_var_names):
                continue
            attrs[aname] = _parse_attr_value(ainfo)

        return attrs

    # ------------------------------------------------------------------
    # String variables (read as attributes/coords)
    # ------------------------------------------------------------------

    def read_string_variable(self, pio_name: str) -> str | np.ndarray:
        """Read a string-type variable."""
        var = self._io.inquire_variable(pio_name)
        if var is None:
            raise KeyError(pio_name)
        result = self._engine.get(var)
        self._engine.perform_gets()
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the ADIOS engine."""
        self._engine.close()


def _parse_string_array(value: Any) -> tuple[str, ...]:
    """Parse a SCORPIO string-array attribute into a tuple of strings.

    Handles formats like:
    - ``'{ "time", "ncol" }'`` → ``("time", "ncol")``
    - ``'"ncol"'`` → ``("ncol",)``
    - ``'ncol'`` → ``("ncol",)``
    """
    if not isinstance(value, str):
        return ()
    s = value.strip()
    if s.startswith("{"):
        inner = s.strip("{ }")
        return tuple(p.strip().strip('"') for p in inner.split(",") if p.strip())
    return (s.strip('"'),)


def _adios_dtype(var) -> type:
    """Get numpy dtype from an ADIOS variable."""
    type_str = var.type()
    return _ADIOS_TYPE_MAP.get(type_str, np.float64)


def _parse_attr_value(attr_info: dict) -> Any:
    """Parse an ADIOS attribute info dict into a Python value."""
    value = attr_info.get("Value", "")
    atype = attr_info.get("Type", "")
    nelements = int(attr_info.get("Elements", "1"))

    if atype == "string":
        # Strip surrounding quotes if present
        if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value
    elif atype in ("float", "double"):
        if nelements > 1 or (isinstance(value, str) and value.startswith("{")):
            # Array attribute like "{ 1.0, 2.0 }"
            items = value.strip("{ }").split(",")
            return np.array([float(x.strip()) for x in items if x.strip()])
        return float(value)
    elif "int" in atype:
        if nelements > 1 or (isinstance(value, str) and value.startswith("{")):
            # Array attribute like "{ 15, 30 }"
            items = value.strip("{ }").split(",")
            return np.array([int(x.strip()) for x in items if x.strip()])
        return int(value)
    else:
        return value


def is_pio_file(filename: str) -> bool:
    """Check if a BP file was written by PIO (has __pio__ namespace)."""
    try:
        adios_obj = adios2.Adios()
        io = adios_obj.declare_io("pio_check")
        engine = io.open(str(filename), adios2.Mode.ReadRandomAccess)
        all_vars = io.available_variables()
        engine.close()
        return any(v.startswith("/__pio__/") for v in all_vars)
    except Exception:
        return False
