"""Low-level store for reading ADIOS BP files written by SCORPIO/PIO.

PIO writes BP files with a ``__pio__`` namespace:
  - ``/__pio__/dim/{name}``   — uint64 scalar per dimension
  - ``/__pio__/var/{name}``   — variable data stored as blocks (one per MPI rank per frame)
  - ``/__pio__/decomp/{ioid}`` — decomposition maps (1-based global indices)
  - ``/__pio__/track/``       — variable-to-decomposition associations

Phase 1 uses simple concat+reshape reconstruction (works for remapped/regular grids).
Phase 2 will add decomposition map support for unstructured grids.
"""

from __future__ import annotations

import contextlib
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

    __slots__ = ("name", "pio_name", "dims", "shape", "dtype", "attrs", "nblocks", "block_counts")

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
    ):
        self.name = name
        self.pio_name = pio_name
        self.dims = dims
        self.shape = shape
        self.dtype = dtype
        self.attrs = attrs
        self.nblocks = nblocks
        self.block_counts = block_counts


class PioStore:
    """Read PIO-formatted ADIOS BP files.

    Opens the file in ReadRandomAccess mode and provides methods to
    read dimensions, variables, and attributes.
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
    # Variable catalog
    # ------------------------------------------------------------------

    def get_variables(self) -> dict[str, VariableInfo]:
        """Catalog all science variables under ``__pio__/var/*``."""
        if self._variable_info is not None:
            return self._variable_info

        dims = self.get_dimensions()
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
            block_info = self._engine.blocks_info(vname, 0)
            nblocks = len(block_info)
            block_counts = [_parse_block_count(bi["Count"]) for bi in block_info]
            total_elements = sum(block_counts)

            # Infer dimensions and shape
            var_dims, var_shape = self._infer_dims_and_shape(
                short_name, total_elements, block_counts, dims, var
            )

            # Read variable attributes
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
        full = self._read_blocks(info)

        if key is not None:
            full = full[key]

        return full

    def _read_blocks(self, info: VariableInfo) -> np.ndarray:
        """Read all blocks for a variable and reconstruct the global array."""
        if info.nblocks == 0:
            return np.empty(info.shape, dtype=info.dtype)

        var = self._io.inquire_variable(info.pio_name)
        var.set_step_selection((0, 1))

        blocks = []
        for bid in range(info.nblocks):
            var.set_block_selection(bid)
            count = info.block_counts[bid]
            block = np.zeros(count, dtype=info.dtype)
            self._engine.get(var, block)
            self._engine.perform_gets()
            blocks.append(block)

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
    ) -> tuple[tuple[str, ...], tuple[int, ...]]:
        """Infer dimension names and shape for a variable.

        Strategy:
        1. Check for explicit shape from ADIOS variable metadata
        2. Try to match against known dimension sizes
        3. Fall back to 1D
        """
        if total_elements == 0:
            return ("_empty",), (0,)

        # Scalar
        if total_elements == 1 and len(block_counts) == 1:
            return (), ()

        # Try to detect time dimension from block structure
        # If all blocks have the same count, nblocks might be nranks × ntimes
        # For regular grid output: each block is one rank's portion of one timestep

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

    def _read_var_attrs(self, pio_name: str, short_name: str) -> dict[str, Any]:
        """Read attributes for a variable.

        PIO stores variable attributes as ADIOS attributes, typically
        associated with the ``__pio__/var/{name}`` path.
        """
        attrs: dict[str, Any] = {}
        # ADIOS attributes associated with the variable
        for aname, ainfo in self._all_attrs.items():
            # Attributes can be stored as:
            #   /__pio__/var/{name}/{attr_name}
            #   {name}/{attr_name}
            var_prefix = f"{pio_name}/"
            short_prefix = f"{short_name}/"
            if aname.startswith(var_prefix):
                attr_name = aname[len(var_prefix) :]
                attrs[attr_name] = _parse_attr_value(ainfo)
            elif aname.startswith(short_prefix):
                attr_name = aname[len(short_prefix) :]
                attrs[attr_name] = _parse_attr_value(ainfo)

        return attrs

    def get_global_attrs(self) -> dict[str, Any]:
        """Read global (file-level) attributes."""
        attrs: dict[str, Any] = {}
        var_names = set(self._all_vars.keys())

        for aname, ainfo in self._all_attrs.items():
            # Skip attributes that belong to variables
            if aname.startswith("/__pio__/"):
                continue
            # Skip if this looks like a variable attribute (contains / after first segment)
            parts = aname.split("/")
            if len(parts) > 1 and parts[0] in var_names:
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


def _adios_dtype(var) -> type:
    """Get numpy dtype from an ADIOS variable."""
    type_str = var.type()
    return _ADIOS_TYPE_MAP.get(type_str, np.float64)


def _parse_attr_value(attr_info: dict) -> Any:
    """Parse an ADIOS attribute info dict into a Python value."""
    value = attr_info.get("Value", "")
    atype = attr_info.get("Type", "")

    if atype == "string":
        # Strip surrounding quotes if present
        if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value
    elif atype in ("float", "double"):
        return float(value)
    elif "int" in atype:
        return int(value)
    else:
        return value


def _parse_block_count(count_value: str | int) -> int:
    """Parse a block Count value which may be a scalar or comma-separated dimensions.

    ADIOS2 ``blocks_info`` reports ``Count`` as a string that can be either
    a single integer (``"100"``) or a comma-separated shape (``"10,20"``).
    Returns the total element count (product of dimensions).
    """
    s = str(count_value).strip()
    if "," in s:
        parts = [int(p) for p in s.split(",") if p.strip()]
        result = 1
        for p in parts:
            result *= p
        return result
    return int(s)


def is_pio_file(filename: str) -> bool:
    """Check if a BP file was written by PIO (has __pio__ namespace)."""
    engine = None
    try:
        adios_obj = adios2.Adios()
        io = adios_obj.declare_io("pio_check")
        engine = io.open(str(filename), adios2.Mode.ReadRandomAccess)
        all_vars = io.available_variables()
        return any(v.startswith("/__pio__/") for v in all_vars)
    except Exception:
        return False
    finally:
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.close()
