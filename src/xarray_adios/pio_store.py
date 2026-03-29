"""PIO store — orchestrates reading of SCORPIO/PIO ADIOS BP files.

PIO writes BP files with a ``__pio__`` namespace:

* ``/__pio__/dim/{name}``   — uint64 scalar per dimension
* ``/__pio__/var/{name}``   — variable data (blocks, one per MPI rank per frame)
* ``/__pio__/decomp/{ioid}`` — decomposition maps (1-based global indices)
* ``/__pio__/track/``       — variable-to-decomposition associations

Reconstruction is handled by helpers in ``_pio_read`` (concat+reshape or
decomposition scatter) with dimension inference in ``_pio_dims``.
"""

from __future__ import annotations

import logging
from typing import Any

import adios2
import numpy as np

from ._common import (
    _NC_TYPE_MAP,
    VariableInfo,
    adios_dtype,
    parse_attr_value,
    parse_block_count,
)
from ._pio_decomp import build_var_decomp_mapping, read_decomp_rank_blocks
from ._pio_dims import dims_from_def, dims_from_def_decomp, infer_dims_and_shape
from ._pio_read import (
    read_blocks_concat,
    read_blocks_decomp,
    try_frame_selective_read,
)

logger = logging.getLogger(__name__)

_PIO_DIM_PREFIX = "/__pio__/dim/"
_PIO_VAR_PREFIX = "/__pio__/var/"


class PioStore:
    """Read PIO-formatted ADIOS BP files.

    Opens the file in ``ReadRandomAccess`` mode and provides methods to
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
        self._decomp_cache: dict[str, list[np.ndarray]] = {}
        self._var_decomp_map: dict[str, str] | None = None
        self._var_defs: dict[str, dict[str, Any]] | None = None
        self._attr_index_built = False
        self._attrs_by_pio: dict[str, dict[str, Any]] = {}
        self._attrs_by_short: dict[str, dict[str, Any]] = {}

    # ── Dimensions ───────────────────────────────────────────────────

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

    # ── Variable definitions (SCORPIO metadata) ──────────────────────

    def _get_var_defs(self) -> dict[str, dict[str, Any]]:
        """Extract SCORPIO ``def/*`` attributes for each variable."""
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
            val = parse_attr_value(ainfo)
            defs.setdefault(var_name, {})[def_key] = val

        self._var_defs = defs
        return defs

    # ── put_var handling ─────────────────────────────────────────────

    def _catalog_putvar(
        self,
        short_name: str,
        pio_name: str,
        var: Any,
        vdef: dict[str, Any],
        dims: dict[str, int],
    ) -> VariableInfo | None:
        """Catalog a ``put_var`` variable stored as raw uint8 bytes."""
        nctype = vdef.get("nctype")
        if nctype is None:
            return None
        real_dtype = _NC_TYPE_MAP.get(int(nctype))
        if real_dtype is None or real_dtype.itemsize == 0:
            return None

        var.set_step_selection((0, 1))
        var.set_block_selection(0)
        raw_count = int(var.count()[0])

        ndims_val = vdef.get("ndims")
        ndims = int(ndims_val) if ndims_val is not None else 1
        header_bytes = max(ndims, 1) * 16
        payload_bytes = raw_count - header_bytes
        if payload_bytes <= 0:
            return None
        total_elements = payload_bytes // real_dtype.itemsize

        def_dims = vdef.get("dims")
        var_dims: tuple[str, ...] | None = None
        var_shape: tuple[int, ...] | None = None
        if def_dims is not None:
            result = dims_from_def(def_dims, total_elements, dims)
            if result is not None:
                var_dims, var_shape = result

        if var_dims is None or var_shape is None:
            var_dims = (f"dim_{short_name}_{total_elements}",)
            var_shape = (total_elements,)

        attrs = self._read_var_attrs(pio_name, short_name)

        return VariableInfo(
            name=short_name,
            pio_name=pio_name,
            dims=var_dims,
            shape=var_shape,
            dtype=np.dtype(real_dtype),
            attrs=attrs,
            nblocks=1,
            block_counts=[raw_count],
            decomp_id=None,
            raw_dtype=np.dtype(np.uint8),
        )

    @staticmethod
    def _fix_putvar_time(variables: dict[str, VariableInfo], ntime: int) -> None:
        """Align the time dimension of put_var variables to *ntime*."""
        for info in variables.values():
            if info.raw_dtype is None or "time" not in info.dims:
                continue
            tidx = info.dims.index("time")
            if info.shape[tidx] != ntime:
                new_shape = list(info.shape)
                new_shape[tidx] = ntime
                info.shape = tuple(new_shape)

    # ── Variable catalog ─────────────────────────────────────────────

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

            dtype = adios_dtype(var)
            if dtype is str:
                continue

            vdef = var_defs.get(short_name, {})

            # Handle put_var variables (raw uint8 bytes with header)
            ncop = vdef.get("ncop")
            is_putvar = dtype == np.uint8 and ncop is not None and str(ncop).strip('"') == "put_var"
            if is_putvar:
                result = self._catalog_putvar(short_name, vname, var, vdef, dims)
                if result is not None:
                    variables[short_name] = result
                continue

            # Block info
            try:
                block_info = self._engine.blocks_info(vname, 0)
            except UnicodeDecodeError:
                logger.debug("Skipping %s: blocks_info raised UnicodeDecodeError", short_name)
                continue
            nblocks = len(block_info)
            block_counts = [parse_block_count(bi["Count"]) for bi in block_info]
            total_elements = sum(block_counts)

            # Scalars
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
                )
                continue

            # Decomp ID
            decomp_id = None
            def_decomp = vdef.get("decomp")
            if def_decomp is not None:
                decomp_id = str(def_decomp).strip('"')
            if decomp_id is None:
                decomp_id = var_decomps.get(short_name)

            # Dimension names and shape
            var_dims: tuple[str, ...] | None = None
            var_shape: tuple[int, ...] | None = None
            def_dims = vdef.get("dims")
            if def_dims is not None:
                if decomp_id is not None:
                    dims_result = dims_from_def_decomp(
                        def_dims,
                        total_elements,
                        dims,
                        decomp_id,
                        self._get_decomp_rank_blocks,
                    )
                else:
                    dims_result = dims_from_def(def_dims, total_elements, dims)
                if dims_result is not None:
                    var_dims, var_shape = dims_result

            if var_dims is None or var_shape is None:
                var_dims, var_shape = infer_dims_and_shape(
                    short_name,
                    total_elements,
                    block_counts,
                    dims,
                    decomp_id=decomp_id,
                    get_decomp_blocks=self._get_decomp_rank_blocks,
                )

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

        # Align put_var time dimension with science variables
        ntime = None
        for info in variables.values():
            if info.raw_dtype is None and "time" in info.dims:
                tidx = info.dims.index("time")
                ntime = info.shape[tidx]
                break
        if ntime is not None:
            self._fix_putvar_time(variables, ntime)

        self._variable_info = variables
        return variables

    # ── Reading variable data ────────────────────────────────────────

    def read_variable(self, name: str, key: tuple | None = None) -> np.ndarray:
        """Read a variable, reconstructing the global array from blocks."""
        info = self.get_variables()[name]

        # Scalar
        if info.shape == ():
            var = self._io.inquire_variable(info.pio_name)
            buf = np.zeros(1, dtype=info.dtype)
            self._engine.get(var, buf)
            self._engine.perform_gets()
            return buf.reshape(())

        # Frame-selective optimisation
        if key is not None and len(info.shape) >= 2:
            result = try_frame_selective_read(
                info,
                key,
                self._engine,
                self._io,
                self._get_decomp_rank_blocks if info.decomp_id else None,
            )
            if result is not None:
                return result

        # Full read
        if info.decomp_id is not None:
            full = read_blocks_decomp(info, self._engine, self._io, self._get_decomp_rank_blocks)
        else:
            full = read_blocks_concat(info, self._engine, self._io)

        if key is not None:
            full = full[key]
        return full

    # ── Decomposition helpers (delegated) ────────────────────────────

    def _get_decomp_rank_blocks(self, ioid: str) -> list[np.ndarray]:
        """Read per-rank decomposition blocks, with caching."""
        return read_decomp_rank_blocks(self._engine, self._io, ioid, self._decomp_cache)

    def _get_var_decomp_mapping(self) -> dict[str, str]:
        """Build variable → decomp ioid mapping, with caching."""
        if self._var_decomp_map is not None:
            return self._var_decomp_map
        self._var_decomp_map = build_var_decomp_mapping(self._all_attrs)
        return self._var_decomp_map

    # ── Attributes ───────────────────────────────────────────────────

    def _ensure_attr_index(self) -> None:
        """Lazily build an index of attributes by variable name."""
        if self._attr_index_built:
            return

        for aname, ainfo in self._all_attrs.items():
            if aname.startswith(_PIO_VAR_PREFIX):
                var_path, _, attr_name = aname.rpartition("/")
                if attr_name:
                    self._attrs_by_pio.setdefault(var_path, {})[attr_name] = parse_attr_value(ainfo)
            else:
                if "/" not in aname:
                    continue
                short_name, _, attr_name = aname.partition("/")
                if attr_name:
                    self._attrs_by_short.setdefault(short_name, {})[attr_name] = parse_attr_value(
                        ainfo
                    )

        self._attr_index_built = True

    def _read_var_attrs(self, pio_name: str, short_name: str) -> dict[str, Any]:
        """Read attributes for a variable."""
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
        _GLOBAL_PREFIX = "/__pio__/global/"

        for aname, ainfo in self._all_attrs.items():
            if aname.startswith(_GLOBAL_PREFIX):
                short = aname[len(_GLOBAL_PREFIX) :]
                attrs[short] = parse_attr_value(ainfo)

        if not attrs:
            var_names = set(self._all_vars.keys())
            short_var_names = {
                v[len(_PIO_VAR_PREFIX) :] for v in self._all_vars if v.startswith(_PIO_VAR_PREFIX)
            }
            for aname, ainfo in self._all_attrs.items():
                if aname.startswith("/__pio__/"):
                    continue
                parts = aname.split("/")
                if len(parts) > 1 and (parts[0] in var_names or parts[0] in short_var_names):
                    continue
                attrs[aname] = parse_attr_value(ainfo)

        return attrs

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the ADIOS engine."""
        self._engine.close()
