"""Xarray backend entrypoint for ADIOS BP files.

Usage::

    import xarray as xr
    ds = xr.open_dataset("file.bp", engine="adios")

Automatically detects whether the file uses PIO namespace (E3SM/SCORPIO
output) or is a generic ADIOS BP file, and selects the appropriate store.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import xarray as xr
from xarray.backends.common import BackendArray, BackendEntrypoint
from xarray.core import indexing

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

from ._common import is_pio_file
from .adios_store import AdiosStore
from .pio_store import PioStore

logger = logging.getLogger(__name__)


# ── Lazy array wrapper (inlined from former _array.py) ───────────────


class AdiosBackendArray(BackendArray):
    """Lazy-loading array that defers I/O to the underlying store.

    Works with both ``PioStore`` and ``AdiosStore`` — any object
    implementing ``read_variable(name, key=None)``.
    """

    __slots__ = ("_store", "_var_name", "shape", "dtype", "_lock")

    def __init__(self, store, var_name: str, shape: tuple, dtype: np.dtype, lock):
        self._store = store
        self._var_name = var_name
        self.shape = shape
        self.dtype = dtype
        self._lock = lock

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._raw_indexing_method
        )

    def _raw_indexing_method(self, key):
        with self._lock:
            return self._store.read_variable(self._var_name, key)


# ── Backend entrypoint ───────────────────────────────────────────────


class AdiosBackendEntrypoint(BackendEntrypoint):
    """Xarray backend for reading ADIOS BP files.

    Supports both PIO-formatted files (E3SM/SCORPIO output with
    ``__pio__`` namespace) and generic ADIOS BP files.
    """

    description = "Open ADIOS BP files (.bp) in xarray"
    open_dataset_parameters = (
        "filename_or_obj",
        "drop_variables",
        "mask_and_scale",
        "decode_times",
    )

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables: str | Iterable[str] | None = None,
        mask_and_scale: bool = True,
        decode_times: bool = True,
    ) -> xr.Dataset:
        filename = str(filename_or_obj)

        if isinstance(drop_variables, str):
            drop_variables = {drop_variables}
        elif drop_variables is not None:
            drop_variables = set(drop_variables)
        else:
            drop_variables = set()

        # Detect file type and open the appropriate store
        store: PioStore | AdiosStore
        if is_pio_file(filename):
            logger.info("Detected PIO format: %s", filename)
            store = PioStore(filename)
        else:
            logger.info("Using generic ADIOS store: %s", filename)
            store = AdiosStore(filename)

        lock = threading.Lock()

        # Build variables
        variables = {}
        var_infos = store.get_variables()

        for name, info in var_infos.items():
            if name in drop_variables:
                continue

            backend_array = AdiosBackendArray(
                store=store,
                var_name=name,
                shape=info.shape,
                dtype=info.dtype,
                lock=lock,
            )
            lazy_data = indexing.LazilyIndexedArray(backend_array)

            # Cast _FillValue to variable dtype so xr.decode_cf compares correctly
            attrs = dict(info.attrs)
            if "_FillValue" in attrs:
                attrs["_FillValue"] = info.dtype.type(attrs["_FillValue"])

            var = xr.Variable(dims=info.dims, data=lazy_data, attrs=attrs)
            variables[name] = var

        # Separate coordinates from data variables
        coord_names = set(store.get_dimensions().keys()) if isinstance(store, PioStore) else set()

        coords = {n: v for n, v in variables.items() if n in coord_names}
        data_vars = {n: v for n, v in variables.items() if n not in coord_names}
        attrs = store.get_global_attrs()

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Apply CF decoding (mask/scale + time) via xarray's built-in
        if mask_and_scale or decode_times:
            ds = xr.decode_cf(
                ds,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
            )

        # Register close callback
        ds.encoding["source"] = filename
        ds.encoding["_adios_store"] = store

        if hasattr(ds, "set_close"):
            ds.set_close(store.close)
        else:
            ds._close = store.close  # type: ignore[attr-defined]

        return ds

    def guess_can_open(self, filename_or_obj) -> bool:
        try:
            path = str(filename_or_obj)
        except Exception:
            return False
        return path.endswith((".bp", ".bp4", ".bp5"))
