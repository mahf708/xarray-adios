"""Xarray backend entrypoint for ADIOS BP files.

Usage::

    import xarray as xr
    ds = xr.open_dataset("file.bp", engine="adios")

Automatically detects whether the file uses PIO namespace (E3SM/SCORPIO output)
or is a generic ADIOS BP file, and selects the appropriate store.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import xarray as xr
from xarray.backends.common import BackendEntrypoint
from xarray.core import indexing

if TYPE_CHECKING:
    from collections.abc import Iterable

from ._array import AdiosBackendArray
from .adios_store import AdiosStore
from .pio_store import PioStore, is_pio_file

logger = logging.getLogger(__name__)


class AdiosBackendEntrypoint(BackendEntrypoint):
    """Xarray backend for reading ADIOS BP files.

    Supports both PIO-formatted files (E3SM/SCORPIO output with ``__pio__``
    namespace) and generic ADIOS BP files. PIO format is auto-detected.

    Parameters for ``open_dataset``
    -------------------------------
    filename_or_obj : str or path-like
        Path to the ``.bp`` file or directory.
    drop_variables : str or iterable of str, optional
        Variables to exclude from the dataset.
    mask_and_scale : bool, default True
        Apply CF masking (``_FillValue``) and scaling (``scale_factor``,
        ``add_offset``).
    decode_times : bool, default True
        Decode time variables using CF conventions.
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

            var = xr.Variable(
                dims=info.dims,
                data=lazy_data,
                attrs=info.attrs,
            )
            variables[name] = var

        # Separate coordinates from data variables
        coords = {}
        data_vars = {}

        if isinstance(store, PioStore):
            dims = store.get_dimensions()
            coord_names = set(dims.keys())
        else:
            dims = store.get_dimensions()
            coord_names = set()

        for name, var in variables.items():
            if name in coord_names:
                coords[name] = var
            else:
                data_vars[name] = var

        # Global attributes
        attrs = store.get_global_attrs()

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Apply CF decoding
        if mask_and_scale or decode_times:
            ds = _decode_cf(ds, mask_and_scale=mask_and_scale, decode_times=decode_times)

        # Store a reference to the store and register a close callback so that
        # Dataset.close() (or using the Dataset as a context manager) will
        # properly close the underlying ADIOS engine.
        ds.encoding["source"] = filename
        ds.encoding["_adios_store"] = store

        if hasattr(ds, "set_close"):
            ds.set_close(store.close)  # xarray >= 0.16
        else:
            ds._close = store.close  # type: ignore[attr-defined]

        return ds

    def guess_can_open(self, filename_or_obj) -> bool:
        try:
            path = str(filename_or_obj)
        except Exception:
            return False
        return path.endswith((".bp", ".bp4", ".bp5"))


def _decode_cf(
    ds: xr.Dataset,
    mask_and_scale: bool = True,
    decode_times: bool = True,
) -> xr.Dataset:
    """Apply CF conventions decoding to variables."""
    import numpy as np

    if mask_and_scale:
        new_vars = {}
        for name, var in ds.variables.items():
            attrs = dict(var.attrs)
            fill_value = attrs.pop("_FillValue", None)
            missing = attrs.pop("missing_value", None)
            scale = attrs.pop("scale_factor", None)
            offset = attrs.pop("add_offset", None)

            encoding = {}
            data = var.data

            if fill_value is not None or missing is not None:
                fv = fill_value if fill_value is not None else missing
                encoding["_FillValue"] = fv
                if np.issubdtype(var.dtype, np.floating):
                    vals = np.array(data)
                    vals = np.where(vals == fv, np.nan, vals)
                    data = vals

            if scale is not None or offset is not None:
                encoding["scale_factor"] = scale
                encoding["add_offset"] = offset
                vals = np.array(data, dtype=np.float64)
                if scale is not None:
                    vals = vals * scale
                if offset is not None:
                    vals = vals + offset
                data = vals

            new_vars[name] = xr.Variable(
                dims=var.dims, data=data, attrs=attrs, encoding=encoding
            )

        ds = ds.assign(new_vars)

    if decode_times:
        try:
            ds = xr.decode_cf(ds, decode_times=True, mask_and_scale=False)
        except Exception as e:
            logger.debug("CF time decoding failed: %s", e)

    return ds
