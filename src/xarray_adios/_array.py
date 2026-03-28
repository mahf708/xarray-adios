"""Lazy-loading BackendArray for ADIOS variables."""

from __future__ import annotations

import numpy as np
from xarray.backends.common import BackendArray
from xarray.core import indexing


class AdiosBackendArray(BackendArray):
    """Lazy array that reads from an ADIOS store on demand.

    Works with both PioStore and AdiosStore — any object implementing
    ``read_variable(name, key=None) -> np.ndarray``.
    """

    __slots__ = ("_store", "_var_name", "shape", "dtype", "_lock")

    def __init__(self, store, var_name: str, shape: tuple[int, ...], dtype: np.dtype, lock):
        self._store = store
        self._var_name = var_name
        self.shape = shape
        self.dtype = dtype
        self._lock = lock

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key,
            self.shape,
            indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple) -> np.ndarray:
        with self._lock:
            return self._store.read_variable(self._var_name, key)
