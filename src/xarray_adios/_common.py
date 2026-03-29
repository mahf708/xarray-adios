"""Shared types, constants, and parsing utilities for xarray-adios.

This module centralises helpers used by both the PIO store and the
generic ADIOS store so that neither module depends on the other.
"""

from __future__ import annotations

import contextlib
from typing import Any

import adios2
import numpy as np

# ── ADIOS type string → numpy dtype ─────────────────────────────────

_ADIOS_TYPE_MAP: dict[str, type] = {
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

# ── NetCDF type code → numpy dtype (for put_var deserialization) ─────

_NC_TYPE_MAP: dict[int, np.dtype] = {
    1: np.dtype(np.int8),
    2: np.dtype("S1"),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.float32),
    6: np.dtype(np.float64),
    7: np.dtype(np.uint8),
    8: np.dtype(np.uint16),
    9: np.dtype(np.uint32),
    10: np.dtype(np.int64),
    11: np.dtype(np.uint64),
}


# ── Variable metadata container ─────────────────────────────────────


class VariableInfo:
    """Metadata about a PIO variable.

    Attributes
    ----------
    name : str
        Short variable name (e.g. ``"PS"``).
    pio_name : str
        Full ADIOS path (e.g. ``"/__pio__/var/PS"``).
    dims : tuple[str, ...]
        Dimension names in order.
    shape : tuple[int, ...]
        Dimension sizes in order.
    dtype : np.dtype
        Element data type.
    attrs : dict
        CF / user attributes.
    nblocks : int
        Number of ADIOS blocks.
    block_counts : list[int]
        Element count per block.
    decomp_id : str | None
        Associated decomposition map ioid, if any.
    raw_dtype : np.dtype | None
        For put_var variables: the on-disk dtype (uint8) before conversion.
    """

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
        "raw_dtype",
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
        raw_dtype: np.dtype | None = None,
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
        self.raw_dtype = raw_dtype


# ── Parsing helpers ──────────────────────────────────────────────────


def adios_dtype(var: Any) -> type:
    """Return the numpy dtype for an ADIOS variable."""
    type_str = var.type()
    return _ADIOS_TYPE_MAP.get(type_str, np.float64)


def parse_attr_value(attr_info: dict[str, str]) -> Any:
    """Convert an ADIOS attribute info dict to a Python value.

    The *attr_info* dict comes from ``io.available_attributes()`` and
    contains keys ``Type``, ``Value``, and ``Elements``.
    """
    value = attr_info.get("Value", "")
    atype = attr_info.get("Type", "")
    nelements = int(attr_info.get("Elements", "1"))

    if atype == "string":
        if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        return value
    elif atype in ("float", "double"):
        if nelements > 1 or (isinstance(value, str) and value.startswith("{")):
            items = value.strip("{ }").split(",")
            return np.array([float(x.strip()) for x in items if x.strip()])
        return float(value)
    elif "int" in atype:
        if nelements > 1 or (isinstance(value, str) and value.startswith("{")):
            items = value.strip("{ }").split(",")
            return np.array([int(x.strip()) for x in items if x.strip()])
        return int(value)
    else:
        return value


def parse_block_count(count_value: str | int) -> int:
    """Parse a block ``Count`` value from ``blocks_info``.

    ADIOS2 reports ``Count`` as either a single integer (``"100"``) or a
    comma-separated shape (``"10,20"``).  Returns the total element count.
    """
    s = str(count_value).strip()
    if "," in s:
        parts = [int(p) for p in s.split(",") if p.strip()]
        result = 1
        for p in parts:
            result *= p
        return result
    return int(s)


def parse_string_array(value: Any) -> tuple[str, ...]:
    """Parse a SCORPIO string-array attribute into a tuple of strings.

    Handles formats like ``'{ "time", "ncol" }'`` and ``'"ncol"'``.
    """
    if not isinstance(value, str):
        return ()
    s = value.strip()
    if s.startswith("{"):
        inner = s.strip("{ }")
        return tuple(p.strip().strip('"') for p in inner.split(",") if p.strip())
    return (s.strip('"'),)


def is_pio_file(filename: str) -> bool:
    """Return True if a BP file was written by PIO (has ``__pio__`` namespace)."""
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
