"""Decomposition map handling for PIO variables.

PIO/SCORPIO distributes array data across MPI ranks using decomposition
maps stored under ``/__pio__/decomp/{ioid}``.  Each map is a 1-D array
of 1-based global indices indicating where each local element belongs in
the reconstructed global array.

This module provides helpers to read decomposition maps and to discover
which decomposition a variable uses (attribute-based and track-based
strategies).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._common import adios_dtype, parse_attr_value, parse_block_count

logger = logging.getLogger(__name__)

_PIO_VAR_PREFIX = "/__pio__/var/"
_PIO_DECOMP_PREFIX = "/__pio__/decomp/"
_PIO_TRACK_PREFIX = "/__pio__/track/"


def read_decomp_rank_blocks(
    engine: Any,
    io: Any,
    ioid: str,
    cache: dict[str, list[np.ndarray]],
) -> list[np.ndarray]:
    """Read per-rank decomposition map blocks for *ioid*.

    Returns a list of 1-D int64 arrays (one per MPI rank) containing
    1-based global indices.  Results are cached in *cache*.
    """
    if ioid in cache:
        return cache[ioid]

    pio_name = f"{_PIO_DECOMP_PREFIX}{ioid}"
    var = io.inquire_variable(pio_name)
    if var is None:
        raise KeyError(f"Decomposition map not found: {pio_name}")

    try:
        bi = engine.blocks_info(pio_name, 0)
    except UnicodeDecodeError as exc:
        raise KeyError(
            f"Cannot read decomposition map {pio_name}: blocks_info raised UnicodeDecodeError"
        ) from exc

    nblocks = len(bi)
    counts = [parse_block_count(b["Count"]) for b in bi]
    dtype = adios_dtype(var)
    if dtype is str:
        dtype = np.int64
    np_dtype = np.dtype(dtype)

    blocks = _read_selected_blocks(engine, io, pio_name, list(range(nblocks)), counts, np_dtype)
    cache[ioid] = blocks
    return blocks


def get_decomp_ids(all_vars: dict[str, Any]) -> list[str]:
    """Return the ioid strings for all decomposition maps in the file."""
    return [
        vname[len(_PIO_DECOMP_PREFIX) :]
        for vname in all_vars
        if vname.startswith(_PIO_DECOMP_PREFIX)
    ]


def build_var_decomp_mapping(
    all_attrs: dict[str, Any],
) -> dict[str, str]:
    """Build a mapping of variable short-name to decomp ioid.

    Uses two discovery strategies (in priority order):

    1. **Attribute-based**: ``{pio_name}/_pio_decomp`` attribute.
    2. **Track-based**: ``/__pio__/track/{varname}`` attribute.
    """
    mapping: dict[str, str] = {}

    # Strategy 1: attribute-based
    for aname, ainfo in all_attrs.items():
        if aname.endswith("/_pio_decomp"):
            val = parse_attr_value(ainfo)
            prefix = aname[: -len("/_pio_decomp")]
            short = prefix[len(_PIO_VAR_PREFIX) :] if prefix.startswith(_PIO_VAR_PREFIX) else prefix
            mapping[short] = str(val)

    # Strategy 2: track-based
    for aname, ainfo in all_attrs.items():
        if not aname.startswith(_PIO_TRACK_PREFIX):
            continue
        suffix = aname[len(_PIO_TRACK_PREFIX) :]
        if "/" not in suffix and suffix not in mapping:
            val = parse_attr_value(ainfo)
            mapping[suffix] = str(val)

    return mapping


def detect_nframes(
    data_blocks: list[np.ndarray],
    decomp_blocks: list[np.ndarray],
    nranks: int,
    ndata_blocks: int,
) -> int:
    """Detect how many frames (timesteps) a variable spans.

    Returns 1 for single-frame or spatial-only variables.
    """
    if ndata_blocks > nranks and ndata_blocks % nranks == 0:
        return ndata_blocks // nranks

    if ndata_blocks == nranks and nranks > 0:
        ratios: list[int] = []
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


# ── Low-level block reader (shared with _pio_read) ──────────────────


def _read_selected_blocks(
    engine: Any,
    io: Any,
    pio_name: str,
    block_ids: list[int],
    block_counts: list[int],
    dtype: np.dtype,
) -> list[np.ndarray]:
    """Read specific blocks by ID and return them as a list."""
    var = io.inquire_variable(pio_name)
    var.set_step_selection((0, 1))

    blocks: list[np.ndarray] = []
    for bid in block_ids:
        var.set_block_selection(bid)
        count = block_counts[bid]
        block = np.zeros(count, dtype=dtype)
        engine.get(var, block)
        engine.perform_gets()
        blocks.append(block)

    return blocks
