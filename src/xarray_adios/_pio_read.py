"""Block reading and array reconstruction for PIO variables.

PIO stores variable data as per-rank blocks.  This module provides
functions to read those blocks and reconstruct the global array using
either concat+reshape (for regular grids) or index-scatter (for
decomposed / unstructured grids).
"""

from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from ._pio_decomp import _read_selected_blocks, detect_nframes

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._common import VariableInfo


# ── Concat + reshape (regular grids) ────────────────────────────────


def read_blocks_concat(
    info: VariableInfo,
    engine: Any,
    io: Any,
) -> np.ndarray:
    """Read all blocks and reconstruct via concatenation and reshape."""
    read_dtype = info.raw_dtype if info.raw_dtype is not None else info.dtype
    blocks = _read_selected_blocks(
        engine, io, info.pio_name, list(range(info.nblocks)), info.block_counts, read_dtype
    )
    flat = np.concatenate(blocks)

    if info.raw_dtype is not None:
        # put_var header: ndims * 16 bytes (ndims * 2 int64 values)
        ndims = max(len(info.shape), 1)
        header_bytes = ndims * 16
        flat = flat[header_bytes:].view(info.dtype)
        target_size = int(np.prod(info.shape)) if info.shape else 1
        if flat.size < target_size and flat.size > 0:
            reps = target_size // flat.size
            flat = np.tile(flat, reps)[:target_size]

    try:
        return flat.reshape(info.shape)
    except ValueError:
        warnings.warn(
            f"Variable '{info.name}': cannot reshape {flat.shape} to {info.shape}, "
            f"returning flat array.",
            stacklevel=2,
        )
        return flat


# ── Decomposition scatter (unstructured grids) ──────────────────────


def read_blocks_decomp(
    info: VariableInfo,
    engine: Any,
    io: Any,
    get_decomp_blocks: Callable[[str], list[np.ndarray]],
) -> np.ndarray:
    """Read blocks and reconstruct using a decomposition map (scatter).

    Each MPI rank wrote a subset of the global array.  The decomposition
    map contains 1-based global indices.  Data is scattered into a
    pre-allocated global array.
    """
    assert info.decomp_id is not None
    decomp_rank_blocks = get_decomp_blocks(info.decomp_id)

    data_blocks = _read_selected_blocks(
        engine, io, info.pio_name, list(range(info.nblocks)), info.block_counts, info.dtype
    )

    all_indices = np.concatenate(decomp_rank_blocks)
    spatial_size = int(np.max(all_indices))

    nranks = len(decomp_rank_blocks)
    ndata_blocks = len(data_blocks)
    nframes = detect_nframes(data_blocks, decomp_rank_blocks, nranks, ndata_blocks)

    if nframes > 1 and ndata_blocks == nranks:
        # Embedded frames: each data block has nframes * decomp_count elements
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
        # Separate blocks per frame
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


# ── Frame-selective reading (optimisation) ───────────────────────────


def try_frame_selective_read(
    info: VariableInfo,
    key: tuple,
    engine: Any,
    io: Any,
    get_decomp_blocks: Callable[[str], list[np.ndarray]] | None,
) -> np.ndarray | None:
    """Read only the requested frames, returning None if not applicable.

    This avoids reading all blocks when only a subset of frames (the
    leading dimension) is requested.
    """
    nframes = info.shape[0]

    frame_key = key[0]
    remaining_key = key[1:]

    if isinstance(frame_key, (int, np.integer)):
        frame_indices = [int(frame_key)]
        squeeze_frame = True
    elif isinstance(frame_key, slice):
        frame_indices = list(range(*frame_key.indices(nframes)))
        squeeze_frame = False
        if len(frame_indices) == nframes:
            return None  # full read — no benefit
    else:
        return None  # fancy indexing — fall back

    if info.decomp_id is not None and get_decomp_blocks is not None:
        return _read_frames_decomp(
            info,
            frame_indices,
            squeeze_frame,
            remaining_key,
            engine,
            io,
            get_decomp_blocks,
        )
    return _read_frames_concat(info, frame_indices, squeeze_frame, remaining_key, engine, io)


def _read_frames_concat(
    info: VariableInfo,
    frame_indices: list[int],
    squeeze_frame: bool,
    remaining_key: tuple,
    engine: Any,
    io: Any,
) -> np.ndarray | None:
    """Read selected frames for a concat+reshape variable."""
    nframes = info.shape[0]
    spatial_shape = info.shape[1:]
    spatial_size = 1
    for s in spatial_shape:
        spatial_size *= s

    if info.nblocks != nframes:
        return None

    if not all(c == spatial_size for c in info.block_counts):
        return None

    selected = _read_selected_blocks(
        engine, io, info.pio_name, frame_indices, info.block_counts, info.dtype
    )

    frames = []
    for block in selected:
        try:
            frames.append(block.reshape(spatial_shape))
        except ValueError:
            return None

    result = np.stack(frames, axis=0)

    if squeeze_frame:
        result = result[0]

    if remaining_key and any(k != slice(None) for k in remaining_key):
        result = result[remaining_key]

    return result


def _read_frames_decomp(
    info: VariableInfo,
    frame_indices: list[int],
    squeeze_frame: bool,
    remaining_key: tuple,
    engine: Any,
    io: Any,
    get_decomp_blocks: Callable[[str], list[np.ndarray]],
) -> np.ndarray | None:
    """Read selected frames for a decomp-reconstructed variable."""
    assert info.decomp_id is not None
    decomp_rank_blocks = get_decomp_blocks(info.decomp_id)
    all_indices = np.concatenate(decomp_rank_blocks)
    spatial_size = int(np.max(all_indices))

    nranks = len(decomp_rank_blocks)
    nframes_total = info.shape[0]

    if info.nblocks == nranks * nframes_total:
        # Separate blocks per frame — read only needed blocks
        result = np.zeros((len(frame_indices), spatial_size), dtype=info.dtype)
        for out_f, src_f in enumerate(frame_indices):
            block_ids = list(range(src_f * nranks, (src_f + 1) * nranks))
            frame_blocks = _read_selected_blocks(
                engine, io, info.pio_name, block_ids, info.block_counts, info.dtype
            )
            for r in range(nranks):
                dmap = decomp_rank_blocks[r]
                valid = dmap > 0
                result[out_f, dmap[valid] - 1] = frame_blocks[r][valid]

    elif info.nblocks == nranks:
        # Embedded frames — read all blocks, scatter requested frames
        all_data = _read_selected_blocks(
            engine, io, info.pio_name, list(range(info.nblocks)), info.block_counts, info.dtype
        )
        result = np.zeros((len(frame_indices), spatial_size), dtype=info.dtype)
        for r in range(nranks):
            dmap = decomp_rank_blocks[r]
            data = all_data[r]
            stride = len(dmap)
            valid = dmap > 0
            for out_f, src_f in enumerate(frame_indices):
                frame_data = data[src_f * stride : (src_f + 1) * stride]
                result[out_f, dmap[valid] - 1] = frame_data[valid]
    else:
        return None

    spatial_shape = info.shape[1:]
    if spatial_shape != (spatial_size,):
        with contextlib.suppress(ValueError):
            result = result.reshape((len(frame_indices), *spatial_shape))

    if squeeze_frame:
        result = result[0]

    if remaining_key and any(k != slice(None) for k in remaining_key):
        result = result[remaining_key]

    return result
