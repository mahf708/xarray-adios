"""Dimension inference for PIO variables.

When SCORPIO ``def/dims`` metadata is present, dimensions are resolved
directly.  Otherwise a simple heuristic matches element counts against
known dimension sizes, falling back to a flat 1-D shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._common import parse_string_array

if TYPE_CHECKING:
    from collections.abc import Callable


def dims_from_def(
    def_dims_val: Any,
    total_elements: int,
    dims: dict[str, int],
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    """Resolve dimensions from a SCORPIO ``def/dims`` attribute.

    Returns ``(dim_names, shape)`` or *None* if the attribute cannot be
    parsed or contains multiple unknowns.
    """
    dim_names = parse_string_array(def_dims_val)
    if not dim_names:
        return None

    shape: list[int] = []
    unknown_idx: int | None = None

    for i, d in enumerate(dim_names):
        size = dims.get(d, -1)
        if size > 0:
            shape.append(size)
        elif d == "time" or (d in dims and size == 0):
            # Unlimited / explicitly-zero dimension — infer from data
            if unknown_idx is not None:
                return None  # multiple unknowns
            unknown_idx = i
            shape.append(0)
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


def dims_from_def_decomp(
    def_dims_val: Any,
    total_elements: int,
    dims: dict[str, int],
    decomp_id: str,
    get_decomp_blocks: Callable[[str], list[np.ndarray]],
) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    """Resolve dimensions for a decomposed variable using ``def/dims``.

    The raw *total_elements* includes MPI padding and does not divide
    evenly by global dimension sizes.  The spatial extent is derived
    from the decomposition map instead.
    """
    dim_names = parse_string_array(def_dims_val)
    if not dim_names:
        return None

    decomp_blocks = get_decomp_blocks(decomp_id)
    all_indices = np.concatenate(decomp_blocks)
    spatial_size = int(np.max(all_indices))  # 1-based
    decomp_total = sum(len(b) for b in decomp_blocks)

    # Compute number of frames
    if decomp_total > 0 and total_elements > decomp_total and total_elements % decomp_total == 0:
        nframes = total_elements // decomp_total
    else:
        nframes = 1

    shape: list[int] = []
    unknown_idx: int | None = None
    known_spatial_product = 1

    for i, d in enumerate(dim_names):
        size = dims.get(d, -1)
        if d == "time" or size == 0:
            shape.append(nframes)
        elif size > 0:
            shape.append(size)
            known_spatial_product *= size
        else:
            if unknown_idx is not None:
                return None
            unknown_idx = i
            shape.append(0)

    if unknown_idx is not None:
        if known_spatial_product > 0 and spatial_size % known_spatial_product == 0:
            shape[unknown_idx] = spatial_size // known_spatial_product
        else:
            return None

    return tuple(dim_names), tuple(shape)


def _find_dim_by_size(
    size: int, dims: dict[str, int], exclude: set[str] | None = None
) -> str | None:
    """Return the first dimension name matching *size*, or None."""
    exclude = exclude or set()
    for dname, dsize in dims.items():
        if dname not in exclude and dsize == size:
            return dname
    return None


def infer_dims_and_shape(
    var_name: str,
    total_elements: int,
    block_counts: list[int],
    dims: dict[str, int],
    decomp_id: str | None,
    get_decomp_blocks: Callable[[str], list[np.ndarray]] | None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Infer dimension names and shape when ``def/dims`` metadata is absent.

    Strategy (simplified):
    1. Decomp-aware: derive spatial size from the map, detect frames.
    2. Coordinate match: 1-D variable matching a known dimension size.
    3. Time x spatial: 2-D with a ``time`` leading dimension.
    4. Fallback: flat 1-D with a generated dimension name.
    """
    if total_elements == 0:
        return ("_empty",), (0,)

    if total_elements == 1 and len(block_counts) == 1:
        return (), ()

    # ── decomp-aware path ────────────────────────────────────────────
    if decomp_id is not None and get_decomp_blocks is not None:
        return _infer_dims_decomp(var_name, total_elements, dims, decomp_id, get_decomp_blocks)

    # ── coordinate variable (1-D, matches a dimension size) ─────────
    for dname, dsize in dims.items():
        if total_elements == dsize:
            return (dname,), (dsize,)

    # ── 2-D: time x spatial ─────────────────────────────────────────
    if "time" in dims:
        ntime = dims["time"]
        if ntime > 0 and total_elements % ntime == 0:
            spatial = total_elements // ntime
            spatial_dim = _find_dim_by_size(spatial, dims, exclude={"time"})
            if spatial_dim:
                return ("time", spatial_dim), (ntime, spatial)

    # ── fallback: flat 1-D ──────────────────────────────────────────
    return (f"dim_{var_name}_{total_elements}",), (total_elements,)


def _infer_dims_decomp(
    var_name: str,
    total_elements: int,
    dims: dict[str, int],
    decomp_id: str,
    get_decomp_blocks: Callable[[str], list[np.ndarray]],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Infer dims/shape for a variable that uses a decomposition map."""
    decomp_blocks = get_decomp_blocks(decomp_id)
    all_indices = np.concatenate(decomp_blocks)
    spatial_size = int(np.max(all_indices))  # 1-based

    decomp_total = sum(len(b) for b in decomp_blocks)
    spatial_dim = _find_dim_by_size(spatial_size, dims) or f"ncol_{spatial_size}"

    if decomp_total > 0 and total_elements > decomp_total and total_elements % decomp_total == 0:
        nframes = total_elements // decomp_total
        if "time" in dims and dims["time"] == nframes:
            time_dim: str | None = "time"
        else:
            time_dim = _find_dim_by_size(nframes, dims)
        if time_dim:
            return (time_dim, spatial_dim), (nframes, spatial_size)
        return (f"frame_{nframes}", spatial_dim), (nframes, spatial_size)

    return (spatial_dim,), (spatial_size,)
