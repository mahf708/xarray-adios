# Lazy Loading Design: xarray-adios Backend

## Current Laziness Levels

### 1. Variable-level laziness (fully implemented)

`xr.open_dataset()` reads only metadata (dimensions, attributes, block info).
Variable data is wrapped in `LazilyIndexedArray` and only read when accessed
(e.g., `.values`, `.compute()`, plotting).

**Impact**: Opening a dataset with hundreds of variables is instant. Only the
variables you touch get loaded.

### 2. Frame/time-level laziness (implemented)

When slicing along the time/frame dimension (e.g., `ds["PS"][0]` or
`ds["T"][0:5]`), only the ADIOS blocks for the requested frames are read.

**How it works**: SCORPIO writes one ADIOS block per frame per I/O aggregator.
The backend maps frame indices to block IDs and calls `set_block_selection()`
only for the needed blocks.

**Impact**: For a file with 1000 timesteps, `ds["PS"][0]` reads 1/1000th of
the data instead of all of it.

**Applies to both reconstruction paths**:
- **concat+reshape** (structured grids): block i = frame i when `nblocks == nframes`
- **decomp scatter** (unstructured grids): for separate-block layout, blocks
  `[f*nranks, ..., (f+1)*nranks-1]` belong to frame `f`

### 3. Spatial laziness (not yet implemented)

Currently, once a frame is selected, the full spatial extent is read. E.g.,
`ds["PS"][0, 5:10]` reads all of frame 0, then slices rows 5-10 in memory.

## Why Spatial Laziness Is Hard

### The ADIOS2 API perspective

ADIOS2 offers three selection mechanisms:

| API | What it does | Works for PIO data? |
|-----|-------------|---------------------|
| `set_step_selection([start, count])` | Read specific ADIOS steps | No -- PIO packs all frames into step 0 |
| `set_selection([[start], [count]])` | Read rectangular sub-region of a global array | No -- PIO uses local blocks, not global arrays |
| `set_block_selection(bid)` | Read a specific block entirely | Yes -- already used for frame selection |

### The SCORPIO write model

SCORPIO writes every variable as a **1D local variable** in ADIOS2:

```c
// pio_darray.cpp:1299-1301
av->adios_varid = spio_define_adios2_variable(...,
    1,        // ndims = 1 (always 1D)
    NULL,     // global shape = NULL (local variable, not global array)
    NULL,     // global offset = NULL
    &av_count // local count only
);
```

Because there is no global shape defined, ADIOS2's `set_selection` (sub-region
reads) cannot be used at read time. Each block is an opaque 1D blob that must
be read entirely.

### The decomposition challenge

For unstructured grids (e.g., `ncol`), the decomp map scatters data
non-contiguously:

```
block[0] -> global[4]     (not contiguous!)
block[1] -> global[132]
block[2] -> global[260]
block[3] -> global[53]
...
```

Even if partial block reads were possible, a spatial slice like
`global[100:200]` requires reading scattered positions from the block.

## Path Forward: Spatial Laziness via Block Granularity

### The knob: `num_iotasks` / `pio_stride`

SCORPIO groups MPI ranks into I/O aggregator groups. The number of groups
(= number of I/O tasks) determines how many blocks are written per frame:

| E3SM config | Blocks per frame | Spatial granularity |
|-------------|-----------------|---------------------|
| `pio_stride=128` (128 tasks) | 1 | None (all spatial data in one block) |
| `pio_stride=32` (128 tasks) | 4 | ~1/4 of spatial domain per block |
| `pio_stride=1` (128 tasks) | 128 | Per-rank granularity |

This is configured in E3SM via `env_run.xml`:

```xml
<entry id="PIO_NUMTASKS">...</entry>  <!-- or computed from PIO_STRIDE -->
<entry id="PIO_STRIDE">...</entry>
```

Or in the component-level `*_modelio.nml`:

```fortran
pio_stride = 32   ! 128 tasks / 32 stride = 4 I/O aggregators
```

### What the backend could do (future work)

With multiple blocks per frame, the backend could:

1. **Build a block-to-region index** from the decomp map at open time.
   For each block, record which global indices it covers.

2. **For structured grids**: Detect contiguous lat-band patterns. E.g.,
   block 0 covers `lat[0:4]`, block 1 covers `lat[4:8]`, etc.

3. **Given a spatial slice**: Identify which blocks overlap the requested
   region, read only those blocks, scatter only the needed elements.

### What SCORPIO could do (aspirational)

For structured grids where the decomposition IS rectangular (after
horiz_remap), SCORPIO could define ADIOS2 variables with actual
multi-dimensional global shapes:

```c
// Aspirational: global-array writes for structured grids
size_t global_shape[] = {ntime, nlat, nlon};
size_t local_start[]  = {frame, lat_start, 0};
size_t local_count[]  = {1, lat_count, nlon};

adios2_define_variable(..., 3, global_shape, local_start, local_count, ...);
```

This would enable ADIOS2's native `set_selection` for rectangular sub-region
reads at the reader side -- no decomp map needed, no scatter, just direct
sub-array access. The relevant SCORPIO code is in:

- **Variable definition**: `externals/scorpio/src/clib/pio_darray.cpp:1299-1301`
- **Write with selection**: `externals/scorpio/src/clib/pio_darray.cpp:1963`
- **Aggregation grouping**: `externals/scorpio/src/clib/pioc.cpp:1182-1197`

For unstructured grids, the 1D-block approach would remain necessary since
the decomposition is inherently non-rectangular.
