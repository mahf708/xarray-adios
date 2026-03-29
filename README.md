# xarray-adios

Thin [xarray](https://docs.xarray.dev/) backend for reading
[ADIOS2](https://adios2.readthedocs.io/) BP files, with first-class support
for the PIO/SCORPIO format used by [E3SM](https://e3sm.org/) and other
climate models.

```python
import xarray as xr

ds = xr.open_dataset("output.bp", engine="adios")
```

## What it does

* Registers an xarray backend via the `xarray.backends` entry point, so
  `engine="adios"` works out of the box after install.
* Auto-detects whether a `.bp` file uses the PIO `__pio__/` namespace or is
  a generic ADIOS file, and selects the right reader.
* Lazy-loads variable data -- metadata is read on open, array data is read
  only when `.values` or `.compute()` is called.
* Applies CF conventions (masking, scaling, time decoding) via
  `xr.decode_cf`.
* Supports PIO-specific features:
  * Decomposition-based scatter reconstruction for unstructured grids
    (e.g. spectral-element `ncol` data).
  * Frame-selective reading -- only the requested timesteps are read from
    disk.
  * SCORPIO `def/*` metadata for dimension and type resolution.
  * `put_var` byte-stream deserialization (coordinates, time bounds, etc.).

## Installation

Requires Python >= 3.10.

```bash
pip install .
```

Or for development:

```bash
pip install -e ".[dev]"
```

Dependencies: `xarray`, `numpy`, `adios2`.

## Architecture

The package is split into focused modules, each under ~300 lines:

```
src/xarray_adios/
├── __init__.py          Package entry point
├── _common.py           Shared types, constants, parsing utilities
├── _pio_dims.py         Dimension inference (def/dims metadata + fallback)
├── _pio_decomp.py       Decomposition map reading and variable association
├── _pio_read.py         Block reading and array reconstruction
├── pio_store.py         PioStore -- orchestrator for PIO-formatted files
├── adios_store.py       AdiosStore -- reader for generic ADIOS BP files
└── backend.py           Xarray backend entry point, lazy array wrapper
```

**Data flow**: `backend.py` detects file type via `is_pio_file()`, creates
the appropriate store (`PioStore` or `AdiosStore`), wraps each variable in
a `LazilyIndexedArray`, and passes the dataset through `xr.decode_cf`.

### How PIO format works

SCORPIO/PIO writes ADIOS BP files with a `__pio__/` namespace:

| Path | Purpose |
|------|---------|
| `/__pio__/dim/{name}` | Dimension sizes (uint64 scalars) |
| `/__pio__/var/{name}` | Variable data as per-rank blocks |
| `/__pio__/decomp/{ioid}` | Decomposition maps (1-based global indices) |
| `/__pio__/track/{varname}` | Variable-to-decomposition associations |
| `/__pio__/global/{name}` | Global attributes |
| `/__pio__/var/{name}/def/*` | SCORPIO metadata (dims, decomp, nctype) |

### Reconstruction strategies

1. **Concat + reshape** (regular grids): concatenate all blocks into a flat
   array, then reshape to the inferred dimensions. Used for structured
   lat-lon output.
2. **Decomposition scatter** (unstructured grids): use the decomposition map
   to scatter each rank's data into the correct global positions. Required
   for native spectral-element grids (e.g. `ne30pg2`).

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite includes:

* **Unit tests** (`test_common.py`, `test_pio_dims.py`, `test_pio_decomp.py`):
  test parsing utilities, dimension inference, and decomposition logic in
  isolation -- no ADIOS I/O needed.
* **Integration tests** (`test_backend.py`): create synthetic BP files via
  helper functions and verify round-trip reading through the xarray backend.
* **E3SM fixture tests** (`test_e3sm_fixtures.py`): validate against real E3SM
  output files (included under `tests/fixtures/`).
* **Regression tests** (`test_ne30pg2_regression.py`): compare ADIOS backend
  output against `netCDF4` for the same data.

### Generating synthetic test data

`tests/generate_fixtures.py` provides `PioFixtureBuilder` for creating
PIO-formatted BP files programmatically:

```python
from tests.generate_fixtures import PioFixtureBuilder

builder = PioFixtureBuilder("/tmp/test.bp")
builder.add_dimension("time", 3)
builder.add_dimension("ncol", 100)
builder.add_variable("PS", dims=("time", "ncol"))
builder.add_scorpio_metadata("PS", dims='{ "time", "ncol" }')
builder.write()
```

## Where this fits

This package is designed to be lightweight enough to embed directly in a
larger project:

* **[E3SM-Project/E3SM](https://github.com/E3SM-Project/E3SM)**: for
  reading SCORPIO BP output in post-processing workflows.
* **[ornladios/adios2](https://github.com/ornladios/ADIOS2)**: as a
  reference xarray integration for ADIOS2 files.
* **Data pipelines** (e.g.
  [ai2cm/ace](https://github.com/ai2cm/ace)): for ingesting E3SM output
  into ML training pipelines.

## License

See LICENSE file.
