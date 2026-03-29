"""Regression tests: compare netCDF vs ADIOS reading for ne30pg2 E3SM output.

These tests verify that opening multi-file ne30pg2 h0 output with
xr.open_mfdataset produces nearly identical results whether the files are
read via the netCDF4 backend or the ADIOS backend.

Fixture files (placed under tests/fixtures/):
  - ne30pg2_h0a.nc, ne30pg2_h0b.nc         (netCDF)
  - ne30pg2_h0a.nc.bp, ne30pg2_h0b.nc.bp   (ADIOS BP)

Each pair (a, b) represents two sequential h0 output files from an ne30pg2
E3SM run, suitable for concatenation along the time dimension.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("adios2")
netCDF4 = pytest.importorskip("netCDF4")

FIXTURE_DIR = Path(__file__).parent / "fixtures"

NC_FILES = sorted(FIXTURE_DIR.glob("ne30pg2_h0*.nc"))
# Exclude .nc.bp directories from the netCDF list
NC_FILES = [f for f in NC_FILES if not f.name.endswith(".bp")]

BP_FILES = sorted(FIXTURE_DIR.glob("ne30pg2_h0*.nc.bp"))

has_ne30pg2_fixtures = len(NC_FILES) >= 2 and len(BP_FILES) >= 2

pytestmark = pytest.mark.skipif(
    not has_ne30pg2_fixtures,
    reason="ne30pg2 fixture files not found (need >=2 NC and >=2 BP files)",
)


def _detect_time_dim(ds):
    """Return the name of the time/unlimited dimension."""
    for name in ("time", "Time"):
        if name in ds.dims:
            return name
    for name in ds.dims:
        if name.startswith("frame"):
            return name
    return None


def _open_mf_nc():
    """Open the netCDF fixture files with open_mfdataset."""
    import xarray as xr

    return xr.open_mfdataset(
        [str(p) for p in NC_FILES],
        combine="by_coords",
    )


def _open_mf_bp():
    """Open the ADIOS BP fixture files with open_mfdataset."""
    import xarray as xr

    ds_first = xr.open_dataset(str(BP_FILES[0]), engine="adios")
    time_dim = _detect_time_dim(ds_first)
    ds_first.close()

    return xr.open_mfdataset(
        [str(p) for p in BP_FILES],
        engine="adios",
        combine="nested",
        concat_dim=time_dim,
        data_vars="all",
    )


def _common_data_vars(ds_nc, ds_bp):
    """Return the sorted list of data variables present in both datasets."""
    return sorted(set(ds_nc.data_vars) & set(ds_bp.data_vars))


class TestNe30pg2AdiosVsNetcdf:
    """Verify ADIOS and netCDF backends produce nearly identical datasets."""

    def test_same_data_variables(self):
        """Both backends should expose the same set of data variables."""
        with _open_mf_nc() as ds_nc, _open_mf_bp() as ds_bp:
            vars_nc = set(ds_nc.data_vars)
            vars_bp = set(ds_bp.data_vars)
            assert vars_nc == vars_bp, (
                f"Variable mismatch:\n"
                f"  netCDF only: {vars_nc - vars_bp}\n"
                f"  ADIOS only:  {vars_bp - vars_nc}"
            )

    def test_same_dimensions(self):
        """Both backends should produce the same dimension sizes."""
        with _open_mf_nc() as ds_nc, _open_mf_bp() as ds_bp:
            dims_nc = dict(ds_nc.sizes)
            dims_bp = dict(ds_bp.sizes)
            assert dims_nc == dims_bp, (
                f"Dimension mismatch:\n  netCDF: {dims_nc}\n  ADIOS:  {dims_bp}"
            )

    def test_same_coordinates(self):
        """Both backends should expose the same coordinate variables."""
        with _open_mf_nc() as ds_nc, _open_mf_bp() as ds_bp:
            coords_nc = set(ds_nc.coords)
            coords_bp = set(ds_bp.coords)
            assert coords_nc == coords_bp, (
                f"Coordinate mismatch:\n"
                f"  netCDF only: {coords_nc - coords_bp}\n"
                f"  ADIOS only:  {coords_bp - coords_nc}"
            )

    def test_variable_shapes_match(self):
        """Every data variable should have identical shape."""
        with _open_mf_nc() as ds_nc, _open_mf_bp() as ds_bp:
            for vname in _common_data_vars(ds_nc, ds_bp):
                assert ds_nc[vname].shape == ds_bp[vname].shape, (
                    f"{vname} shape: NC={ds_nc[vname].shape} vs BP={ds_bp[vname].shape}"
                )

    def test_variable_dtypes_compatible(self):
        """Every data variable should have compatible dtypes."""
        with _open_mf_nc() as ds_nc, _open_mf_bp() as ds_bp:
            for vname in _common_data_vars(ds_nc, ds_bp):
                dt_nc = ds_nc[vname].dtype
                dt_bp = ds_bp[vname].dtype
                assert np.issubdtype(dt_nc, np.number) == np.issubdtype(dt_bp, np.number), (
                    f"{vname} dtype kind: NC={dt_nc} vs BP={dt_bp}"
                )

    def test_all_values_close(self):
        """Every data variable should have nearly identical values."""
        with _open_mf_nc() as ds_nc, _open_mf_bp() as ds_bp:
            for vname in _common_data_vars(ds_nc, ds_bp):
                val_nc = ds_nc[vname].values
                val_bp = ds_bp[vname].values
                if np.issubdtype(val_nc.dtype, np.floating):
                    np.testing.assert_allclose(
                        val_bp,
                        val_nc,
                        rtol=1e-6,
                        atol=0,
                        err_msg=f"Value mismatch for {vname}",
                    )
                elif np.issubdtype(val_nc.dtype, np.integer):
                    np.testing.assert_array_equal(
                        val_bp,
                        val_nc,
                        err_msg=f"Value mismatch for {vname}",
                    )

    def test_global_attrs_match(self):
        """Global attributes should match between backends."""
        # Skip attrs that differ between runs (timestamps, etc.)
        skip_attrs = {"history"}
        with _open_mf_nc() as ds_nc, _open_mf_bp() as ds_bp:
            attrs_nc = set(ds_nc.attrs.keys())
            attrs_bp = set(ds_bp.attrs.keys())
            common = (attrs_nc & attrs_bp) - skip_attrs
            assert len(common) > 0, "No common global attributes"
            for aname in sorted(common):
                val_nc = ds_nc.attrs[aname]
                val_bp = ds_bp.attrs[aname]
                if isinstance(val_nc, np.ndarray):
                    np.testing.assert_array_equal(
                        val_nc,
                        val_bp,
                        err_msg=f"Global attr mismatch: {aname}",
                    )
                else:
                    assert val_nc == val_bp, (
                        f"Global attr '{aname}': NC={val_nc!r} vs BP={val_bp!r}"
                    )

    def test_variable_attrs_present(self):
        """Data variables should have matching units and long_name."""
        with _open_mf_nc() as ds_nc, _open_mf_bp() as ds_bp:
            for vname in _common_data_vars(ds_nc, ds_bp):
                attrs_nc = ds_nc[vname].attrs
                attrs_bp = ds_bp[vname].attrs
                for key in ("units", "long_name"):
                    if key in attrs_nc:
                        assert key in attrs_bp, f"{vname} missing '{key}' in ADIOS"
                        assert attrs_nc[key] == attrs_bp[key], (
                            f"{vname}.{key}: NC={attrs_nc[key]!r} vs BP={attrs_bp[key]!r}"
                        )


class TestNe30pg2SingleFile:
    """Sanity checks on individual ne30pg2 files."""

    def test_nc_files_open(self):
        """Each netCDF file should open successfully."""
        import xarray as xr

        for p in NC_FILES:
            with xr.open_dataset(str(p)) as ds:
                assert len(ds.data_vars) > 0, f"No variables in {p.name}"

    def test_bp_files_open(self):
        """Each BP file should open successfully via the ADIOS backend."""
        import xarray as xr

        for p in BP_FILES:
            with xr.open_dataset(str(p), engine="adios") as ds:
                assert len(ds.data_vars) > 0, f"No variables in {p.name}"

    def test_per_file_variable_count_matches(self):
        """Each NC/BP pair should have the same number of data variables."""
        import xarray as xr

        for nc_path, bp_path in zip(NC_FILES, BP_FILES, strict=True):
            with (
                xr.open_dataset(str(nc_path)) as ds_nc,
                xr.open_dataset(str(bp_path), engine="adios") as ds_bp,
            ):
                assert len(ds_nc.data_vars) == len(ds_bp.data_vars), (
                    f"Variable count mismatch in {nc_path.name}: "
                    f"NC={len(ds_nc.data_vars)} vs BP={len(ds_bp.data_vars)}"
                )
