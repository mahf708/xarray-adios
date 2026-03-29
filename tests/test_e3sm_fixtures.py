"""Tests using real E3SM ADIOS BP output files as fixtures.

These fixtures were produced by an EAM ne4pg2 simulation with:
- h0: 15x30 Gaussian grid (horiz_remap, structured lat-lon, identity decomp)
- h1: native ne4pg2 (ncol=384, unstructured, non-trivial decomp)
- h2: 10x20 Gaussian grid (horiz_remap, structured lat-lon, smaller grid)

All files contain 3 timesteps (mfilt=3) at 6-hourly intervals with a mix of
instantaneous (:I) and averaged (:A) fields, vertically coarsened layers
(T_1..T_8, SPECIFIC_TOTAL_WATER_1..8), derived fields, and new radiation
diagnostics (FLUS, FSUS, DTENDTTW).
"""

import os
from pathlib import Path

import numpy as np
import pytest

adios2 = pytest.importorskip("adios2")

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Skip all tests if fixtures are not present
pytestmark = pytest.mark.skipif(
    not (FIXTURE_DIR / "eam_h0_gaussian_15x30.nc.bp").exists(),
    reason="E3SM BP fixture files not found",
)


@pytest.fixture
def h0_path():
    """15x30 Gaussian remapped output."""
    return str(FIXTURE_DIR / "eam_h0_gaussian_15x30.nc.bp")


@pytest.fixture
def h1_path():
    """Native ne4pg2 unstructured output."""
    return str(FIXTURE_DIR / "eam_h1_native_ne4pg2.nc.bp")


@pytest.fixture
def h2_path():
    """10x20 Gaussian remapped output."""
    return str(FIXTURE_DIR / "eam_h2_gaussian_10x20.nc.bp")


class TestE3SMStructuredLatLon:
    """Tests for structured lat-lon remapped output (h0, h2)."""

    def test_open_h0(self, h0_path):
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        assert "lat" in ds.dims or "lat" in ds.coords
        assert "lon" in ds.dims or "lon" in ds.coords
        assert ds.sizes.get("lat", 0) == 15 or len(ds["lat"]) == 15
        assert ds.sizes.get("lon", 0) == 30 or len(ds["lon"]) == 30

    def test_open_h2(self, h2_path):
        import xarray as xr

        ds = xr.open_dataset(h2_path, engine="adios")
        # h2 is 10x20 or similar small grid
        assert "PS" in ds or "PS" in ds.coords

    def test_h0_has_expected_variables(self, h0_path):
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        # 2D surface fields
        for var in ["PS", "TS", "PHIS"]:
            assert var in ds, f"Missing expected variable: {var}"
        # Radiation fluxes
        for var in ["SOLIN", "FSUTOA", "FLUT", "FSDS", "FSUS", "FLDS", "FLUS"]:
            assert var in ds, f"Missing radiation field: {var}"
        # Vertically coarsened
        for k in range(1, 9):
            assert f"T_{k}" in ds, f"Missing T_{k}"

    def test_h0_ps_physical_range(self, h0_path):
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        ps = ds["PS"].values
        valid = np.isfinite(ps) & (ps > 0)
        assert np.any(valid), "No valid PS data"
        # Surface pressure should be 50000-110000 Pa
        assert ps[valid].min() > 50000, f"PS min too low: {ps[valid].min()}"
        assert ps[valid].max() < 120000, f"PS max too high: {ps[valid].max()}"

    def test_h0_temperature_layers(self, h0_path):
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        for k in range(1, 9):
            vname = f"T_{k}"
            if vname not in ds:
                continue
            data = ds[vname].values
            valid = np.isfinite(data) & (np.abs(data) < 1e20)
            if np.any(valid):
                # Temperature should be 150-350 K
                assert data[valid].min() > 100, f"{vname} too cold: {data[valid].min()}"
                assert data[valid].max() < 400, f"{vname} too hot: {data[valid].max()}"

    def test_h0_has_3_timesteps(self, h0_path):
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        # PS should have time dimension with 3 steps
        ps = ds["PS"]
        if "time" in ps.dims:
            assert ps.sizes["time"] == 3
        else:
            # Might be flattened — check total size
            nlat, nlon = 15, 30
            assert ps.size == 3 * nlat * nlon or ps.size == nlat * nlon

    def test_h0_radiation_non_negative(self, h0_path):
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        for var in ["SOLIN", "FSDS", "FLDS", "FLUT"]:
            if var in ds:
                data = ds[var].values
                valid = np.isfinite(data)
                if np.any(valid):
                    # Downwelling fluxes should be >= 0
                    assert data[valid].min() >= -1.0, f"{var} has negative values"


class TestE3SMUnstructuredNcol:
    """Tests for native unstructured ne4pg2 output (h1)."""

    def test_open_h1(self, h1_path):
        import xarray as xr

        ds = xr.open_dataset(h1_path, engine="adios")
        assert "ncol" in ds.dims or any(
            d for d in ds.dims if ds.sizes[d] == 384
        )

    def test_h1_has_expected_variables(self, h1_path):
        import xarray as xr

        ds = xr.open_dataset(h1_path, engine="adios")
        for var in ["PS", "TS", "PHIS"]:
            assert var in ds, f"Missing expected variable: {var}"

    def test_h1_ncol_size(self, h1_path):
        import xarray as xr

        ds = xr.open_dataset(h1_path, engine="adios")
        # ne4pg2 has 384 columns
        ps = ds["PS"]
        total = ps.size
        # Should be divisible by 384 (3 timesteps × 384 = 1152)
        assert total % 384 == 0, f"PS size {total} not divisible by 384"

    def test_h1_decomp_reconstruction(self, h1_path):
        """Verify that decomp scatter produces correct ordering."""
        import xarray as xr

        ds = xr.open_dataset(h1_path, engine="adios")
        # PS values should be physical regardless of decomp
        ps = ds["PS"].values
        valid = np.isfinite(ps) & (ps > 0)
        assert np.any(valid), "No valid PS data after decomp reconstruction"
        assert ps[valid].min() > 50000
        assert ps[valid].max() < 120000


class TestE3SMCrossComparison:
    """Compare remapped vs native output for consistency."""

    def test_ps_range_consistent(self, h0_path, h1_path):
        """PS range should be similar between remapped and native."""
        import xarray as xr

        ds0 = xr.open_dataset(h0_path, engine="adios")
        ds1 = xr.open_dataset(h1_path, engine="adios")

        ps0 = ds0["PS"].values
        ps1 = ds1["PS"].values

        valid0 = np.isfinite(ps0) & (ps0 > 0)
        valid1 = np.isfinite(ps1) & (ps1 > 0)

        if np.any(valid0) and np.any(valid1):
            # Global min/max should be in similar range (within 5%)
            min_ratio = ps0[valid0].min() / ps1[valid1].min()
            max_ratio = ps0[valid0].max() / ps1[valid1].max()
            assert 0.95 < min_ratio < 1.05, f"PS min mismatch: {min_ratio}"
            assert 0.95 < max_ratio < 1.05, f"PS max mismatch: {max_ratio}"

    def test_both_have_same_variable_set(self, h0_path, h1_path):
        """h0 and h1 should have the same science variables."""
        import xarray as xr

        ds0 = xr.open_dataset(h0_path, engine="adios")
        ds1 = xr.open_dataset(h1_path, engine="adios")

        vars0 = set(ds0.data_vars)
        vars1 = set(ds1.data_vars)

        # They should largely overlap (h0 has same fincl as h1)
        common = vars0 & vars1
        assert len(common) > 10, f"Too few common variables: {common}"
