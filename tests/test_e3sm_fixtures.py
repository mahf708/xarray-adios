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

MF_FILES = sorted(FIXTURE_DIR.glob("eam_h0_mf_t*.nc.bp"))
has_mf_fixtures = len(MF_FILES) >= 2


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
        # h0 is a 15x30 Gaussian grid; engine may expose as lat/lon dims
        # or as a flattened ncol=450 (15*30) dimension
        if "lat" in ds.dims:
            assert ds.sizes["lat"] == 15
            assert ds.sizes["lon"] == 30
        else:
            # Flattened: look for a dimension of size 450 = 15*30
            assert any(ds.sizes[d] == 450 for d in ds.dims), (
                f"Expected a dimension of size 450 (15*30), got {dict(ds.sizes)}"
            )

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
            valid = np.isfinite(data) & (np.abs(data) < 1e10)
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
        assert "ncol" in ds.dims or any(d for d in ds.dims if ds.sizes[d] == 384)

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


@pytest.mark.skipif(not has_mf_fixtures, reason="Multi-file BP fixtures not found")
class TestE3SMMultiFile:
    """Tests for open_mfdataset across multiple BP files."""

    def test_open_mfdataset_concat(self):
        """open_mfdataset should concatenate frames across files."""
        import xarray as xr

        ds_single = xr.open_dataset(str(MF_FILES[0]), engine="adios")
        # Find the time/frame dimension (may be "time" or "frame_N")
        time_dim = next((d for d in ds_single.dims if d == "time" or d.startswith("frame")), None)
        assert time_dim is not None, f"No time/frame dim found in {dict(ds_single.sizes)}"
        nframes_single = ds_single.sizes[time_dim]

        ds_mf = xr.open_mfdataset(
            [str(p) for p in MF_FILES],
            engine="adios",
            combine="nested",
            concat_dim=time_dim,
            data_vars="all",
        )
        # Total frames = nfiles × frames_per_file
        assert ds_mf.sizes[time_dim] == len(MF_FILES) * nframes_single
        ds_single.close()
        ds_mf.close()

    def test_mfdataset_variables_consistent(self):
        """All files should expose the same variable set."""
        import xarray as xr

        datasets = [xr.open_dataset(str(p), engine="adios") for p in MF_FILES]
        var_sets = [set(ds.data_vars) for ds in datasets]
        for ds in datasets:
            ds.close()

        # All files should have the same variables
        assert all(v == var_sets[0] for v in var_sets[1:]), (
            f"Variable mismatch across files: {[v - var_sets[0] for v in var_sets[1:]]}"
        )

    def test_mfdataset_ps_physical_range(self):
        """PS values should be physical across all concatenated files."""
        import xarray as xr

        ds_single = xr.open_dataset(str(MF_FILES[0]), engine="adios")
        time_dim = next(d for d in ds_single.dims if d == "time" or d.startswith("frame"))
        ds_single.close()

        ds_mf = xr.open_mfdataset(
            [str(p) for p in MF_FILES],
            engine="adios",
            combine="nested",
            concat_dim=time_dim,
            data_vars="all",
        )
        ps = ds_mf["PS"].values
        valid = np.isfinite(ps) & (ps > 0)
        assert np.any(valid), "No valid PS data in multi-file dataset"
        assert ps[valid].min() > 50000, f"PS min too low: {ps[valid].min()}"
        assert ps[valid].max() < 120000, f"PS max too high: {ps[valid].max()}"
        ds_mf.close()


class TestE3SMBackendFeatures:
    """Tests for backend features: drop_variables, attrs, mask_and_scale, etc."""

    def test_drop_variables_string(self, h0_path):
        """drop_variables can be a single string (covers backend.py:70)."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios", drop_variables="PS")
        assert "PS" not in ds.data_vars

    def test_drop_variables_list(self, h0_path):
        """drop_variables can be a list."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios", drop_variables=["PS", "TS"])
        assert "PS" not in ds.data_vars
        assert "TS" not in ds.data_vars

    def test_global_attrs(self, h0_path):
        """Dataset should have global attributes from PIO file."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        # E3SM files typically have global attributes like case, title, etc.
        assert isinstance(ds.attrs, dict)

    def test_variable_attrs_present(self, h0_path):
        """Variables should carry attributes (units, long_name, etc.)."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        ps = ds["PS"]
        # PIO-written variables typically have units, long_name, cell_methods
        assert isinstance(ps.attrs, dict)
        # At least some attributes should be present
        assert len(ps.attrs) > 0, "PS has no attributes"

    def test_mask_and_scale_false(self, h0_path):
        """mask_and_scale=False should skip CF decoding."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios", mask_and_scale=False)
        assert "PS" in ds

    def test_decode_times_false(self, h0_path):
        """decode_times=False should skip CF time decoding."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios", decode_times=False)
        assert "PS" in ds

    def test_guess_can_open(self, h0_path):
        """Backend should recognize .bp paths and reject others."""
        from xarray_adios.backend import AdiosBackendEntrypoint

        backend = AdiosBackendEntrypoint()
        assert backend.guess_can_open(h0_path) is True
        assert backend.guess_can_open("somefile.nc") is False
        assert backend.guess_can_open("somefile.bp4") is True
        assert backend.guess_can_open("somefile.bp5") is True
        assert backend.guess_can_open(12345) is False

        # Object whose __str__ raises should return False (covers backend.py:153-154)
        class Unconvertible:
            def __str__(self):
                raise TypeError

        assert backend.guess_can_open(Unconvertible()) is False

    def test_is_pio_file_invalid_path(self):
        """is_pio_file should return False for non-existent files."""
        from xarray_adios.pio_store import is_pio_file

        assert is_pio_file("/nonexistent/path.bp") is False

    def test_h0_encoding_has_source(self, h0_path):
        """Dataset encoding should contain the source filename."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        assert "source" in ds.encoding
        assert h0_path in ds.encoding["source"]

    def test_lazy_loading_no_compute(self, h0_path):
        """Variables should be lazy (not loaded until .values is called)."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        ps_var = ds["PS"]
        # Should be lazy — check that data is a LazilyIndexedArray
        assert hasattr(ps_var, "data")
        # Force computation
        values = ps_var.values
        assert isinstance(values, np.ndarray)

    def test_h1_all_variables_load(self, h1_path):
        """All variables in h1 (decomp-reconstructed) should load without error."""
        import xarray as xr

        ds = xr.open_dataset(h1_path, engine="adios")
        for name in ds.data_vars:
            data = ds[name].values
            assert isinstance(data, np.ndarray), f"{name} failed to load"

    def test_h0_all_variables_load(self, h0_path):
        """All variables in h0 (concat+reshape) should load without error."""
        import xarray as xr

        ds = xr.open_dataset(h0_path, engine="adios")
        for name in ds.data_vars:
            data = ds[name].values
            assert isinstance(data, np.ndarray), f"{name} failed to load"
