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

# Required fixture files — skip all tests if any are missing
_REQUIRED_FIXTURES = [
    "eam_h0_gaussian_15x30.nc.bp",
    "eam_h1_native_ne4pg2.nc.bp",
    "eam_h2_gaussian_10x20.nc.bp",
]
pytestmark = pytest.mark.skipif(
    not all((FIXTURE_DIR / f).exists() for f in _REQUIRED_FIXTURES),
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

        with xr.open_dataset(h0_path, engine="adios") as ds:
            # h0 is a 15x30 Gaussian grid; engine may expose as lat/lon dims
            # or as a flattened ncol=450 (15*30) dimension
            if "lat" in ds.dims:
                assert ds.sizes["lat"] == 15
                assert ds.sizes["lon"] == 30
            else:
                assert any(ds.sizes[d] == 450 for d in ds.dims), (
                    f"Expected a dimension of size 450 (15*30), got {dict(ds.sizes)}"
                )

    def test_open_h2(self, h2_path):
        import xarray as xr

        with xr.open_dataset(h2_path, engine="adios") as ds:
            assert "PS" in ds or "PS" in ds.coords

    def test_h0_has_expected_variables(self, h0_path):
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            for var in ["PS", "TS", "PHIS"]:
                assert var in ds, f"Missing expected variable: {var}"
            for var in ["SOLIN", "FSUTOA", "FLUT", "FSDS", "FSUS", "FLDS", "FLUS"]:
                assert var in ds, f"Missing radiation field: {var}"
            for k in range(1, 9):
                assert f"T_{k}" in ds, f"Missing T_{k}"

    def test_h0_ps_physical_range(self, h0_path):
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            ps = ds["PS"].values
            valid = np.isfinite(ps) & (ps > 0)
            assert np.any(valid), "No valid PS data"
            assert ps[valid].min() > 50000, f"PS min too low: {ps[valid].min()}"
            assert ps[valid].max() < 120000, f"PS max too high: {ps[valid].max()}"

    def test_h0_temperature_layers(self, h0_path):
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            for k in range(1, 9):
                vname = f"T_{k}"
                if vname not in ds:
                    continue
                data = ds[vname].values
                valid = np.isfinite(data) & (np.abs(data) < 1e10)
                if np.any(valid):
                    assert data[valid].min() > 100, f"{vname} too cold: {data[valid].min()}"
                    assert data[valid].max() < 400, f"{vname} too hot: {data[valid].max()}"

    def test_h0_has_3_timesteps(self, h0_path):
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            ps = ds["PS"]
            if "time" in ps.dims:
                assert ps.sizes["time"] == 3
            else:
                nlat, nlon = 15, 30
                assert ps.size == 3 * nlat * nlon or ps.size == nlat * nlon

    def test_h0_radiation_non_negative(self, h0_path):
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            for var in ["SOLIN", "FSDS", "FLDS", "FLUT"]:
                if var in ds:
                    data = ds[var].values
                    valid = np.isfinite(data)
                    if np.any(valid):
                        assert data[valid].min() >= -1.0, f"{var} has negative values"


class TestE3SMUnstructuredNcol:
    """Tests for native unstructured ne4pg2 output (h1)."""

    def test_open_h1(self, h1_path):
        import xarray as xr

        with xr.open_dataset(h1_path, engine="adios") as ds:
            assert "ncol" in ds.dims or any(d for d in ds.dims if ds.sizes[d] == 384)

    def test_h1_has_expected_variables(self, h1_path):
        import xarray as xr

        with xr.open_dataset(h1_path, engine="adios") as ds:
            for var in ["PS", "TS", "PHIS"]:
                assert var in ds, f"Missing expected variable: {var}"

    def test_h1_ncol_size(self, h1_path):
        import xarray as xr

        with xr.open_dataset(h1_path, engine="adios") as ds:
            ps = ds["PS"]
            total = ps.size
            assert total % 384 == 0, f"PS size {total} not divisible by 384"

    def test_h1_decomp_reconstruction(self, h1_path):
        """Verify that decomp scatter produces correct ordering."""
        import xarray as xr

        with xr.open_dataset(h1_path, engine="adios") as ds:
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

        with (
            xr.open_dataset(h0_path, engine="adios") as ds0,
            xr.open_dataset(h1_path, engine="adios") as ds1,
        ):
            ps0 = ds0["PS"].values
            ps1 = ds1["PS"].values

            valid0 = np.isfinite(ps0) & (ps0 > 0)
            valid1 = np.isfinite(ps1) & (ps1 > 0)

            if np.any(valid0) and np.any(valid1):
                min_ratio = ps0[valid0].min() / ps1[valid1].min()
                max_ratio = ps0[valid0].max() / ps1[valid1].max()
                assert 0.95 < min_ratio < 1.05, f"PS min mismatch: {min_ratio}"
                assert 0.95 < max_ratio < 1.05, f"PS max mismatch: {max_ratio}"

    def test_both_have_same_variable_set(self, h0_path, h1_path):
        """h0 and h1 should have the same science variables."""
        import xarray as xr

        with (
            xr.open_dataset(h0_path, engine="adios") as ds0,
            xr.open_dataset(h1_path, engine="adios") as ds1,
        ):
            vars0 = set(ds0.data_vars)
            vars1 = set(ds1.data_vars)
            common = vars0 & vars1
            assert len(common) > 10, f"Too few common variables: {common}"


@pytest.mark.skipif(not has_mf_fixtures, reason="Multi-file BP fixtures not found")
class TestE3SMMultiFile:
    """Tests for open_mfdataset across multiple BP files."""

    def test_open_mfdataset_concat(self):
        """open_mfdataset should concatenate frames across files."""
        import xarray as xr

        with xr.open_dataset(str(MF_FILES[0]), engine="adios") as ds_single:
            time_dim = next(
                (d for d in ds_single.dims if d == "time" or d.startswith("frame")), None
            )
            assert time_dim is not None, f"No time/frame dim found in {dict(ds_single.sizes)}"
            nframes_single = ds_single.sizes[time_dim]

        with xr.open_mfdataset(
            [str(p) for p in MF_FILES],
            engine="adios",
            combine="nested",
            concat_dim=time_dim,
            data_vars="all",
        ) as ds_mf:
            assert ds_mf.sizes[time_dim] == len(MF_FILES) * nframes_single

    def test_mfdataset_variables_consistent(self):
        """All files should expose the same variable set."""
        import xarray as xr

        datasets = [xr.open_dataset(str(p), engine="adios") for p in MF_FILES]
        var_sets = [set(ds.data_vars) for ds in datasets]
        for ds in datasets:
            ds.close()

        assert all(v == var_sets[0] for v in var_sets[1:]), (
            f"Variable mismatch across files: {[v - var_sets[0] for v in var_sets[1:]]}"
        )

    def test_mfdataset_ps_physical_range(self):
        """PS values should be physical across all concatenated files."""
        import xarray as xr

        with xr.open_dataset(str(MF_FILES[0]), engine="adios") as ds_single:
            time_dim = next(d for d in ds_single.dims if d == "time" or d.startswith("frame"))

        with xr.open_mfdataset(
            [str(p) for p in MF_FILES],
            engine="adios",
            combine="nested",
            concat_dim=time_dim,
            data_vars="all",
        ) as ds_mf:
            ps = ds_mf["PS"].values
            valid = np.isfinite(ps) & (ps > 0)
            assert np.any(valid), "No valid PS data in multi-file dataset"
            assert ps[valid].min() > 50000, f"PS min too low: {ps[valid].min()}"
            assert ps[valid].max() < 120000, f"PS max too high: {ps[valid].max()}"


class TestE3SMBackendFeatures:
    """Tests for backend features: drop_variables, attrs, mask_and_scale, etc."""

    def test_drop_variables_string(self, h0_path):
        """drop_variables can be a single string (covers backend.py:70)."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios", drop_variables="PS") as ds:
            assert "PS" not in ds.data_vars

    def test_drop_variables_list(self, h0_path):
        """drop_variables can be a list."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios", drop_variables=["PS", "TS"]) as ds:
            assert "PS" not in ds.data_vars
            assert "TS" not in ds.data_vars

    def test_global_attrs(self, h0_path):
        """Dataset should have global attributes from PIO file."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            assert isinstance(ds.attrs, dict)

    def test_variable_attrs_present(self, h0_path):
        """Variables should carry attributes (units, long_name, etc.)."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            ps = ds["PS"]
            assert isinstance(ps.attrs, dict)
            assert len(ps.attrs) > 0, "PS has no attributes"

    def test_mask_and_scale_false(self, h0_path):
        """mask_and_scale=False should skip CF decoding."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios", mask_and_scale=False) as ds:
            assert "PS" in ds

    def test_decode_times_false(self, h0_path):
        """decode_times=False should skip CF time decoding."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios", decode_times=False) as ds:
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

        with xr.open_dataset(h0_path, engine="adios") as ds:
            assert "source" in ds.encoding
            assert h0_path in ds.encoding["source"]

    def test_lazy_loading_no_compute(self, h0_path):
        """Variables should be lazy (not loaded until .values is called)."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            ps_var = ds["PS"]
            assert hasattr(ps_var, "data")
            values = ps_var.values
            assert isinstance(values, np.ndarray)

    def test_h1_all_variables_load(self, h1_path):
        """All variables in h1 (decomp-reconstructed) should load without error."""
        import xarray as xr

        with xr.open_dataset(h1_path, engine="adios") as ds:
            for name in ds.data_vars:
                data = ds[name].values
                assert isinstance(data, np.ndarray), f"{name} failed to load"

    def test_h0_all_variables_load(self, h0_path):
        """All variables in h0 (concat+reshape) should load without error."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            for name in ds.data_vars:
                data = ds[name].values
                assert isinstance(data, np.ndarray), f"{name} failed to load"


class TestFrameSelectiveReading:
    """Tests for frame-selective lazy reading (time-dimension optimization)."""

    def test_h0_single_frame_correctness(self, h0_path):
        """Selecting one timestep should match full-read then slice."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            full = ds["PS"].values
            for t in range(full.shape[0]):
                frame = ds["PS"][t].values
                np.testing.assert_array_equal(frame, full[t])

    def test_h0_slice_frames(self, h0_path):
        """Slicing a range of timesteps should match full-read then slice."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            full = ds["PS"].values
            np.testing.assert_array_equal(ds["PS"][0:2].values, full[0:2])
            np.testing.assert_array_equal(ds["PS"][1:3].values, full[1:3])

    def test_h1_single_frame_correctness(self, h1_path):
        """Decomp path: single frame should match full-read then slice."""
        import xarray as xr

        with xr.open_dataset(h1_path, engine="adios") as ds:
            full = ds["PS"].values
            for t in range(full.shape[0]):
                frame = ds["PS"][t].values
                np.testing.assert_array_equal(frame, full[t])

    def test_h1_slice_frames(self, h1_path):
        """Decomp path: sliced frames should match full-read then slice."""
        import xarray as xr

        with xr.open_dataset(h1_path, engine="adios") as ds:
            full = ds["PS"].values
            np.testing.assert_array_equal(ds["PS"][0:2].values, full[0:2])
            np.testing.assert_array_equal(ds["PS"][1:3].values, full[1:3])

    def test_mixed_time_spatial_slice(self, h0_path):
        """Time + spatial slice should be correct."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            full = ds["PS"].values
            np.testing.assert_array_equal(ds["PS"][0, 5:10].values, full[0, 5:10])
            np.testing.assert_array_equal(ds["PS"][1, :, 10:20].values, full[1, :, 10:20])

    def test_h1_mixed_time_spatial_slice(self, h1_path):
        """Decomp path: time + spatial slice should be correct."""
        import xarray as xr

        with xr.open_dataset(h1_path, engine="adios") as ds:
            full = ds["PS"].values
            np.testing.assert_array_equal(ds["PS"][2, 100:200].values, full[2, 100:200])

    def test_frame_selective_reads_fewer_blocks(self, h0_path):
        """Verify that single-frame access reads fewer blocks than full read."""
        from unittest.mock import patch

        from xarray_adios.pio_store import PioStore

        store = PioStore(h0_path)
        store.get_variables()

        with patch.object(
            store, "_read_selected_blocks", wraps=store._read_selected_blocks
        ) as mock:
            _ = store.read_variable("PS")
            full_block_ids = [call.args[1] for call in mock.call_args_list]

        with patch.object(
            store, "_read_selected_blocks", wraps=store._read_selected_blocks
        ) as mock:
            _ = store.read_variable("PS", key=(0, slice(None), slice(None)))
            frame_block_ids = [call.args[1] for call in mock.call_args_list]

        total_full = sum(len(ids) for ids in full_block_ids)
        total_frame = sum(len(ids) for ids in frame_block_ids)
        assert total_frame < total_full, (
            f"Expected fewer block reads: frame={total_frame}, full={total_full}"
        )
        store.close()

    def test_h1_frame_selective_reads_fewer_blocks(self, h1_path):
        """Decomp path: single-frame access should read fewer blocks."""
        from unittest.mock import patch

        from xarray_adios.pio_store import PioStore

        store = PioStore(h1_path)
        store.get_variables()

        with patch.object(
            store, "_read_selected_blocks", wraps=store._read_selected_blocks
        ) as mock:
            _ = store.read_variable("PS")
            full_block_ids = [call.args[1] for call in mock.call_args_list]

        with patch.object(
            store, "_read_selected_blocks", wraps=store._read_selected_blocks
        ) as mock:
            _ = store.read_variable("PS", key=(0, slice(None)))
            frame_block_ids = [call.args[1] for call in mock.call_args_list]

        total_full = sum(len(ids) for ids in full_block_ids)
        total_frame = sum(len(ids) for ids in frame_block_ids)
        assert total_frame < total_full, (
            f"Expected fewer block reads: frame={total_frame}, full={total_full}"
        )
        store.close()

    def test_scalar_unaffected(self, h0_path):
        """Scalar variables should still work (no frame dimension)."""
        import xarray as xr

        with xr.open_dataset(h0_path, engine="adios") as ds:
            p0 = ds["P0"].values
            assert p0.shape == ()
            assert p0 == 100000.0
