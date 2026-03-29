"""Tests for the xarray ADIOS backend."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("adios2")

from xarray_adios.backend import AdiosBackendEntrypoint
from xarray_adios.pio_store import PioStore, is_pio_file

from .conftest import write_pio_bp, write_pio_bp_decomp, write_simple_bp


class TestGuessCanOpen:
    def test_bp_extension(self):
        be = AdiosBackendEntrypoint()
        assert be.guess_can_open("test.bp") is True
        assert be.guess_can_open("/path/to/data.bp4") is True
        assert be.guess_can_open("output.bp5") is True

    def test_non_bp_extension(self):
        be = AdiosBackendEntrypoint()
        assert be.guess_can_open("test.nc") is False
        assert be.guess_can_open("test.h5") is False
        assert be.guess_can_open("test.txt") is False


class TestIsPioFile:
    def test_pio_file(self, tmp_bp_dir):
        path = str(tmp_bp_dir / "pio_test.bp")
        write_pio_bp(
            path,
            variables={"T": np.ones(10, dtype=np.float32)},
            dimensions={"ncol": 10},
        )
        assert is_pio_file(path) is True

    def test_generic_file(self, tmp_bp_dir):
        path = str(tmp_bp_dir / "generic_test.bp")
        write_simple_bp(
            path,
            variables={"temp": np.ones((5, 10), dtype=np.float64)},
        )
        assert is_pio_file(path) is False


class TestPioStore:
    def test_read_dimensions(self, tmp_bp_dir):
        path = str(tmp_bp_dir / "dims_test.bp")
        dims = {"lat": 46, "lon": 72, "lev": 72, "time": 5}
        write_pio_bp(
            path,
            variables={"T": np.ones(46, dtype=np.float32)},
            dimensions=dims,
        )
        store = PioStore(path)
        read_dims = store.get_dimensions()
        for dname, dsize in dims.items():
            assert read_dims[dname] == dsize
        store.close()

    def test_read_1d_variable(self, tmp_bp_dir):
        path = str(tmp_bp_dir / "var1d_test.bp")
        lat_data = np.linspace(-90, 90, 46, dtype=np.float64)
        write_pio_bp(
            path,
            variables={"lat": lat_data},
            dimensions={"lat": 46},
        )
        store = PioStore(path)
        result = store.read_variable("lat")
        np.testing.assert_allclose(result, lat_data)
        store.close()

    def test_read_2d_variable(self, tmp_bp_dir):
        """Test reading a variable that maps to time × lat."""
        path = str(tmp_bp_dir / "var2d_test.bp")
        ntime, nlat = 3, 10
        data = np.arange(ntime * nlat, dtype=np.float32)
        write_pio_bp(
            path,
            variables={"T": data},
            dimensions={"time": ntime, "lat": nlat},
        )
        store = PioStore(path)
        result = store.read_variable("T")
        assert result.shape == (ntime, nlat)
        np.testing.assert_array_equal(result.ravel(), data)
        store.close()

    def test_variable_attrs(self, tmp_bp_dir):
        path = str(tmp_bp_dir / "attrs_test.bp")
        write_pio_bp(
            path,
            variables={"T": np.ones(10, dtype=np.float32)},
            dimensions={"ncol": 10},
            var_attrs={"T": {"units": "K", "long_name": "Temperature"}},
        )
        store = PioStore(path)
        var_infos = store.get_variables()
        assert "T" in var_infos
        assert var_infos["T"].attrs.get("units") == "K"
        assert var_infos["T"].attrs.get("long_name") == "Temperature"
        store.close()


class TestOpenDataset:
    def test_open_pio_dataset(self, tmp_bp_dir):
        """Test opening a PIO file via xr.open_dataset."""
        path = str(tmp_bp_dir / "ds_test.bp")
        nlat, nlon = 10, 20
        ntime = 2

        lat_data = np.linspace(-90, 90, nlat, dtype=np.float64)
        lon_data = np.linspace(0, 360, nlon, endpoint=False, dtype=np.float64)
        ps_data = np.random.rand(ntime * nlat * nlon).astype(np.float32)

        write_pio_bp(
            path,
            variables={
                "lat": lat_data,
                "lon": lon_data,
                "PS": ps_data,
            },
            dimensions={"time": ntime, "lat": nlat, "lon": nlon},
            var_attrs={"PS": {"units": "Pa", "long_name": "Surface Pressure"}},
            global_attrs={"title": "Test E3SM output"},
        )

        with xr.open_dataset(path, engine="adios") as ds:
            # Check coordinates
            assert "lat" in ds.coords
            assert "lon" in ds.coords

            # Check data variable
            assert "PS" in ds.data_vars
            assert ds["PS"].dims == ("time", "lat", "lon")
            assert ds["PS"].shape == (ntime, nlat, nlon)
            assert ds["PS"].attrs.get("units") == "Pa"

            # Check global attrs
            assert ds.attrs.get("title") == "Test E3SM output"

            # Check data values
            np.testing.assert_array_equal(
                ds["PS"].values.ravel(),
                ps_data,
            )

    def test_open_generic_dataset(self, tmp_bp_dir):
        """Test opening a generic (non-PIO) BP file."""
        path = str(tmp_bp_dir / "generic_ds_test.bp")

        data = np.random.rand(5, 10).astype(np.float64)
        write_simple_bp(
            path,
            variables={"temperature": data},
            attrs={"source": "test"},
        )

        with xr.open_dataset(path, engine="adios") as ds:
            assert "temperature" in ds.data_vars
            assert ds["temperature"].shape == (5, 10)
            np.testing.assert_array_equal(ds["temperature"].values, data)

    def test_drop_variables(self, tmp_bp_dir):
        path = str(tmp_bp_dir / "drop_test.bp")
        write_pio_bp(
            path,
            variables={
                "T": np.ones(10, dtype=np.float32),
                "Q": np.ones(10, dtype=np.float32),
            },
            dimensions={"ncol": 10},
        )
        with xr.open_dataset(path, engine="adios", drop_variables=["Q"]) as ds:
            assert "T" in ds.data_vars or "T" in ds.coords
            assert "Q" not in ds

    def test_lazy_loading(self, tmp_bp_dir):
        """Test that data is not loaded until accessed."""
        path = str(tmp_bp_dir / "lazy_test.bp")
        write_pio_bp(
            path,
            variables={"T": np.ones(100, dtype=np.float32)},
            dimensions={"ncol": 100},
        )
        with xr.open_dataset(path, engine="adios") as ds:
            # At this point, data should not be loaded yet
            # Accessing .values triggers loading
            result = ds["T"].values
            assert result.shape == (100,)
            np.testing.assert_array_equal(result, 1.0)


class TestDecompStore:
    """Tests for decomposition map reading and variable-to-decomp association."""

    def test_has_decomp_maps(self, tmp_bp_dir):
        """Detect presence of decomposition maps."""
        path = str(tmp_bp_dir / "decomp_detect.bp")
        ncol = 8
        decomp = np.array([3, 1, 4, 2, 8, 5, 7, 6], dtype=np.int64)
        data = np.arange(ncol, dtype=np.float32)

        write_pio_bp_decomp(
            path,
            variables={"T": data},
            dimensions={"ncol": ncol},
            decomp_maps={"512": decomp},
            var_decomps={"T": "512"},
        )

        store = PioStore(path)
        assert store.has_decomp_maps() is True
        assert "512" in store.get_decomp_ids()
        store.close()

    def test_no_decomp_maps(self, tmp_bp_dir):
        """File without decomp maps returns False."""
        path = str(tmp_bp_dir / "no_decomp.bp")
        write_pio_bp(
            path,
            variables={"T": np.ones(10, dtype=np.float32)},
            dimensions={"ncol": 10},
        )
        store = PioStore(path)
        assert store.has_decomp_maps() is False
        assert store.get_decomp_ids() == []
        store.close()

    def test_var_decomp_mapping_attribute(self, tmp_bp_dir):
        """Discover variable→decomp association via track attributes."""
        path = str(tmp_bp_dir / "decomp_attr.bp")
        ncol = 6
        decomp = np.array([2, 4, 6, 1, 3, 5], dtype=np.int64)
        data = np.arange(ncol, dtype=np.float64)

        write_pio_bp_decomp(
            path,
            variables={"T": data},
            dimensions={"ncol": ncol},
            decomp_maps={"100": decomp},
            var_decomps={"T": "100"},
        )

        store = PioStore(path)
        var_infos = store.get_variables()
        assert "T" in var_infos
        assert var_infos["T"].decomp_id == "100"
        store.close()

    def test_var_decomp_mapping_heuristic(self, tmp_bp_dir):
        """Discover variable→decomp association via block-count heuristic."""
        path = str(tmp_bp_dir / "decomp_heuristic.bp")
        ncol = 6
        decomp = np.array([2, 4, 6, 1, 3, 5], dtype=np.int64)
        data = np.arange(ncol, dtype=np.float64)

        # Don't pass var_decomps — reader should use heuristic
        write_pio_bp_decomp(
            path,
            variables={"T": data},
            dimensions={"ncol": ncol},
            decomp_maps={"100": decomp},
            var_decomps=None,
        )

        store = PioStore(path)
        var_infos = store.get_variables()
        assert "T" in var_infos
        assert var_infos["T"].decomp_id == "100"
        store.close()


class TestDecompReconstruction:
    """Tests for scatter-based variable reconstruction using decomp maps."""

    def test_1d_scatter(self, tmp_bp_dir):
        """Reconstruct a 1-D spatial variable using a decomp map."""
        path = str(tmp_bp_dir / "decomp_1d.bp")
        ncol = 6
        # decomp: element i in the block goes to global position decomp[i] (1-based)
        decomp = np.array([3, 1, 6, 2, 4, 5], dtype=np.int64)
        # Data in block order
        block_data = np.array([30.0, 10.0, 60.0, 20.0, 40.0, 50.0], dtype=np.float64)
        # Expected global result: position 1→10, 2→20, 3→30, 4→40, 5→50, 6→60
        expected = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)

        write_pio_bp_decomp(
            path,
            variables={"T": block_data},
            dimensions={"ncol": ncol},
            decomp_maps={"1": decomp},
            var_decomps={"T": "1"},
        )

        store = PioStore(path)
        result = store.read_variable("T")
        assert result.shape == (ncol,)
        np.testing.assert_array_equal(result, expected)
        store.close()

    def test_2d_time_spatial_scatter(self, tmp_bp_dir):
        """Reconstruct a time × spatial variable using a decomp map.

        With a single rank, each data block contains ntime × ncol_per_rank
        elements.  The decomp map is applied to each time-step's spatial slice.
        """
        path = str(tmp_bp_dir / "decomp_2d.bp")
        ncol = 4
        ntime = 3
        # Decomp: elements go to global positions [2, 4, 1, 3] (1-based)
        decomp = np.array([2, 4, 1, 3], dtype=np.int64)

        # Build block data: ntime frames, each with ncol elements in block order
        # Frame 0: [20, 40, 10, 30] → global [10, 20, 30, 40]
        # Frame 1: [120, 140, 110, 130] → global [110, 120, 130, 140]
        # Frame 2: [220, 240, 210, 230] → global [210, 220, 230, 240]
        block_data = np.array(
            [20, 40, 10, 30, 120, 140, 110, 130, 220, 240, 210, 230],
            dtype=np.float32,
        )
        expected = np.array(
            [
                [10, 20, 30, 40],
                [110, 120, 130, 140],
                [210, 220, 230, 240],
            ],
            dtype=np.float32,
        )

        write_pio_bp_decomp(
            path,
            variables={"PS": block_data},
            dimensions={"time": ntime, "ncol": ncol},
            decomp_maps={"1": decomp},
            var_decomps={"PS": "1"},
        )

        store = PioStore(path)
        result = store.read_variable("PS")
        assert result.shape == (ntime, ncol)
        np.testing.assert_array_equal(result, expected)
        store.close()

    def test_decomp_with_zero_indices(self, tmp_bp_dir):
        """Decomp indices of 0 mean 'unmapped' — those positions stay zero."""
        path = str(tmp_bp_dir / "decomp_zeros.bp")
        ncol = 4
        # Only positions 1, 3 are filled; 0 means unmapped
        decomp = np.array([1, 0, 3], dtype=np.int64)
        block_data = np.array([10.0, 99.0, 30.0], dtype=np.float64)

        write_pio_bp_decomp(
            path,
            variables={"T": block_data},
            dimensions={"ncol": ncol},
            decomp_maps={"1": decomp},
            var_decomps={"T": "1"},
        )

        store = PioStore(path)
        result = store.read_variable("T")
        # Spatial size = max(decomp) = 3; indices 1 and 3 filled, index 2 stays 0
        assert result.shape == (3,)
        assert result[0] == 10.0  # decomp index 1 → position 0
        assert result[1] == 0.0  # unmapped (index 0 skipped)
        assert result[2] == 30.0  # decomp index 3 → position 2
        store.close()

    def test_mixed_decomp_and_regular(self, tmp_bp_dir):
        """File with both decomp-mapped and regular (concat+reshape) variables."""
        path = str(tmp_bp_dir / "decomp_mixed.bp")
        ncol = 4
        decomp = np.array([2, 4, 1, 3], dtype=np.int64)

        # T uses decomp scatter
        t_block = np.array([20.0, 40.0, 10.0, 30.0], dtype=np.float32)
        t_expected = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        # lat is a 1-D coordinate — no decomp, different size than decomp
        lat_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        write_pio_bp_decomp(
            path,
            variables={"T": t_block, "lat": lat_data},
            dimensions={"ncol": ncol, "lat": 3},
            decomp_maps={"1": decomp},
            var_decomps={"T": "1"},
        )

        store = PioStore(path)
        var_infos = store.get_variables()

        # T should use decomp
        assert var_infos["T"].decomp_id == "1"
        t_result = store.read_variable("T")
        np.testing.assert_array_equal(t_result, t_expected)

        # lat should NOT use decomp
        assert var_infos["lat"].decomp_id is None
        lat_result = store.read_variable("lat")
        np.testing.assert_array_equal(lat_result, lat_data)

        store.close()

    def test_decomp_dimension_inference(self, tmp_bp_dir):
        """Decomp-based variables get correct dimension names from the file dims."""
        path = str(tmp_bp_dir / "decomp_dim_infer.bp")
        ncol = 6
        ntime = 2
        decomp = np.array([3, 1, 6, 2, 4, 5], dtype=np.int64)
        block_data = np.arange(ntime * ncol, dtype=np.float32)

        write_pio_bp_decomp(
            path,
            variables={"PS": block_data},
            dimensions={"time": ntime, "ncol": ncol},
            decomp_maps={"1": decomp},
            var_decomps={"PS": "1"},
        )

        store = PioStore(path)
        var_infos = store.get_variables()
        assert var_infos["PS"].dims == ("time", "ncol")
        assert var_infos["PS"].shape == (ntime, ncol)
        store.close()


class TestDecompOpenDataset:
    """Test opening decomp-mapped files via xr.open_dataset."""

    def test_open_decomp_dataset(self, tmp_bp_dir):
        """Full integration: open a decomp-mapped file as xr.Dataset."""
        path = str(tmp_bp_dir / "decomp_ds.bp")
        ncol = 5
        decomp = np.array([5, 3, 1, 4, 2], dtype=np.int64)

        block_data = np.array([50.0, 30.0, 10.0, 40.0, 20.0], dtype=np.float64)
        expected = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)

        write_pio_bp_decomp(
            path,
            variables={"SST": block_data},
            dimensions={"ncol": ncol},
            decomp_maps={"42": decomp},
            var_decomps={"SST": "42"},
            var_attrs={"SST": {"units": "K", "long_name": "Sea Surface Temperature"}},
            global_attrs={"title": "Decomp test"},
        )

        with xr.open_dataset(path, engine="adios") as ds:
            assert "SST" in ds.data_vars
            assert ds["SST"].dims == ("ncol",)
            assert ds["SST"].shape == (ncol,)
            np.testing.assert_array_equal(ds["SST"].values, expected)
            assert ds["SST"].attrs.get("units") == "K"
            assert ds.attrs.get("title") == "Decomp test"

    def test_open_decomp_dataset_time_spatial(self, tmp_bp_dir):
        """Integration test: time × spatial decomp-mapped variable."""
        path = str(tmp_bp_dir / "decomp_ts_ds.bp")
        ncol = 3
        ntime = 2
        decomp = np.array([3, 1, 2], dtype=np.int64)

        # Block data: 2 frames × 3 elements each, in decomp order
        # Frame 0: pos3→300, pos1→100, pos2→200 → [100, 200, 300]
        # Frame 1: pos3→600, pos1→400, pos2→500 → [400, 500, 600]
        block_data = np.array([300, 100, 200, 600, 400, 500], dtype=np.float32)
        expected = np.array(
            [
                [100, 200, 300],
                [400, 500, 600],
            ],
            dtype=np.float32,
        )

        write_pio_bp_decomp(
            path,
            variables={"T": block_data},
            dimensions={"time": ntime, "ncol": ncol},
            decomp_maps={"1": decomp},
            var_decomps={"T": "1"},
        )

        with xr.open_dataset(path, engine="adios") as ds:
            assert "T" in ds.data_vars
            assert ds["T"].dims == ("time", "ncol")
            assert ds["T"].shape == (ntime, ncol)
            np.testing.assert_array_equal(ds["T"].values, expected)
