"""Tests for the xarray ADIOS backend."""

import numpy as np
import pytest
import xarray as xr

adios2 = pytest.importorskip("adios2")

from xarray_adios.backend import AdiosBackendEntrypoint
from xarray_adios.pio_store import PioStore, is_pio_file

from .conftest import write_pio_bp, write_simple_bp


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

        ds = xr.open_dataset(path, engine="adios")

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

        ds = xr.open_dataset(path, engine="adios")
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
        ds = xr.open_dataset(path, engine="adios", drop_variables=["Q"])
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
        ds = xr.open_dataset(path, engine="adios")
        # At this point, data should not be loaded yet
        # Accessing .values triggers loading
        result = ds["T"].values
        assert result.shape == (100,)
        np.testing.assert_array_equal(result, 1.0)
