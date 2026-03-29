"""Unit tests for dimension inference logic."""

import numpy as np

from xarray_adios._pio_dims import (
    _find_dim_by_size,
    dims_from_def,
    dims_from_def_decomp,
    infer_dims_and_shape,
)


class TestDimsFromDef:
    def test_known_dims(self):
        dims = {"time": 3, "ncol": 100}
        result = dims_from_def('{ "time", "ncol" }', 300, dims)
        assert result == (("time", "ncol"), (3, 100))

    def test_infer_time(self):
        dims = {"time": 0, "ncol": 100}
        result = dims_from_def('{ "time", "ncol" }', 300, dims)
        assert result == (("time", "ncol"), (3, 100))

    def test_single_dim(self):
        dims = {"ncol": 50}
        result = dims_from_def('"ncol"', 50, dims)
        assert result == (("ncol",), (50,))

    def test_multiple_unknowns_returns_none(self):
        dims = {"time": 0, "lev": 0}
        assert dims_from_def('{ "time", "lev" }', 100, dims) is None

    def test_non_string_returns_none(self):
        assert dims_from_def(42, 100, {"ncol": 100}) is None

    def test_indivisible_returns_none(self):
        dims = {"time": 0, "ncol": 7}
        assert dims_from_def('{ "time", "ncol" }', 100, dims) is None


class TestDimsFromDefDecomp:
    def _make_decomp_reader(self, blocks):
        """Create a fake decomp reader that returns pre-built blocks."""
        cache = {"test_ioid": blocks}

        def reader(ioid):
            return cache[ioid]

        return reader

    def test_basic_decomp(self):
        decomp_blocks = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        reader = self._make_decomp_reader(decomp_blocks)
        dims = {"time": 0, "ncol": 6}

        result = dims_from_def_decomp('{ "time", "ncol" }', 18, dims, "test_ioid", reader)
        assert result is not None
        assert result[0] == ("time", "ncol")
        assert result[1] == (3, 6)

    def test_single_frame(self):
        decomp_blocks = [np.array([1, 2, 3, 4, 5])]
        reader = self._make_decomp_reader(decomp_blocks)
        dims = {"ncol": 5}

        result = dims_from_def_decomp('"ncol"', 5, dims, "test_ioid", reader)
        assert result is not None
        assert result[0] == ("ncol",)
        assert result[1] == (5,)


class TestInferDimsAndShape:
    def test_empty(self):
        assert infer_dims_and_shape("X", 0, [], {}, None, None) == (("_empty",), (0,))

    def test_scalar(self):
        assert infer_dims_and_shape("X", 1, [1], {}, None, None) == ((), ())

    def test_coordinate_match(self):
        dims = {"lat": 10, "lon": 20}
        result = infer_dims_and_shape("lat", 10, [10], dims, None, None)
        assert result == (("lat",), (10,))

    def test_time_x_spatial(self):
        dims = {"time": 3, "ncol": 100}
        result = infer_dims_and_shape("PS", 300, [300], dims, None, None)
        assert result == (("time", "ncol"), (3, 100))

    def test_fallback_flat(self):
        dims = {"time": 7}
        result = infer_dims_and_shape("X", 99, [99], dims, None, None)
        assert result == (("dim_X_99",), (99,))

    def test_decomp_aware(self):
        decomp_blocks = [np.array([1, 2, 3, 4, 5])]

        def reader(ioid):
            return decomp_blocks

        dims = {"time": 3, "ncol": 5}
        result = infer_dims_and_shape("T", 15, [15], dims, "100", reader)
        assert result == (("time", "ncol"), (3, 5))


class TestFindDimBySize:
    def test_found(self):
        assert _find_dim_by_size(10, {"lat": 10, "lon": 20}) == "lat"

    def test_not_found(self):
        assert _find_dim_by_size(99, {"lat": 10, "lon": 20}) is None

    def test_exclude(self):
        assert _find_dim_by_size(10, {"lat": 10, "time": 10}, exclude={"time"}) == "lat"
