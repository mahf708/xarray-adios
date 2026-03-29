"""Unit tests for _common parsing utilities."""

import numpy as np
import pytest

from xarray_adios._common import (
    _ADIOS_TYPE_MAP,
    _NC_TYPE_MAP,
    parse_attr_value,
    parse_block_count,
    parse_string_array,
)


class TestParseStringArray:
    def test_braces_format(self):
        assert parse_string_array('{ "time", "ncol" }') == ("time", "ncol")

    def test_single_quoted(self):
        assert parse_string_array('"ncol"') == ("ncol",)

    def test_bare_string(self):
        assert parse_string_array("ncol") == ("ncol",)

    def test_three_dims(self):
        assert parse_string_array('{ "time", "lev", "ncol" }') == ("time", "lev", "ncol")

    def test_empty_braces(self):
        assert parse_string_array("{ }") == ()

    def test_non_string_returns_empty(self):
        assert parse_string_array(42) == ()
        assert parse_string_array(None) == ()


class TestParseAttrValue:
    def test_string_type(self):
        assert parse_attr_value({"Type": "string", "Value": '"hello"', "Elements": "1"}) == "hello"

    def test_string_no_quotes(self):
        assert parse_attr_value({"Type": "string", "Value": "hello", "Elements": "1"}) == "hello"

    def test_float_scalar(self):
        result = parse_attr_value({"Type": "float", "Value": "3.14", "Elements": "1"})
        assert result == pytest.approx(3.14)

    def test_double_scalar(self):
        result = parse_attr_value({"Type": "double", "Value": "2.718", "Elements": "1"})
        assert result == pytest.approx(2.718)

    def test_float_array(self):
        result = parse_attr_value({"Type": "float", "Value": "{ 1.0, 2.0, 3.0 }", "Elements": "3"})
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_int_scalar(self):
        assert parse_attr_value({"Type": "int32_t", "Value": "42", "Elements": "1"}) == 42

    def test_int_array(self):
        result = parse_attr_value({"Type": "int32_t", "Value": "{ 10, 20 }", "Elements": "2"})
        np.testing.assert_array_equal(result, [10, 20])

    def test_unknown_type(self):
        assert parse_attr_value({"Type": "unknown", "Value": "raw", "Elements": "1"}) == "raw"


class TestParseBlockCount:
    def test_scalar_string(self):
        assert parse_block_count("100") == 100

    def test_scalar_int(self):
        assert parse_block_count(100) == 100

    def test_comma_separated(self):
        assert parse_block_count("10,20") == 200

    def test_3d_shape(self):
        assert parse_block_count("2,3,4") == 24


class TestTypeMaps:
    def test_adios_type_map_covers_basics(self):
        assert _ADIOS_TYPE_MAP["float"] == np.float32
        assert _ADIOS_TYPE_MAP["double"] == np.float64
        assert _ADIOS_TYPE_MAP["int32_t"] == np.int32
        assert _ADIOS_TYPE_MAP["string"] is str

    def test_nc_type_map_covers_basics(self):
        assert _NC_TYPE_MAP[5] == np.dtype(np.float32)
        assert _NC_TYPE_MAP[6] == np.dtype(np.float64)
        assert _NC_TYPE_MAP[4] == np.dtype(np.int32)
