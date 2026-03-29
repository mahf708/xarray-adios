"""Unit tests for decomposition logic."""

import numpy as np

from xarray_adios._pio_decomp import build_var_decomp_mapping, detect_nframes


class TestBuildVarDecompMapping:
    def test_attribute_based(self):
        attrs = {
            "/__pio__/var/T/_pio_decomp": {"Type": "string", "Value": '"512"', "Elements": "1"},
        }
        result = build_var_decomp_mapping(attrs)
        assert result == {"T": "512"}

    def test_track_based(self):
        attrs = {
            "/__pio__/track/PS": {"Type": "string", "Value": '"100"', "Elements": "1"},
        }
        result = build_var_decomp_mapping(attrs)
        assert result == {"PS": "100"}

    def test_attribute_takes_priority(self):
        attrs = {
            "/__pio__/var/T/_pio_decomp": {"Type": "string", "Value": '"512"', "Elements": "1"},
            "/__pio__/track/T": {"Type": "string", "Value": '"100"', "Elements": "1"},
        }
        result = build_var_decomp_mapping(attrs)
        assert result["T"] == "512"

    def test_empty(self):
        assert build_var_decomp_mapping({}) == {}


class TestDetectNframes:
    def test_separate_blocks(self):
        # 2 ranks, 6 data blocks = 3 frames
        data = [np.zeros(10) for _ in range(6)]
        decomp = [np.arange(1, 11), np.arange(11, 21)]
        assert detect_nframes(data, decomp, nranks=2, ndata_blocks=6) == 3

    def test_embedded_frames(self):
        # 2 ranks, data blocks are 3x the decomp size
        data = [np.zeros(30), np.zeros(30)]
        decomp = [np.arange(1, 11), np.arange(11, 21)]
        assert detect_nframes(data, decomp, nranks=2, ndata_blocks=2) == 3

    def test_single_frame(self):
        data = [np.zeros(10), np.zeros(10)]
        decomp = [np.arange(1, 11), np.arange(11, 21)]
        assert detect_nframes(data, decomp, nranks=2, ndata_blocks=2) == 1

    def test_mismatched_ratios(self):
        data = [np.zeros(30), np.zeros(20)]
        decomp = [np.arange(1, 11), np.arange(11, 21)]
        assert detect_nframes(data, decomp, nranks=2, ndata_blocks=2) == 1
