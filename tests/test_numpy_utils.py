import numpy as np

from utils.numpy_utils import validate_layout, transpose_layout


def test_layout_none_skips():
    arr = np.zeros((1, 84, 8400), dtype=np.int32)
    validate_layout(arr, "NONE")  # should not raise
    out = transpose_layout(arr, "NONE")
    assert out is arr


def test_chw_hwc_roundtrip():
    arr = np.zeros((640, 640, 3), dtype=np.uint8)
    validate_layout(arr, "CHW")
    chw = transpose_layout(arr, "CHW")
    assert chw.shape == (3, 640, 640)
    validate_layout(chw, "HWC")
    hwc = transpose_layout(chw, "HWC")
    assert hwc.shape == (640, 640, 3)

