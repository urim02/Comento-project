import numpy as np
import pytest
from depth_map import generate_depth_map

def test_generate_depth_map_valid():
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    result = generate_depth_map(dummy)

    assert result.shape == dummy.shape
    assert result.dtype == dummy.dtype
    assert isinstance(result, np.ndarray)

def test_generate_depth_map_none_input():
    with pytest.raises(ValueError):
        generate_depth_map(None)
