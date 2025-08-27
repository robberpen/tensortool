"""
Basic unit tests for ImageTensorConverter.
Run with: python -m pytest tests/test_converter.py -v
"""

import pytest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensor_image_utils import ImageTensorConverter
from tensor_image_utils.exceptions import UnsupportedFormatError, InvalidDimensionError


class TestImageTensorConverter:
    """Test cases for ImageTensorConverter class."""
    
    @pytest.fixture
    def converter(self):
        """Create converter instance for testing."""
        return ImageTensorConverter()
    
    @pytest.fixture
    def sample_hwc_tensor(self):
        """Create sample HWC tensor for testing."""
        return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_nchw_tensor(self):
        """Create sample NCHW tensor for testing."""
        return np.random.randint(0, 255, (1, 3, 32, 32), dtype=np.uint8)
    
    def test_initialization(self, converter):
        """Test converter initialization."""
        assert converter is not None
        assert len(converter.get_supported_formats()) > 0
        
        # Check expected formats are supported
        supported = converter.get_supported_formats()
        expected_formats = {'jpg', 'jpeg', 'bmp', 'webp', 'npy', 'rgb888', 'bgr888'}
        assert expected_formats.issubset(supported)
    
    def test_get_supported_formats(self, converter):
        """Test supported formats retrieval."""
        formats = converter.get_supported_formats()
        assert isinstance(formats, set)
        assert len(formats) >= 7  # At least 7 formats
        
        # Modify returned set shouldn't affect internal state
        formats.add('fake_format')
        assert 'fake_format' not in converter.get_supported_formats()
    
    def test_to_nchw_conversion(self, converter, sample_hwc_tensor):
        """Test HWC to NCHW conversion."""
        result = converter.to_nchw(sample_hwc_tensor)
        
        assert result.ndim == 4
        assert result.shape[0] == 1  # Batch dimension
        assert result.shape[1] == 3  # Channels
        assert result.shape[2] == sample_hwc_tensor.shape[0]  # Height
        assert result.shape[3] == sample_hwc_tensor.shape[1]  # Width
        
        # Check data integrity
        assert np.array_equal(result[0, 0, :, :], sample_hwc_tensor[:, :, 0])
        assert np.array_equal(result[0, 1, :, :], sample_hwc_tensor[:, :, 1])
        assert np.array_equal(result[0, 2, :, :], sample_hwc_tensor[:, :, 2])
    
    def test_to_nhwc_conversion(self, converter, sample_nchw_tensor):
        """Test NCHW to NHWC conversion."""
        result = converter.to_nhwc(sample_nchw_tensor)
        
        assert result.ndim == 4
        assert result.shape[0] == 1  # Batch dimension
        assert result.shape[1] == sample_nchw_tensor.shape[2]  # Height
        assert result.shape[2] == sample_nchw_tensor.shape[3]  # Width
        assert result.shape[3] == 3  # Channels
        
        # Check data integrity
        assert np.array_equal(result[0, :, :, 0], sample_nchw_tensor[0, 0, :, :])
        assert np.array_equal(result[0, :, :, 1], sample_nchw_tensor[0, 1, :, :])
        assert np.array_equal(result[0, :, :, 2], sample_nchw_tensor[0, 2, :, :])
    
    def test_to_hwc_from_3d(self, converter, sample_hwc_tensor):
        """Test HWC conversion from 3D tensor."""
        # Test with already HWC tensor
        result = converter.to_hwc(sample_hwc_tensor)
        assert np.array_equal(result, sample_hwc_tensor)
        
        # Test with CHW tensor
        chw_tensor = np.transpose(sample_hwc_tensor, (2, 0, 1))
        result = converter.to_hwc(chw_tensor)
        # Should convert back to original HWC
        assert result.shape == sample_hwc_tensor.shape
    
    def test_to_hwc_from_4d(self, converter, sample_nchw_tensor):
        """Test HWC conversion from 4D tensor."""
        result = converter.to_hwc(sample_nchw_tensor)
        
        assert result.ndim == 3
        assert result.shape[0] == sample_nchw_tensor.shape[2]  # Height
        assert result.shape[1] == sample_nchw_tensor.shape[3]  # Width
        assert result.shape[2] == sample_nchw_tensor.shape[1]  # Channels
    
    def test_resize_tensor(self, converter, sample_hwc_tensor):
        """Test tensor resizing."""
        target_size = (128, 96)  # (width, height)
        result = converter.resize(sample_hwc_tensor, target_size)
        
        assert result.ndim == 3
        assert result.shape[0] == 96   # Height
        assert result.shape[1] == 128  # Width
        assert result.shape[2] == 3    # Channels
        assert result.dtype == np.uint8
    
    def test_get_tensor_info(self, converter, sample_hwc_tensor):
        """Test tensor information retrieval."""
        info = converter.get_tensor_info(sample_hwc_tensor)
        
        assert isinstance(info, dict)
        assert info['shape'] == sample_hwc_tensor.shape
        assert info['dtype'] == str(sample_hwc_tensor.dtype)
        assert info['ndim'] == sample_hwc_tensor.ndim
        assert info['size'] == sample_hwc_tensor.size
        assert 'min_value' in info
        assert 'max_value' in info
        assert 'mean_value' in info
    
    def test_invalid_input_types(self, converter):
        """Test error handling for invalid input types."""
        with pytest.raises(SystemExit):
            converter.to_nchw("not_an_array")
        
        with pytest.raises(SystemExit):
            converter.resize([1, 2, 3], (64, 64))
        
        with pytest.raises(SystemExit):
            converter.get_tensor_info("not_an_array")
    
    def test_invalid_tensor_dimensions(self, converter):
        """Test error handling for invalid tensor dimensions."""
        # 2D tensor for functions expecting 3D
        tensor_2d = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        
        with pytest.raises(SystemExit):
            converter.to_nchw(tensor_2d)
        
        with pytest.raises(SystemExit):
            converter.resize(tensor_2d, (32, 32))
    
    def test_invalid_target_sizes(self, converter, sample_hwc_tensor):
        """Test error handling for invalid resize target sizes."""
        with pytest.raises(SystemExit):
            converter.resize(sample_hwc_tensor, (0, 32))
        
        with pytest.raises(SystemExit):
            converter.resize(sample_hwc_tensor, (-10, 32))
        
        with pytest.raises(SystemExit):
            converter.resize(sample_hwc_tensor, (32,))  # Wrong tuple length
    
    def test_unsupported_file_format(self, converter):
        """Test error handling for unsupported file formats."""
        with pytest.raises(SystemExit):
            converter.load("test.unknown")
        
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(SystemExit):
                converter.save(np.zeros((10, 10, 3), dtype=np.uint8), temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_nonexistent_file(self, converter):
        """Test error handling for nonexistent files."""
        with pytest.raises(SystemExit):
            converter.load("nonexistent_file.jpg")
    
    def test_invalid_file_paths(self, converter, sample_hwc_tensor):
        """Test error handling for invalid file paths."""
        with pytest.raises(SystemExit):
            converter.load(123)  # Not a string
        
        with pytest.raises(SystemExit):
            converter.save(sample_hwc_tensor, 456)  # Not a string
    
    @patch('sys.exit')
    def test_error_handling_doesnt_exit_in_tests(self, mock_exit, converter):
        """Test that error handling calls sys.exit but we can mock it."""
        mock_exit.return_value = None
        
        # This should trigger an error that calls sys.exit(1)
        converter.load("nonexistent.jpg")
        
        # Verify sys.exit was called
        mock_exit.assert_called_once_with(1)


class TestRoundTripOperations:
    """Test round-trip operations between different tensor formats."""
    
    @pytest.fixture
    def converter(self):
        return ImageTensorConverter()
    
    def test_hwc_nchw_roundtrip(self, converter):
        """Test HWC -> NCHW -> HWC round trip conversion."""
        original = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        
        # HWC -> NCHW
        nchw = converter.to_nchw(original)
        
        # NCHW -> HWC
        result = converter.to_hwc(nchw)
        
        assert np.array_equal(original, result)
    
    def test_nchw_nhwc_roundtrip(self, converter):
        """Test NCHW -> NHWC -> NCHW conversion."""
        original = np.random.randint(0, 255, (1, 3, 24, 32), dtype=np.uint8)
        
        # NCHW -> NHWC
        nhwc = converter.to_nhwc(original)
        
        # Convert back through HWC (since we don't have direct NHWC->NCHW)
        hwc = converter.to_hwc(nhwc)
        nchw_result = converter.to_nchw(hwc)
        
        assert np.array_equal(original, nchw_result)
    
    def test_resize_consistency(self, converter):
        """Test resize operation consistency."""
        original = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Resize up then down
        resized_up = converter.resize(original, (128, 128))
        resized_back = converter.resize(resized_up, (64, 64))
        
        # Should have same dimensions as original
        assert resized_back.shape == original.shape
        assert resized_back.dtype == original.dtype
        
        # Values might differ due to interpolation, but should be in valid range
        assert resized_back.min() >= 0
        assert resized_back.max() <= 255


if __name__ == "__main__":
    pytest.main([__file__, "-v"])