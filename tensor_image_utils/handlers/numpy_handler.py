"""
Handler for NumPy format (.npy) files.
Handles different data types and converts them to uint8 [0,255] range.
"""

import os
from typing import Set
import numpy as np

from .base import FormatHandler
from ..exceptions import FileProcessingError, CorruptedDataError, InvalidDimensionError


class NumpyHandler(FormatHandler):
    """Handler for NumPy .npy format files."""
    
    @property
    def supported_extensions(self) -> Set[str]:
        """Return supported NumPy extensions."""
        return {"npy"}
    
    def can_handle(self, filepath: str) -> bool:
        """Check if file has .npy extension."""
        extension = self._get_file_extension(filepath)
        return extension in self.supported_extensions
    
    def load(self, filepath: str, **kwargs) -> np.ndarray:
        """
        Load NumPy array from .npy file and convert to uint8.
        
        Args:
            filepath: Path to .npy file
            
        Returns:
            np.ndarray: Image data in HWC format (RGB, uint8)
            
        Raises:
            FileProcessingError: If file doesn't exist or cannot be read
            CorruptedDataError: If data format is invalid
            InvalidDimensionError: If array dimensions are invalid
        """
        if not os.path.exists(filepath):
            raise FileProcessingError(f"File not found: {filepath}")
        
        try:
            # Load numpy array
            data = np.load(filepath)
            
            # Convert to standard format
            processed_data = self._process_numpy_data(data)
            
            # Validate final data
            self._validate_image_data(processed_data)
            
            return processed_data
            
        except (IOError, OSError) as e:
            raise FileProcessingError(f"Cannot read .npy file {filepath}: {str(e)}")
        except ValueError as e:
            raise CorruptedDataError(f"Invalid NumPy data in {filepath}: {str(e)}")
        except Exception as e:
            raise FileProcessingError(f"Unexpected error loading {filepath}: {str(e)}")
    
    def save(self, data: np.ndarray, filepath: str) -> None:
        """
        Save image data as .npy file.
        
        Args:
            data: Image data in HWC format (uint8)
            filepath: Target file path
            
        Raises:
            InvalidDimensionError: If data format is invalid
            FileProcessingError: If file cannot be saved
        """
        # Validate input data
        self._validate_image_data(data)
        
        try:
            # Save as numpy array (keeping uint8 format)
            np.save(filepath, data)
            
        except (IOError, OSError) as e:
            raise FileProcessingError(f"Cannot save .npy file {filepath}: {str(e)}")
        except Exception as e:
            raise FileProcessingError(f"Unexpected error saving {filepath}: {str(e)}")
    
    def _process_numpy_data(self, data: np.ndarray) -> np.ndarray:
        """
        Process NumPy data and convert to standard uint8 HWC format.
        
        Args:
            data: Raw NumPy array from file
            
        Returns:
            np.ndarray: Processed data in HWC uint8 format
            
        Raises:
            InvalidDimensionError: If data dimensions are invalid
            CorruptedDataError: If data cannot be converted to image format
        """
        # Handle different array dimensions
        if data.ndim == 2:
            # 2D array - treat as grayscale, convert to 3-channel
            processed = self._convert_grayscale_to_rgb(data)
        elif data.ndim == 3:
            # 3D array - check if HWC or CHW format
            processed = self._process_3d_array(data)
        elif data.ndim == 4:
            # 4D array - assume NCHW or NHWC, take first batch
            processed = self._process_4d_array(data)
        else:
            raise InvalidDimensionError(f"Unsupported array dimensions: {data.ndim}D")
        
        # Convert data type to uint8
        processed = self._convert_to_uint8(processed)
        
        return processed
    
    def _convert_grayscale_to_rgb(self, data: np.ndarray) -> np.ndarray:
        """Convert 2D grayscale to 3D RGB."""
        # Repeat grayscale values across 3 channels
        return np.repeat(data[:, :, np.newaxis], 3, axis=2)
    
    def _process_3d_array(self, data: np.ndarray) -> np.ndarray:
        """Process 3D array and determine if it's HWC or CHW format."""
        h, w, c = data.shape
        
        # Heuristic: if last dimension is 1, 3, or 4, likely HWC
        # if first dimension is 1, 3, or 4, likely CHW
        if c in [1, 3, 4] and h > c and w > c:
            # Likely HWC format
            if c == 1:
                # Convert single channel to RGB
                return np.repeat(data, 3, axis=2)
            elif c == 3:
                # Already RGB
                return data
            elif c == 4:
                # RGBA - drop alpha channel
                return data[:, :, :3]
        elif data.shape[0] in [1, 3, 4] and h > data.shape[0] and w > data.shape[0]:
            # Likely CHW format - transpose to HWC
            if data.shape[0] == 1:
                # Single channel - convert to RGB
                transposed = np.transpose(data, (1, 2, 0))
                return np.repeat(transposed, 3, axis=2)
            elif data.shape[0] == 3:
                # RGB channels
                return np.transpose(data, (1, 2, 0))
            elif data.shape[0] == 4:
                # RGBA - drop alpha and transpose
                rgb_data = data[:3, :, :]
                return np.transpose(rgb_data, (1, 2, 0))
        
        raise InvalidDimensionError(f"Cannot determine format for shape {data.shape}")
    
    def _process_4d_array(self, data: np.ndarray) -> np.ndarray:
        """Process 4D array (batch format) and extract first image."""
        if data.shape[0] == 0:
            raise InvalidDimensionError("Empty batch dimension")
        
        # Take first image from batch
        first_image = data[0]
        
        # Process as 3D array
        return self._process_3d_array(first_image)
    
    def _convert_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """
        Convert data to uint8 format with proper scaling.
        
        Args:
            data: Input array of any numeric type
            
        Returns:
            np.ndarray: Data converted to uint8 [0, 255]
        """
        if data.dtype == np.uint8:
            return data
        
        # Handle different data types and ranges
        if data.dtype in [np.float32, np.float64]:
            # Floating point data
            if data.min() >= 0 and data.max() <= 1.0:
                # Assume [0, 1] range
                return (data * 255).astype(np.uint8)
            elif data.min() >= -1.0 and data.max() <= 1.0:
                # Assume [-1, 1] range
                return ((data + 1) * 127.5).astype(np.uint8)
            else:
                # Unknown range - normalize to [0, 1] then scale
                data_min, data_max = data.min(), data.max()
                if data_max > data_min:
                    normalized = (data - data_min) / (data_max - data_min)
                    return (normalized * 255).astype(np.uint8)
                else:
                    return np.zeros_like(data, dtype=np.uint8)
        
        elif data.dtype in [np.int8, np.int16, np.int32, np.int64]:
            # Signed integer data
            if data.min() >= 0 and data.max() <= 255:
                return data.astype(np.uint8)
            else:
                # Normalize to [0, 255]
                data_min, data_max = data.min(), data.max()
                if data_max > data_min:
                    normalized = (data - data_min) / (data_max - data_min)
                    return (normalized * 255).astype(np.uint8)
                else:
                    return np.zeros_like(data, dtype=np.uint8)
        
        elif data.dtype in [np.uint16, np.uint32, np.uint64]:
            # Unsigned integer data
            if data.max() <= 255:
                return data.astype(np.uint8)
            else:
                # Scale down to [0, 255]
                return ((data / data.max()) * 255).astype(np.uint8)
        
        else:
            raise CorruptedDataError(f"Unsupported data type: {data.dtype}")