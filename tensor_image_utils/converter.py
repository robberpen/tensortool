"""
Main ImageTensorConverter class - the primary API interface.
Provides unified access to all format handlers and tensor operations.
"""

import sys
import os
from typing import Set, Tuple, Optional, Union
import numpy as np
import cv2

from .handlers.base import FormatHandler
from .handlers.standard import StandardImageHandler
from .handlers.numpy_handler import NumpyHandler  
from .handlers.raw_binary import RawBinaryHandler
from .exceptions import (
    TensorImageError, UnsupportedFormatError, 
    InvalidDimensionError, FileProcessingError
)


class ImageTensorConverter:
    """
    Main converter class for image-tensor operations.
    
    Supports formats: .jpg, .jpeg, .bmp, .webp, .png, .npy, .rgb888, .bgr888
    All operations maintain RGB channel order and uint8 data type.
    """
    
    def __init__(self):
        """Initialize the converter with all format handlers."""
        self._handlers = self._initialize_handlers()
        self._supported_formats = self._get_all_supported_formats()
    
    def _initialize_handlers(self) -> list[FormatHandler]:
        """Initialize all format handlers."""
        return [
            StandardImageHandler(),
            NumpyHandler(),
            RawBinaryHandler()
        ]
    
    def _get_all_supported_formats(self) -> Set[str]:
        """Get all supported file extensions from all handlers."""
        formats = set()
        for handler in self._handlers:
            formats.update(handler.supported_extensions)
        return formats
    
    def _get_handler(self, filepath: str) -> FormatHandler:
        """
        Get appropriate handler for the given file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            FormatHandler: Handler that can process the file
            
        Raises:
            UnsupportedFormatError: If no handler supports the file
        """
        for handler in self._handlers:
            if handler.can_handle(filepath):
                return handler
        
        extension = self._get_file_extension(filepath)
        raise UnsupportedFormatError(
            f"Unsupported format: .{extension}. "
            f"Supported formats: {sorted(self._supported_formats)}"
        )
    
    def _get_file_extension(self, filepath: str) -> str:
        """Extract file extension from filepath (without dot)."""
        return filepath.lower().split('.')[-1] if '.' in filepath else ""
    
    def _handle_error(self, error: Exception, context: str) -> None:
        """
        Handle errors with consistent error reporting and exit.
        
        Args:
            error: The exception that occurred
            context: Context description for the error
        """
        print(f"ERROR [{context}]: {error}", file=sys.stderr)
        sys.exit(1)
    
    def load(self, filepath: str, **kwargs) -> np.ndarray:
        """
        Load image from file and return as numpy array.
        
        Args:
            filepath: Path to image file
            **kwargs: Additional parameters (e.g., width, height for raw formats)
            
        Returns:
            np.ndarray: Image data in HWC RGB format (uint8)
            
        Note:
            For raw binary formats (.rgb888, .bgr888), you can provide:
            - width: Image width
            - height: Image height
        """
        try:
            if not isinstance(filepath, str):
                raise FileProcessingError("filepath must be a string")
            
            handler = self._get_handler(filepath)
            return handler.load(filepath, **kwargs)
            
        except TensorImageError as e:
            self._handle_error(e, f"loading {filepath}")
        except Exception as e:
            self._handle_error(e, f"unexpected error loading {filepath}")
    
    def save(self, tensor: np.ndarray, filepath: str) -> None:
        """
        Save numpy array as image file.
        
        Args:
            tensor: Image data in HWC RGB format (uint8)
            filepath: Target file path
        """
        try:
            if not isinstance(tensor, np.ndarray):
                raise InvalidDimensionError("tensor must be a numpy array")
            
            if not isinstance(filepath, str):
                raise FileProcessingError("filepath must be a string")
            
            handler = self._get_handler(filepath)
            handler.save(tensor, filepath)
            
        except TensorImageError as e:
            self._handle_error(e, f"saving {filepath}")
        except Exception as e:
            self._handle_error(e, f"unexpected error saving {filepath}")
    
    def to_nchw(self, hwc_tensor: np.ndarray) -> np.ndarray:
        """
        Convert tensor from HWC to NCHW format.
        
        Args:
            hwc_tensor: Input tensor in HWC format
            
        Returns:
            np.ndarray: Tensor in NCHW format (1, C, H, W)
        """
        try:
            if not isinstance(hwc_tensor, np.ndarray):
                raise InvalidDimensionError("Input must be a numpy array")
            
            if hwc_tensor.ndim != 3:
                raise InvalidDimensionError(f"Expected 3D tensor (HWC), got {hwc_tensor.ndim}D")
            
            # HWC -> CHW -> NCHW (add batch dimension)
            chw_tensor = np.transpose(hwc_tensor, (2, 0, 1))
            nchw_tensor = np.expand_dims(chw_tensor, axis=0)
            
            return nchw_tensor
            
        except TensorImageError as e:
            self._handle_error(e, "converting HWC to NCHW")
        except Exception as e:
            self._handle_error(e, "unexpected error in HWC to NCHW conversion")
    
    def to_nhwc(self, nchw_tensor: np.ndarray) -> np.ndarray:
        """
        Convert tensor from NCHW to NHWC format.
        
        Args:
            nchw_tensor: Input tensor in NCHW format
            
        Returns:
            np.ndarray: Tensor in NHWC format (N, H, W, C)
        """
        try:
            if not isinstance(nchw_tensor, np.ndarray):
                raise InvalidDimensionError("Input must be a numpy array")
            
            if nchw_tensor.ndim != 4:
                raise InvalidDimensionError(f"Expected 4D tensor (NCHW), got {nchw_tensor.ndim}D")
            
            # NCHW -> NHWC
            nhwc_tensor = np.transpose(nchw_tensor, (0, 2, 3, 1))
            
            return nhwc_tensor
            
        except TensorImageError as e:
            self._handle_error(e, "converting NCHW to NHWC")
        except Exception as e:
            self._handle_error(e, "unexpected error in NCHW to NHWC conversion")
    
    def to_hwc(self, tensor: np.ndarray) -> np.ndarray:
        """
        Convert tensor to HWC format from various input formats.
        
        Args:
            tensor: Input tensor (can be HWC, CHW, NCHW, or NHWC)
            
        Returns:
            np.ndarray: Tensor in HWC format
        """
        try:
            if not isinstance(tensor, np.ndarray):
                raise InvalidDimensionError("Input must be a numpy array")
            
            if tensor.ndim == 3:
                # Assume input is already HWC or needs CHW->HWC conversion
                # Heuristic: if last dimension is 1, 3, or 4, likely HWC
                if tensor.shape[2] in [1, 3, 4]:
                    return tensor  # Already HWC
                else:
                    # Assume CHW -> HWC
                    return np.transpose(tensor, (1, 2, 0))
            
            elif tensor.ndim == 4:
                if tensor.shape[0] == 1:
                    # Single batch - remove batch dimension
                    single_image = tensor[0]
                    return self.to_hwc(single_image)  # Recursive call
                else:
                    raise InvalidDimensionError(
                        f"Cannot convert batch with size {tensor.shape[0]} to single HWC tensor"
                    )
            
            else:
                raise InvalidDimensionError(f"Unsupported tensor dimensions: {tensor.ndim}D")
                
        except TensorImageError as e:
            self._handle_error(e, "converting to HWC format")
        except Exception as e:
            self._handle_error(e, "unexpected error in HWC conversion")
    
    def resize(self, tensor: np.ndarray, target_size: Tuple[int, int], 
               interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize tensor to target size.
        
        Args:
            tensor: Input tensor in HWC format
            target_size: Target size as (width, height)
            interpolation: OpenCV interpolation method
            
        Returns:
            np.ndarray: Resized tensor in HWC format
        """
        try:
            if not isinstance(tensor, np.ndarray):
                raise InvalidDimensionError("Input must be a numpy array")
            
            if tensor.ndim != 3:
                raise InvalidDimensionError(f"Expected 3D tensor (HWC), got {tensor.ndim}D")
            
            if len(target_size) != 2:
                raise InvalidDimensionError("target_size must be (width, height)")
            
            target_width, target_height = target_size
            
            if target_width <= 0 or target_height <= 0:
                raise InvalidDimensionError(f"Invalid target size: {target_width}x{target_height}")
            
            # OpenCV resize expects (width, height)
            resized = cv2.resize(tensor, (target_width, target_height), interpolation=interpolation)
            
            # cv2.resize might return 2D array for single channel, ensure 3D
            if resized.ndim == 2:
                resized = np.expand_dims(resized, axis=2)
            
            return resized
            
        except cv2.error as e:
            self._handle_error(e, f"resizing tensor to {target_size}")
        except TensorImageError as e:
            self._handle_error(e, f"resizing tensor to {target_size}")
        except Exception as e:
            self._handle_error(e, f"unexpected error resizing tensor to {target_size}")
    
    def get_supported_formats(self) -> Set[str]:
        """
        Get all supported file formats.
        
        Returns:
            Set[str]: Set of supported file extensions
        """
        return self._supported_formats.copy()
    
    def get_tensor_info(self, tensor: np.ndarray) -> dict:
        """
        Get information about a tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            dict: Tensor information
        """
        try:
            if not isinstance(tensor, np.ndarray):
                raise InvalidDimensionError("Input must be a numpy array")
            
            return {
                'shape': tensor.shape,
                'dtype': str(tensor.dtype),
                'ndim': tensor.ndim,
                'size': tensor.size,
                'memory_usage_bytes': tensor.nbytes,
                'min_value': float(tensor.min()),
                'max_value': float(tensor.max()),
                'mean_value': float(tensor.mean())
            }
            
        except Exception as e:
            self._handle_error(e, "getting tensor information")