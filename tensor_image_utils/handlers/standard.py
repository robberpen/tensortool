"""
Handler for standard image formats (.jpg, .bmp, .webp) using OpenCV.
All images are converted to RGB format for consistency.
"""

import os
from typing import Set
import numpy as np
import cv2

from .base import FormatHandler
from ..exceptions import FileProcessingError, CorruptedDataError, InvalidDimensionError


class StandardImageHandler(FormatHandler):
    """Handler for standard image formats using OpenCV."""
    
    @property
    def supported_extensions(self) -> Set[str]:
        """Return supported standard image extensions."""
        return {"jpg", "jpeg", "bmp", "webp", "png"}
    
    def can_handle(self, filepath: str) -> bool:
        """Check if file has supported extension."""
        extension = self._get_file_extension(filepath)
        return extension in self.supported_extensions
    
    def load(self, filepath: str, **kwargs) -> np.ndarray:
        """
        Load standard image file using OpenCV.
        
        Args:
            filepath: Path to image file
            
        Returns:
            np.ndarray: Image data in HWC RGB format (uint8)
            
        Raises:
            FileProcessingError: If file doesn't exist or cannot be read
            CorruptedDataError: If image data is corrupted
        """
        if not os.path.exists(filepath):
            raise FileProcessingError(f"File not found: {filepath}")
        
        try:
            # Load image with OpenCV (BGR format)
            image_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                raise CorruptedDataError(f"Cannot decode image file: {filepath}")
            
            # Convert BGR to RGB for consistency
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Validate loaded data
            self._validate_image_data(image_rgb, expected_channels=3)
            
            return image_rgb
            
        except cv2.error as e:
            raise CorruptedDataError(f"OpenCV error loading {filepath}: {str(e)}")
        except Exception as e:
            raise FileProcessingError(f"Unexpected error loading {filepath}: {str(e)}")
    
    def save(self, data: np.ndarray, filepath: str) -> None:
        """
        Save image data to standard format file.
        
        Args:
            data: Image data in HWC RGB format (uint8)
            filepath: Target file path
            
        Raises:
            InvalidDimensionError: If data format is invalid
            FileProcessingError: If file cannot be saved
        """
        # Validate input data
        self._validate_image_data(data, expected_channels=3)
        
        try:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            
            # Determine encoding parameters based on format
            extension = self._get_file_extension(filepath)
            encode_params = self._get_encode_params(extension)
            
            # Save image
            success = cv2.imwrite(filepath, image_bgr, encode_params)
            
            if not success:
                raise FileProcessingError(f"Failed to save image: {filepath}")
                
        except cv2.error as e:
            raise FileProcessingError(f"OpenCV error saving {filepath}: {str(e)}")
        except Exception as e:
            raise FileProcessingError(f"Unexpected error saving {filepath}: {str(e)}")
    
    def _get_encode_params(self, extension: str) -> list:
        """Get encoding parameters for different image formats."""
        if extension in ["jpg", "jpeg"]:
            # JPEG quality 95%
            return [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif extension == "webp":
            # WebP quality 95%
            return [cv2.IMWRITE_WEBP_QUALITY, 95]
        elif extension in ["bmp", "png"]:
            # No compression parameters needed
            return []
        else:
            return []