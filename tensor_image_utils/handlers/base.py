"""
Abstract base class for format handlers.
Defines the interface that all format handlers must implement.
"""

from abc import ABC, abstractmethod
from typing import Set, Optional
import numpy as np

from ..exceptions import InvalidDimensionError


class FormatHandler(ABC):
    """Abstract base class for all format handlers."""
    
    @property
    @abstractmethod
    def supported_extensions(self) -> Set[str]:
        """Return set of supported file extensions (without dot)."""
        pass
    
    @abstractmethod
    def can_handle(self, filepath: str) -> bool:
        """
        Check if this handler can process the given file.
        
        Args:
            filepath: Path to the file to check
            
        Returns:
            True if this handler can process the file, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str, **kwargs) -> np.ndarray:
        """
        Load image data from file and return as numpy array.
        
        Args:
            filepath: Path to the image file
            **kwargs: Additional parameters specific to the handler
            
        Returns:
            np.ndarray: Image data in HWC format (RGB, uint8)
            
        Raises:
            FileProcessingError: If file cannot be loaded
            CorruptedDataError: If file data is corrupted
            InvalidDimensionError: If image dimensions are invalid
        """
        pass
    
    @abstractmethod 
    def save(self, data: np.ndarray, filepath: str) -> None:
        """
        Save numpy array as image file.
        
        Args:
            data: Image data as numpy array (HWC format, RGB, uint8)
            filepath: Target file path
            
        Raises:
            FileProcessingError: If file cannot be saved
            InvalidDimensionError: If data dimensions are invalid
        """
        pass
    
    def _validate_image_data(self, data: np.ndarray, expected_channels: Optional[int] = None) -> None:
        """
        Validate image data format and dimensions.
        
        Args:
            data: Image data to validate
            expected_channels: Expected number of channels (None to skip check)
            
        Raises:
            InvalidDimensionError: If data format is invalid
        """
        if data.ndim not in [2, 3]:
            raise InvalidDimensionError(f"Image data must be 2D or 3D, got {data.ndim}D")
        
        if data.ndim == 3:
            if expected_channels and data.shape[2] != expected_channels:
                raise InvalidDimensionError(
                    f"Expected {expected_channels} channels, got {data.shape[2]}"
                )
        
        if data.dtype != np.uint8:
            raise InvalidDimensionError(f"Image data must be uint8, got {data.dtype}")
    
    def _get_file_extension(self, filepath: str) -> str:
        """Extract file extension from filepath (without dot)."""
        return filepath.lower().split('.')[-1] if '.' in filepath else ""