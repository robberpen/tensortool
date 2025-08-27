"""
Tensor Image Utilities - A lightweight image-tensor conversion library.
Designed for embedded systems with minimal dependencies.
"""

from .converter import ImageTensorConverter
from .exceptions import TensorImageError, UnsupportedFormatError, InvalidDimensionError

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = ["ImageTensorConverter", "TensorImageError", "UnsupportedFormatError", "InvalidDimensionError"]