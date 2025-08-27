"""
Custom exception classes for tensor-image-utils.
All exceptions trigger sys.exit(1) when caught by the main converter.
"""


class TensorImageError(Exception):
    """Base exception class for all tensor-image-utils errors."""
    pass


class UnsupportedFormatError(TensorImageError):
    """Raised when attempting to process an unsupported file format."""
    pass


class InvalidDimensionError(TensorImageError):
    """Raised when image dimensions are invalid or cannot be determined."""
    pass


class CorruptedDataError(TensorImageError):
    """Raised when file data is corrupted or cannot be properly decoded."""
    pass


class FileProcessingError(TensorImageError):
    """Raised when file I/O operations fail."""
    pass