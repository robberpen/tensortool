"""
Handler for raw binary formats (.rgb888, .bgr888).
Supports both filename-encoded dimensions and explicit width/height parameters.
"""

import os
import re
from typing import Set, Tuple, Optional
import numpy as np

from .base import FormatHandler
from ..exceptions import FileProcessingError, InvalidDimensionError


class RawBinaryHandler(FormatHandler):
    """Handler for raw binary image formats."""
    
    @property
    def supported_extensions(self) -> Set[str]:
        """Return supported raw binary extensions."""
        return {"rgb888", "bgr888"}
    
    def can_handle(self, filepath: str) -> bool:
        """Check if file has supported raw binary extension."""
        extension = self._get_file_extension(filepath)
        return extension in self.supported_extensions
    
    def load(self, filepath: str, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
        """
        Load raw binary image file.
        
        Args:
            filepath: Path to raw binary file
            width: Image width (optional if encoded in filename)
            height: Image height (optional if encoded in filename)
            
        Returns:
            np.ndarray: Image data in HWC RGB format (uint8)
            
        Raises:
            FileProcessingError: If file doesn't exist or cannot be read
            InvalidDimensionError: If dimensions cannot be determined or are invalid
        """
        if not os.path.exists(filepath):
            raise FileProcessingError(f"File not found: {filepath}")
        
        # Determine image dimensions
        img_width, img_height = self._get_dimensions(filepath, width, height)
        
        # Determine format (RGB or BGR)
        extension = self._get_file_extension(filepath)
        is_bgr = (extension == "bgr888")
        
        try:
            # Read raw binary data
            with open(filepath, 'rb') as f:
                raw_data = f.read()
            
            # Validate file size
            expected_size = img_width * img_height * 3  # 3 bytes per pixel
            if len(raw_data) != expected_size:
                raise InvalidDimensionError(
                    f"File size mismatch: expected {expected_size} bytes "
                    f"for {img_width}x{img_height} image, got {len(raw_data)} bytes"
                )
            
            # Convert to numpy array
            image_data = np.frombuffer(raw_data, dtype=np.uint8)
            image_data = image_data.reshape((img_height, img_width, 3))
            
            # Convert BGR to RGB if necessary
            if is_bgr:
                # BGR888 -> RGB888 conversion
                image_data = image_data[:, :, ::-1]  # Reverse channel order
            
            # Validate final data
            self._validate_image_data(image_data, expected_channels=3)
            
            return image_data
            
        except IOError as e:
            raise FileProcessingError(f"Cannot read raw binary file {filepath}: {str(e)}")
        except Exception as e:
            raise FileProcessingError(f"Unexpected error loading {filepath}: {str(e)}")
    
    def save(self, data: np.ndarray, filepath: str) -> None:
        """
        Save image data as raw binary file.
        
        Args:
            data: Image data in HWC RGB format (uint8)
            filepath: Target file path
            
        Raises:
            InvalidDimensionError: If data format is invalid
            FileProcessingError: If file cannot be saved
        """
        # Validate input data
        self._validate_image_data(data, expected_channels=3)
        
        extension = self._get_file_extension(filepath)
        
        try:
            # Convert to target format
            if extension == "bgr888":
                # Convert RGB to BGR
                output_data = data[:, :, ::-1]  # Reverse channel order
            else:  # rgb888
                output_data = data
            
            # Save as raw binary
            with open(filepath, 'wb') as f:
                f.write(output_data.tobytes())
                
        except IOError as e:
            raise FileProcessingError(f"Cannot save raw binary file {filepath}: {str(e)}")
        except Exception as e:
            raise FileProcessingError(f"Unexpected error saving {filepath}: {str(e)}")
    
    def _get_dimensions(self, filepath: str, width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
        """
        Determine image dimensions from filename or parameters.
        
        Args:
            filepath: Path to the file
            width: Explicitly provided width
            height: Explicitly provided height
            
        Returns:
            Tuple[int, int]: (width, height)
            
        Raises:
            InvalidDimensionError: If dimensions cannot be determined
        """
        # If both width and height are provided, use them
        if width is not None and height is not None:
            if width <= 0 or height <= 0:
                raise InvalidDimensionError(f"Invalid dimensions: {width}x{height}")
            return width, height
        
        # Try to parse dimensions from filename
        parsed_width, parsed_height = self._parse_dimensions_from_filename(filepath)
        
        # Use parsed dimensions if available, otherwise use provided parameters
        final_width = width if width is not None else parsed_width
        final_height = height if height is not None else parsed_height
        
        if final_width is None or final_height is None:
            raise InvalidDimensionError(
                f"Cannot determine image dimensions for {filepath}. "
                f"Either encode in filename (e.g., image_640x480.rgb888) or "
                f"provide width and height parameters."
            )
        
        if final_width <= 0 or final_height <= 0:
            raise InvalidDimensionError(f"Invalid dimensions: {final_width}x{final_height}")
        
        return final_width, final_height
    
    def _parse_dimensions_from_filename(self, filepath: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse image dimensions from filename.
        
        Expected format: filename_WIDTHxHEIGHT.extension
        Example: image_640x480.rgb888
        
        Args:
            filepath: Path to the file
            
        Returns:
            Tuple[Optional[int], Optional[int]]: (width, height) or (None, None)
        """
        # Extract filename without path
        filename = os.path.basename(filepath)
        
        # Regex pattern to match dimensions in filename
        # Matches: _640x480. or _640x480_ or _640x480 at end
        pattern = r'_(\d+)x(\d+)(?:[._]|$)'
        
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            try:
                width = int(match.group(1))
                height = int(match.group(2))
                return width, height
            except ValueError:
                pass
        
        return None, None
    
    def get_file_info(self, filepath: str, width: Optional[int] = None, height: Optional[int] = None) -> dict:
        """
        Get information about a raw binary file without loading it.
        
        Args:
            filepath: Path to the file
            width: Image width (optional)
            height: Image height (optional)
            
        Returns:
            dict: File information including dimensions, format, and size
        """
        if not os.path.exists(filepath):
            raise FileProcessingError(f"File not found: {filepath}")
        
        try:
            # Get dimensions
            img_width, img_height = self._get_dimensions(filepath, width, height)
            
            # Get file size
            file_size = os.path.getsize(filepath)
            expected_size = img_width * img_height * 3
            
            # Determine format
            extension = self._get_file_extension(filepath)
            
            return {
                'filepath': filepath,
                'format': extension.upper(),
                'width': img_width,
                'height': img_height,
                'channels': 3,
                'file_size_bytes': file_size,
                'expected_size_bytes': expected_size,
                'size_match': file_size == expected_size,
                'pixels': img_width * img_height
            }
            
        except Exception as e:
            raise FileProcessingError(f"Cannot get file info for {filepath}: {str(e)}")