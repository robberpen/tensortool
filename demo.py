#!/usr/bin/env python3
"""
Demo script for tensor-image-utils with command-line interface.
Supports loading, converting, resizing, and saving images in various formats.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np

# Add the current directory to the path to import the module
sys.path.insert(0, str(Path(__file__).parent))

from tensor_image_utils import ImageTensorConverter


def parse_dimensions(dim_str):
    """
    Parse dimension string in format 'H,W' or 'HxW'.
    
    Args:
        dim_str: Dimension string like '320,320' or '640x480'
        
    Returns:
        tuple: (height, width) as integers
        
    Raises:
        ValueError: If format is invalid
    """
    if not dim_str:
        return None
    
    # Handle both comma and 'x' separators
    if ',' in dim_str:
        parts = dim_str.split(',')
    elif 'x' in dim_str.lower():
        parts = dim_str.lower().split('x')
    else:
        raise ValueError(f"Invalid dimension format: {dim_str}. Use 'H,W' or 'HxW'")
    
    if len(parts) != 2:
        raise ValueError(f"Dimension must have exactly 2 values, got {len(parts)}")
    
    try:
        height = int(parts[0].strip())
        width = int(parts[1].strip())
        
        if height <= 0 or width <= 0:
            raise ValueError(f"Dimensions must be positive, got {height}x{width}")
        
        return height, width
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Dimension values must be integers: {dim_str}")
        raise


def validate_file_path(filepath, must_exist=True):
    """
    Validate file path.
    
    Args:
        filepath: Path to validate
        must_exist: Whether file must exist
        
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist=True
        ValueError: If path is invalid
    """
    if not filepath:
        raise ValueError("File path cannot be empty")
    
    path = Path(filepath)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not must_exist:
        # Check if parent directory exists
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {parent_dir}")
            except OSError as e:
                raise ValueError(f"Cannot create directory {parent_dir}: {e}")


def load_image(converter, filepath, shape=None):
    """
    Load image using the converter.
    
    Args:
        converter: ImageTensorConverter instance
        filepath: Path to image file
        shape: Optional (height, width) for raw formats
        
    Returns:
        np.ndarray: Loaded image tensor
    """
    print(f"Loading image from: {filepath}")
    
    # Check if file needs explicit dimensions (raw formats)
    extension = Path(filepath).suffix.lower().lstrip('.')
    
    if extension in ['rgb888', 'bgr888'] and shape:
        height, width = shape
        print(f"Loading raw binary format with dimensions: {height}x{width}")
        tensor = converter.load(filepath, width=width, height=height)
    else:
        tensor = converter.load(filepath)
    
    print(f"Loaded tensor: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"Value range: [{tensor.min()}, {tensor.max()}]")
    
    return tensor


def save_image(converter, tensor, filepath):
    """
    Save image using the converter.
    
    Args:
        converter: ImageTensorConverter instance
        tensor: Image tensor to save
        filepath: Target file path
    """
    print(f"Saving image to: {filepath}")
    
    # Validate tensor format
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        raise ValueError(f"Tensor must be HWC RGB format, got shape {tensor.shape}")
    
    converter.save(tensor, filepath)
    
    # Verify saved file
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        print(f"Saved successfully: {filepath} ({file_size} bytes)")
    else:
        raise RuntimeError(f"Failed to save file: {filepath}")


def resize_image(converter, tensor, target_size):
    """
    Resize image tensor.
    
    Args:
        converter: ImageTensorConverter instance
        tensor: Input tensor
        target_size: (height, width) tuple
        
    Returns:
        np.ndarray: Resized tensor
    """
    height, width = target_size
    print(f"Resizing from {tensor.shape[:2]} to ({height}, {width})")
    
    # Convert to (width, height) for cv2.resize
    target_cv2 = (width, height)
    resized = converter.resize(tensor, target_cv2)
    
    print(f"Resized tensor: shape={resized.shape}")
    return resized


def get_tensor_info(converter, tensor):
    """Print detailed tensor information."""
    info = converter.get_tensor_info(tensor)
    print(f"\nTensor Information:")
    print(f"  Shape: {info['shape']}")
    print(f"  Data type: {info['dtype']}")
    print(f"  Dimensions: {info['ndim']}D")
    print(f"  Total elements: {info['size']}")
    print(f"  Memory usage: {info['memory_usage_bytes']} bytes ({info['memory_usage_bytes']/1024:.1f} KB)")
    print(f"  Value range: [{info['min_value']}, {info['max_value']}]")
    print(f"  Mean value: {info['mean_value']:.2f}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Demo script for tensor-image-utils library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load JPG, resize to 320x320, save as NPY
  python demo.py --load ./coco502.jpg --resize 320,320 --save ./coco320x320.npy
  
  # Load NPY with specific shape, save as BMP
  python demo.py --load ./coco320x320.npy --shape 320,320 --save ./coco320x320.bmp
  
  # Load raw RGB888 file with shape, resize and save as WebP
  python demo.py --load image_640x480.rgb888 --shape 480,640 --resize 224,224 --save output.webp
  
  # Simple format conversion
  python demo.py --load input.png --save output.jpg
  
Supported formats:
  Input:  .jpg, .jpeg, .png, .bmp, .webp, .npy, .rgb888, .bgr888
  Output: .jpg, .jpeg, .png, .bmp, .webp, .npy, .rgb888, .bgr888
        """
    )
    
    parser.add_argument('--load', required=True, 
                       help='Input file path (supports .jpg, .npy, .rgb888, etc.)')
    
    parser.add_argument('--save', required=True,
                       help='Output file path (format determined by extension)')
    
    parser.add_argument('--shape', 
                       help='Specify Height,Width for input file (e.g., "320,320" or "480x640"). Required for raw binary formats.')
    
    parser.add_argument('--resize',
                       help='Resize to Height,Width (e.g., "640,640" or "224x224")')
    
    parser.add_argument('--info', action='store_true',
                       help='Display detailed tensor information')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        if args.verbose:
            print("Initializing ImageTensorConverter...")
        
        converter = ImageTensorConverter()
        
        if args.verbose:
            supported = sorted(converter.get_supported_formats())
            print(f"Supported formats: {supported}")
        
        # Validate file paths
        validate_file_path(args.load, must_exist=True)
        validate_file_path(args.save, must_exist=False)
        
        # Parse dimensions if provided
        shape = None
        if args.shape:
            shape = parse_dimensions(args.shape)
            if args.verbose:
                print(f"Input shape specified: {shape[0]}x{shape[1]} (HxW)")
        
        resize_dims = None
        if args.resize:
            resize_dims = parse_dimensions(args.resize)
            if args.verbose:
                print(f"Resize target: {resize_dims[0]}x{resize_dims[1]} (HxW)")
        
        # Load image
        tensor = load_image(converter, args.load, shape)
        
        # Display tensor info if requested
        if args.info:
            get_tensor_info(converter, tensor)
        
        # Resize if requested
        if resize_dims:
            tensor = resize_image(converter, tensor, resize_dims)
        
        # Save image
        save_image(converter, tensor, args.save)
        
        print(f"\n✓ Successfully processed: {args.load} → {args.save}")
        
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()