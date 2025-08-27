#!/usr/bin/env python3
"""
Basic usage examples for tensor-image-utils.
Demonstrates common use cases and API patterns.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensor_image_utils import ImageTensorConverter


def example_basic_operations():
    """Demonstrate basic load/save operations."""
    print("=== Basic Operations Example ===")
    
    converter = ImageTensorConverter()
    
    # Show supported formats
    print(f"Supported formats: {sorted(converter.get_supported_formats())}")
    
    try:
        # Load a standard image file (assuming test data exists)
        test_image_path = "../tests/test_data/sample_640x480.jpg"
        if os.path.exists(test_image_path):
            image_tensor = converter.load(test_image_path)
            print(f"Loaded image shape: {image_tensor.shape}")
            print(f"Data type: {image_tensor.dtype}")
            
            # Get tensor information
            info = converter.get_tensor_info(image_tensor)
            print(f"Tensor info: {info}")
            
            # Save as different format
            converter.save(image_tensor, "output_test.bmp")
            print("Saved as BMP format")
            
            # Clean up
            if os.path.exists("output_test.bmp"):
                os.remove("output_test.bmp")
        else:
            print("Test image not found. Run test data generator first.")
            
    except Exception as e:
        print(f"Error in basic operations: {e}")


def example_tensor_conversions():
    """Demonstrate tensor format conversions."""
    print("\n=== Tensor Conversion Example ===")
    
    converter = ImageTensorConverter()
    
    # Create a sample image tensor (HWC format)
    sample_image = np.random.randint(0, 255, (64, 48, 3), dtype=np.uint8)
    print(f"Original HWC tensor: {sample_image.shape}")
    
    # Convert to NCHW (common for deep learning frameworks)
    nchw_tensor = converter.to_nchw(sample_image)
    print(f"NCHW tensor: {nchw_tensor.shape}")
    
    # Convert to NHWC (TensorFlow format)
    nhwc_tensor = converter.to_nhwc(nchw_tensor)
    print(f"NHWC tensor: {nhwc_tensor.shape}")
    
    # Convert back to HWC
    hwc_tensor = converter.to_hwc(nhwc_tensor)
    print(f"Back to HWC: {hwc_tensor.shape}")
    
    # Verify round-trip conversion
    if np.array_equal(sample_image, hwc_tensor):
        print("✓ Round-trip conversion successful!")
    else:
        print("✗ Round-trip conversion failed!")


def example_resize_operations():
    """Demonstrate tensor resizing."""
    print("\n=== Resize Operations Example ===")
    
    converter = ImageTensorConverter()
    
    # Create sample tensor
    original = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    print(f"Original size: {original.shape[:2]} (H, W)")
    
    # Resize to different dimensions
    sizes = [(224, 224), (64, 128), (300, 200)]
    
    for target_width, target_height in sizes:
        resized = converter.resize(original, (target_width, target_height))
        print(f"Resized to ({target_width}, {target_height}): shape = {resized.shape}")


def example_raw_binary_handling():
    """Demonstrate raw binary format handling."""
    print("\n=== Raw Binary Format Example ===")
    
    converter = ImageTensorConverter()
    
    # Create test data
    test_width, test_height = 32, 24
    test_image = np.random.randint(0, 255, (test_height, test_width, 3), dtype=np.uint8)
    
    # Save as raw RGB888 format
    rgb_filename = f"test_{test_width}x{test_height}.rgb888"
    converter.save(test_image, rgb_filename)
    print(f"Saved as raw RGB888: {rgb_filename}")
    
    # Load it back (dimensions encoded in filename)
    loaded_rgb = converter.load(rgb_filename)
    print(f"Loaded RGB888 shape: {loaded_rgb.shape}")
    
    # Save as raw BGR888 format
    bgr_filename = f"test_{test_width}x{test_height}.bgr888"
    converter.save(test_image, bgr_filename)
    print(f"Saved as raw BGR888: {bgr_filename}")
    
    # Load with explicit dimensions
    loaded_bgr = converter.load("test_32x24.bgr888", width=32, height=24)
    print(f"Loaded BGR888 shape: {loaded_bgr.shape}")
    
    # Verify data integrity (should be identical after BGR->RGB conversion)
    if np.array_equal(test_image, loaded_rgb):
        print("✓ RGB888 round-trip successful!")
    
    # Clean up
    for filename in [rgb_filename, bgr_filename]:
        if os.path.exists(filename):
            os.remove(filename)


def example_numpy_format_handling():
    """Demonstrate NumPy format handling with different data types."""
    print("\n=== NumPy Format Example ===")
    
    converter = ImageTensorConverter()
    
    # Test different NumPy data types
    test_cases = [
        ("uint8", np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)),
        ("float32_0_1", np.random.random((32, 32, 3)).astype(np.float32)),
        ("float32_neg1_1", (np.random.random((32, 32, 3)) * 2 - 1).astype(np.float32)),
        ("int16", np.random.randint(-1000, 1000, (32, 32, 3), dtype=np.int16)),
    ]
    
    for test_name, test_data in test_cases:
        filename = f"test_{test_name}.npy"
        
        # Save original data
        np.save(filename, test_data)
        print(f"\nSaved {test_name}: shape={test_data.shape}, dtype={test_data.dtype}")
        print(f"  Original range: [{test_data.min():.3f}, {test_data.max():.3f}]")
        
        # Load and convert to uint8
        loaded = converter.load(filename)
        print(f"  Loaded as uint8: shape={loaded.shape}, dtype={loaded.dtype}")
        print(f"  Converted range: [{loaded.min()}, {loaded.max()}]")
        
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)


def example_error_handling():
    """Demonstrate error handling behavior."""
    print("\n=== Error Handling Example ===")
    
    converter = ImageTensorConverter()
    
    print("Note: The following operations will cause sys.exit(1) in real usage:")
    print("- Loading nonexistent file")
    print("- Using unsupported format")
    print("- Invalid tensor dimensions")
    print("- Corrupted data")
    
    # In real usage, these would call sys.exit(1):
    # converter.load("nonexistent.jpg")           # FileNotFoundError
    # converter.load("file.unknown")              # UnsupportedFormatError
    # converter.to_nchw(np.array([1, 2, 3]))     # InvalidDimensionError


def main():
    """Run all examples."""
    print("Tensor Image Utils - Usage Examples")
    print("=" * 50)
    
    example_basic_operations()
    example_tensor_conversions()
    example_resize_operations()
    example_raw_binary_handling()
    example_numpy_format_handling()
    example_error_handling()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    main()
