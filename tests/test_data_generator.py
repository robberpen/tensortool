#!/usr/bin/env python3
"""
Generate test sample images for tensor-image-utils testing.
Run this script to create all necessary test files.
"""

import os
import numpy as np
import cv2

def create_test_data_dir():
    """Create test data directory if not exists."""
    test_dir = "tests/test_data"
    os.makedirs(test_dir, exist_ok=True)
    return test_dir

def generate_test_image(width: int, height: int, channels: int = 3) -> np.ndarray:
    """Generate a colorful test image with gradient pattern."""
    # Create gradient pattern
    x_grad = np.linspace(0, 255, width, dtype=np.uint8)
    y_grad = np.linspace(0, 255, height, dtype=np.uint8)
    
    # Create RGB channels
    if channels == 3:
        image = np.zeros((height, width, channels), dtype=np.uint8)
        
        # Red channel - horizontal gradient
        image[:, :, 0] = x_grad[np.newaxis, :]
        
        # Green channel - vertical gradient  
        image[:, :, 1] = y_grad[:, np.newaxis]
        
        # Blue channel - diagonal pattern
        for y in range(height):
            for x in range(width):
                image[y, x, 2] = (x + y) % 256
                
    elif channels == 1:
        # Grayscale checkerboard pattern
        image = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                image[y, x] = 255 if (x // 8 + y // 8) % 2 else 0
                
    return image

def create_standard_images(test_dir: str):
    """Create standard format test images (.jpg, .bmp, .webp)."""
    
    # 640x480 JPEG
    img_640_480 = generate_test_image(640, 480, 3)
    cv2.imwrite(os.path.join(test_dir, "sample_640x480.jpg"), img_640_480)
    
    # 320x240 BMP
    img_320_240 = generate_test_image(320, 240, 3) 
    cv2.imwrite(os.path.join(test_dir, "sample_320x240.bmp"), img_320_240)
    
    # 256x256 WebP
    img_256_256 = generate_test_image(256, 256, 3)
    cv2.imwrite(os.path.join(test_dir, "sample_256x256.webp"), img_256_256)
    
    print("✓ Created standard format images")

def create_numpy_files(test_dir: str):
    """Create .npy test files."""
    
    # 100x100x3 RGB tensor (HWC format)
    img_100_100 = generate_test_image(100, 100, 3)
    tensor_hwc = img_100_100.astype(np.float32) / 255.0  # Normalize to [0,1]
    np.save(os.path.join(test_dir, "sample_100x100.npy"), tensor_hwc)
    
    # 50x50x3 tensor in NCHW format for testing
    img_50_50 = generate_test_image(50, 50, 3)
    tensor_nchw = np.transpose(img_50_50, (2, 0, 1)).astype(np.float32) / 255.0
    tensor_nchw = np.expand_dims(tensor_nchw, axis=0)  # Add batch dimension
    np.save(os.path.join(test_dir, "sample_1x3x50x50_nchw.npy"), tensor_nchw)
    
    print("✓ Created numpy format files")

def create_raw_binary_files(test_dir: str):
    """Create raw binary format files (.rgb888, .bgr888)."""
    
    # 64x64 RGB888
    img_64_64_rgb = generate_test_image(64, 64, 3)
    with open(os.path.join(test_dir, "test_image_64x64.rgb888"), "wb") as f:
        f.write(img_64_64_rgb.tobytes())
    
    # 32x32 BGR888 (convert RGB to BGR)
    img_32_32_rgb = generate_test_image(32, 32, 3)
    img_32_32_bgr = cv2.cvtColor(img_32_32_rgb, cv2.COLOR_RGB2BGR)
    with open(os.path.join(test_dir, "test_image_32x32.bgr888"), "wb") as f:
        f.write(img_32_32_bgr.tobytes())
    
    print("✓ Created raw binary format files")

def create_corrupted_files(test_dir: str):
    """Create corrupted files for error testing."""
    
    # Corrupted JPEG
    with open(os.path.join(test_dir, "corrupted.jpg"), "wb") as f:
        f.write(b"This is not a JPEG file")
    
    # Invalid dimension raw file
    fake_data = np.random.randint(0, 255, (100,), dtype=np.uint8)
    with open(os.path.join(test_dir, "invalid_size_10x10.rgb888"), "wb") as f:
        f.write(fake_data.tobytes())
    
    print("✓ Created corrupted test files")

def main():
    """Generate all test data files."""
    print("Generating test data files...")
    
    test_dir = create_test_data_dir()
    
    create_standard_images(test_dir)
    create_numpy_files(test_dir)  
    create_raw_binary_files(test_dir)
    create_corrupted_files(test_dir)
    
    print(f"\n✓ All test data files created in: {test_dir}")
    print("\nGenerated files:")
    for file in sorted(os.listdir(test_dir)):
        file_path = os.path.join(test_dir, file)
        size = os.path.getsize(file_path)
        print(f"  - {file} ({size} bytes)")

if __name__ == "__main__":
    main()
