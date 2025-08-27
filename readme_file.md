# tensor-image-utils

A lightweight Python library for converting between image files and tensor formats, designed for embedded systems and edge computing applications.

## Features

- **Multiple Format Support**: .jpg, .jpeg, .bmp, .webp, .png, .npy, .rgb888, .bgr888
- **Tensor Operations**: Convert between HWC, NCHW, and NHWC formats
- **Minimal Dependencies**: Only requires NumPy and OpenCV
- **Embedded-Ready**: Optimized for Yocto Linux environments
- **Type Safety**: Full type hints for better development experience
- **Consistent RGB**: All operations maintain RGB channel order
- **Error Handling**: Fail-fast with clear error messages

## Installation

### From Source
```bash
git clone <repository-url>
cd tensor-image-utils
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- NumPy >= 1.19.0
- OpenCV >= 4.5.0

## Quick Start

```python
from tensor_image_utils import ImageTensorConverter

# Initialize converter
converter = ImageTensorConverter()

# Load image as tensor (HWC RGB uint8)
image_tensor = converter.load("input.jpg")

# Convert to different tensor formats
nchw_tensor = converter.to_nchw(image_tensor)  # Shape: (1, 3, H, W)
nhwc_tensor = converter.to_nhwc(nchw_tensor)   # Shape: (1, H, W, 3)

# Resize image
resized = converter.resize(image_tensor, (224, 224))

# Save in different format
converter.save(resized, "output.webp")
```

## Supported Formats

### Standard Image Formats
- **JPEG** (`.jpg`, `.jpeg`) - Loaded as RGB, saved with 95% quality
- **BMP** (`.bmp`) - Uncompressed bitmap
- **WebP** (`.webp`) - Modern web format, 95% quality
- **PNG** (`.png`) - Lossless compression

### NumPy Arrays
- **NPY** (`.npy`) - NumPy binary format
- Automatic conversion from various data types to uint8
- Supports 2D (grayscale), 3D (HWC/CHW), and 4D (batch) arrays

### Raw Binary Formats
- **RGB888** (`.rgb888`) - Raw RGB binary data
- **BGR888** (`.bgr888`) - Raw BGR binary data  
- Headerless format with dimensions encoded in filename or provided explicitly

## Usage Examples

### Basic Operations
```python
converter = ImageTensorConverter()

# Load and get info
tensor = converter.load("image.jpg")
info = converter.get_tensor_info(tensor)
print(f"Shape: {info['shape']}, Range: [{info['min_value']}, {info['max_value']}]")

# Format conversion
converter.save(tensor, "converted.bmp")
```

### Tensor Format Conversion
```python
# HWC to NCHW (for PyTorch/ONNX)
hwc_tensor = converter.load("input.jpg")        # Shape: (H, W, 3)
nchw_tensor = converter.to_nchw(hwc_tensor)     # Shape: (1, 3, H, W)

# NCHW to NHWC (for TensorFlow)
nhwc_tensor = converter.to_nhwc(nchw_tensor)    # Shape: (1, H, W, 3)

# Back to HWC for processing
hwc_result = converter.to_hwc(nhwc_tensor)      # Shape: (H, W, 3)
```

### Raw Binary Formats
```python
# Method 1: Dimensions encoded in filename
converter.save(tensor, "image_640x480.rgb888")
loaded = converter.load("image_640x480.rgb888")

# Method 2: Explicit dimensions
converter.load("data.rgb888", width=640, height=480)
```

### Resize Operations
```python
# Resize with different interpolation methods
import cv2

resized = converter.resize(tensor, (224, 224))  # Default: bilinear
resized = converter.resize(tensor, (224, 224), cv2.INTER_CUBIC)
resized = converter.resize(tensor, (224, 224), cv2.INTER_NEAREST)
```

## Data Type Handling

### NumPy Format Conversion
The library automatically converts various NumPy data types to uint8:

- **uint8**: No conversion needed
- **float32/float64**: 
  - `[0, 1]` range → scaled to `[0, 255]`
  - `[-1, 1]` range → scaled to `[0, 255]`
  - Other ranges → normalized then scaled
- **int8/int16/int32**: Normalized to `[0, 255]`
- **uint16/uint32**: Scaled down to `[0, 255]`

### Channel Order Consistency
- All loaded images are converted to **RGB** channel order
- Raw `.bgr888` files are automatically converted to RGB
- Output maintains RGB order unless saving to BGR format

## Error Handling

The library follows a **fail-fast** approach:
- File not found → `sys.exit(1)`
- Unsupported format → `sys.exit(1)`  
- Invalid dimensions → `sys.exit(1)`
- Corrupted data → `sys.exit(1)`

All errors print descriptive messages to stderr before exiting.

## API Reference

### ImageTensorConverter

#### Loading Methods
- `load(filepath, **kwargs)` - Load image as HWC RGB uint8 tensor
- For raw formats, use `width=` and `height=` kwargs

#### Saving Methods  
- `save(tensor, filepath)` - Save HWC RGB uint8 tensor as image

#### Tensor Conversion Methods
- `to_nchw(hwc_tensor)` - Convert HWC to NCHW format (adds batch dim)
- `to_nhwc(nchw_tensor)` - Convert NCHW to NHWC format  
- `to_hwc(tensor)` - Convert various formats to HWC

#### Utility Methods
- `resize(tensor, target_size, interpolation=cv2.INTER_LINEAR)` - Resize tensor
- `get_supported_formats()` - Get set of supported file extensions
- `get_tensor_info(tensor)` - Get tensor metadata

## Development

### Project Structure
```
tensor-image-utils/
├── src/tensor_image_utils/
│   ├── __init__.py
│   ├── converter.py          # Main API class
│   ├── exceptions.py         # Custom exceptions
│   └── handlers/            # Format-specific handlers
│       ├── __init__.py
│       ├── base.py          # Abstract base class
│       ├── standard.py      # Standard image formats
│       ├── numpy_handler.py # NumPy format handler
│       └── raw_binary.py    # Raw binary format handler
├── tests/                   # Unit tests
├── examples/               # Usage examples
└── docs/                   # Documentation
```

### Running Tests
```bash
# Generate test data first
python tests/generate_test_data.py

# Run tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_converter.py -v
```

### Running Examples
```bash
# Generate test data
python tests/generate_test_data.py

# Run examples
python examples/basic_usage.py
python examples/format_conversion.py
```

## Design Principles

1. **Embedded-First**: Minimal dependencies for resource-constrained environments
2. **Type Safety**: Comprehensive type hints for development tooling
3. **Fail-Fast**: Clear error messages with immediate exit on errors
4. **Consistency**: RGB channel order and uint8 data type throughout
5. **Extensibility**: Plugin architecture for adding new formats
6. **Performance**: Efficient memory usage and minimal data copying

## License

[Your License Here]

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Changelog

### v0.1.0 (Initial Release)
- Support for standard image formats (JPEG, BMP, WebP, PNG)
- NumPy array format support with automatic type conversion
- Raw binary format support (RGB888, BGR888)
- Tensor format conversions (HWC, NCHW, NHWC)
- Resize operations with OpenCV interpolation
- Comprehensive error handling with sys.exit
- Full type hints and documentation