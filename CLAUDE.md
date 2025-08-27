# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **tensor-image-utils**, a lightweight Python library for converting between image files and tensor formats, designed for embedded systems and edge computing. The library uses a factory + strategy pattern architecture with fail-fast error handling.

## Core Architecture

The main entry point is `ImageTensorConverter` in `tensor_image_utils/converter.py`, which delegates to format-specific handlers:

- **StandardImageHandler** (`handlers/standard.py`) - JPEG, BMP, WebP, PNG using OpenCV
- **NumpyHandler** (`handlers/numpy_handler.py`) - .npy files with automatic type conversion
- **RawBinaryHandler** (`handlers/raw_binary.py`) - .rgb888/.bgr888 with dimension parsing

All handlers inherit from `FormatHandler` (`handlers/base.py`) abstract base class.

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies (numpy>=1.19.0, opencv-python>=4.5.0)
pip install numpy opencv-python

# No setup.py or requirements.txt exists - install manually
```

### Testing
```bash
# Generate test data first
python tests/test_data_generator.py

# Run tests using pytest
python -m pytest tests/basic_tests.py -v

# Run specific test
python -m pytest tests/basic_tests.py::TestImageTensorConverter::test_to_nchw_conversion -v
```

### Running Examples
```bash
# Generate test data
python tests/test_data_generator.py

# Run usage examples
python examples/usage_examples.py
```

## Key Design Principles

1. **Fail-Fast Error Handling**: All errors call `sys.exit(1)` with stderr messages
2. **RGB Consistency**: All operations maintain RGB channel order (BGR formats auto-converted)
3. **uint8 Standard**: All tensors are normalized to uint8 range [0-255]
4. **HWC Default**: Primary tensor format is Height-Width-Channels

## Tensor Format Conversions

- `to_nchw()`: HWC → NCHW (adds batch dimension for PyTorch/ONNX)
- `to_nhwc()`: NCHW → NHWC (for TensorFlow)
- `to_hwc()`: Various formats → HWC (with automatic format detection)

## Raw Binary Format Convention

Files use naming pattern: `filename_{width}x{height}.{rgb888|bgr888}`
Example: `image_1920x1080.rgb888`

Can also specify dimensions explicitly:
```python
converter.load("data.rgb888", width=640, height=480)
```

## Common Issues

- **No package structure**: Library uses relative imports, run from project root
- **Manual dependencies**: No requirements.txt, install numpy and opencv-python manually
- **Test data required**: Run `test_data_generator.py` before running tests or examples
- **Path handling**: Examples expect `../tests/test_data/` relative path structure

## File Structure
```
tensor_image_utils/
├── converter.py          # Main ImageTensorConverter class
├── exceptions.py         # Custom exceptions
└── handlers/
    ├── base.py          # Abstract FormatHandler base
    ├── standard.py      # Standard image formats
    ├── numpy_handler.py # .npy format handler
    └── raw_binary.py    # Raw binary formats
```