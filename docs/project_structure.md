# tensor-image-utils Project Structure

```
tensor-image-utils/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── src/
│   └── tensor_image_utils/
│       ├── __init__.py
│       ├── converter.py          # 主要的ImageTensorConverter類別
│       ├── handlers/
│       │   ├── __init__.py
│       │   ├── base.py          # FormatHandler抽象基類
│       │   ├── standard.py      # StandardImageHandler
│       │   ├── numpy_handler.py # NumpyHandler  
│       │   └── raw_binary.py    # RawBinaryHandler
│       └── exceptions.py        # 自定義例外類別
├── tests/
│   ├── __init__.py
│   ├── test_converter.py
│   ├── test_handlers.py
│   └── test_data/              # 測試樣本檔案
│       ├── sample_640x480.jpg
│       ├── sample_320x240.bmp
│       ├── sample_256x256.webp
│       ├── sample_100x100.npy
│       ├── test_image_64x64.rgb888
│       └── test_image_32x32.bgr888
├── examples/
│   ├── basic_usage.py
│   ├── format_conversion.py
│   └── tensor_operations.py
└── docs/
    ├── api_reference.md
    └── usage_guide.md
```

## Core Files Content

### requirements.txt
```
numpy>=1.19.0
opencv-python>=4.5.0
```

### .gitignore
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

### src/tensor_image_utils/__init__.py
```python
"""
Tensor Image Utilities - A lightweight image-tensor conversion library.
Designed for embedded systems with minimal dependencies.
"""

from .converter import ImageTensorConverter
from .exceptions import TensorImageError, UnsupportedFormatError, InvalidDimensionError

__version__ = "0.1.0"
__author__ = "Your Name"
__all__ = ["ImageTensorConverter", "TensorImageError", "UnsupportedFormatError", "InvalidDimensionError"]
```