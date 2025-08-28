# TensorConverter Architecture Design

## Core Architecture

### Class Hierarchy (Factory + Strategy Pattern)
```
ImageTensorConverter
├── _format_handlers: Dict[str, FormatHandler]
├── _supported_formats: Set[str] 
└── FormatHandlerFactory

FormatHandler (Abstract Base)
├── StandardImageHandler (.jpg, .bmp, .webp)
├── NumpyHandler (.npy)
└── RawBinaryHandler (.rgb888, .bgr888)
```

### Key Design Decisions

1. **Factory Pattern**: 動態創建格式處理器
2. **Strategy Pattern**: 每種格式獨立的處理邏輯
3. **Open-Closed Principle**: 易於擴展新格式 (未來NV12/YUV)
4. **Fail-Fast**: 錯誤立即拋出並終止
5. **Type Safety**: 完整型別提示

## API Interface Design

### Core Methods
```python
class ImageTensorConverter:
    def __init__(self) -> None
    
    # Loading methods
    def load(self, filepath: str, **kwargs) -> np.ndarray
    
    # Saving methods  
    def save(self, tensor: np.ndarray, filepath: str) -> None
    
    # Conversion methods
    def to_nchw(self, hwc_tensor: np.ndarray) -> np.ndarray
    def to_nhwc(self, nchw_tensor: np.ndarray) -> np.ndarray
    def to_hwc(self, tensor: np.ndarray) -> np.ndarray
    
    # Utility methods
    def resize(self, tensor: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray
    def get_supported_formats(self) -> Set[str]
```

## Format Handling Strategy

### File Extension → Handler Mapping
- `.jpg/.jpeg/.bmp/.webp` → cv2.imread/cv2.imwrite
- `.npy` → numpy.load/numpy.save
- `.rgb888/.bgr888` → custom binary reader/writer

### Raw Binary Naming Convention
- Format: `filename_{width}x{height}.{rgb888|bgr888}`
- Example: `image_1920x1080.rgb888`
- Regex: `r".*_(\d+)x(\d+)\.(rgb888|bgr888)$"`

## Error Handling Strategy

### Error Categories
1. **FileNotFoundError** → sys.exit(1)
2. **UnsupportedFormatError** → sys.exit(1)  
3. **InvalidDimensionError** → sys.exit(1)
4. **CorruptedDataError** → sys.exit(1)

### Implementation
```python
def _handle_error(self, error: Exception, context: str) -> None:
    print(f"ERROR [{context}]: {error}", file=sys.stderr)
    sys.exit(1)
```

## Data Flow Architecture

```
Input File → Format Detection → Loader Selection → 
Data Loading → Tensor Conversion → Shape Transformation → 
Error Handling → Output
```

## Dependencies
- **numpy**: 核心張量操作
- **opencv-python**: 圖像I/O和處理
- **typing**: 型別提示支援 (Python 3.8+)
- **abc**: 抽象基類支援

## Performance Considerations
- 統一使用cv2處理所有標準圖像格式
- 記憶體效率的張量轉換
- 避免不必要的資料複製
- RGB/BGR通道順序處理優化