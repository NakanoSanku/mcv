# mcv-paddle

PaddleOCR backend for MCV (Multi-Computer Vision).

## Installation

```bash
# Install with PaddleOCR support
pip install mcv-paddle[paddleocr]

# Or install paddleocr separately
pip install mcv-paddle
pip install paddleocr
```

## Usage

```python
import cv2
from mcv.paddle import PaddleOCRTemplate

# Load image
screen = cv2.imread("screenshot.png")

# Find specific text
template = PaddleOCRTemplate(pattern="确认", lang="ch", threshold=0.8)
result = template.find(screen)
if result:
    print(f"Found '{result.text}' at {result.center}")

# Find text matching regex
template = PaddleOCRTemplate(pattern=r"\d+", regex=True, lang="ch")
results = template.find_all(screen)
for r in results:
    print(f"Number: {r.text}, confidence: {r.confidence:.2f}")

# Extract all text in region
template = PaddleOCRTemplate(pattern=None, lang="ch")
results = template.find_all(screen, roi=[100, 100, 200, 200])
for r in results:
    print(f"{r.text}: {r.confidence:.2f}")
```

## API

### PaddleOCRTemplate

Main class for OCR-based text matching.

**Constructor:**
- `pattern`: Target text or regex pattern. `None` returns all text.
- `regex`: If True, treat pattern as regex.
- `threshold`: Minimum confidence threshold [0, 1].
- `lang`: PaddleOCR language code ("ch", "en", etc.).
- `use_angle_cls`: Enable text angle classification.
- `use_gpu`: Enable GPU acceleration.

**Methods:**
- `find(image, roi=None, threshold=None) -> PaddleOCRMatchResult | None`
- `find_all(image, roi=None, threshold=None, max_count=None) -> List[PaddleOCRMatchResult]`

### PaddleOCRMatchResult

OCR match result with text information.

**Properties:**
- `text`: Recognized text string
- `confidence`: Recognition confidence [0, 1]
- `quad`: Four corner points of text bounding box
- `top_left`, `bottom_right`, `center`: Derived coordinates
- `to_match_result()`: Convert to core `MatchResult`
