"""PaddleOCR-based text template matching.

This module adapts PaddleOCR output to the MCV Template API.
It supports exact/regex text search and full-text extraction via pattern=None.

Example:
    >>> from mcv.paddle import PaddleOCRTemplate
    >>> # Find specific text
    >>> template = PaddleOCRTemplate(pattern="确认", lang="ch")
    >>> result = template.find(image)
    >>> if result:
    ...     print(f"Found '{result.text}' at {result.center}")
    >>>
    >>> # Extract all text in region
    >>> template = PaddleOCRTemplate(pattern=None, lang="ch")
    >>> results = template.find_all(image, roi=[100, 100, 200, 200])
    >>> for r in results:
    ...     print(f"{r.text}: {r.confidence:.2f}")
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from mcv.base import MatchResult, ROI, ROILike, Template

if TYPE_CHECKING:
    from paddleocr import PaddleOCR

# Type aliases
QuadPoint = Tuple[float, float]
Quad = Tuple[QuadPoint, QuadPoint, QuadPoint, QuadPoint]

# Global OCR client cache
_ocr_cache: Dict[Tuple[str, bool, bool], "PaddleOCR"] = {}
_ocr_cache_lock = threading.Lock()


class PaddleOCRNotInstalledError(ImportError):
    """Raised when paddleocr package is not installed."""


def _get_ocr_client(lang: str, use_angle_cls: bool, use_gpu: bool) -> "PaddleOCR":
    """Get or create a cached PaddleOCR client.

    PaddleOCR initialization is expensive (loads models), so we cache
    instances by their configuration to avoid repeated loading.

    Args:
        lang: Language code (e.g., "ch", "en", "japan")
        use_angle_cls: Whether to enable angle classifier
        use_gpu: Whether to use GPU acceleration

    Returns:
        Cached or newly created PaddleOCR instance

    Raises:
        PaddleOCRNotInstalledError: If paddleocr is not installed
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise PaddleOCRNotInstalledError(
            "paddleocr is not installed. Install with:\n"
            "  pip install paddleocr\n"
            "For GPU support:\n"
            "  pip install paddleocr paddlepaddle-gpu"
        ) from exc

    key = (lang, use_angle_cls, use_gpu)

    # Fast path: check without lock
    client = _ocr_cache.get(key)
    if client is not None:
        return client

    # Slow path: acquire lock and double-check
    with _ocr_cache_lock:
        client = _ocr_cache.get(key)
        if client is not None:
            return client

        # Suppress PaddleOCR's verbose logging
        client = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
        _ocr_cache[key] = client
        return client


def clear_ocr_cache() -> None:
    """Clear the global OCR client cache.

    This can be useful for releasing memory or forcing model reload.
    """
    with _ocr_cache_lock:
        _ocr_cache.clear()


@dataclass(frozen=True)
class PaddleOCRMatchResult:
    """PaddleOCR match result with recognized text information.

    This extends the concept of MatchResult with OCR-specific fields
    while maintaining compatibility through the to_match_result() method.

    Attributes:
        text: Recognized text string
        confidence: Recognition confidence score [0, 1]
        quad: Four corner points of the text bounding quadrilateral
              in format ((x0,y0), (x1,y1), (x2,y2), (x3,y3))
              representing top-left, top-right, bottom-right, bottom-left
        top_left: Cached top-left corner of axis-aligned bounding box
        bottom_right: Cached bottom-right corner of axis-aligned bounding box
        center: Cached center point of the bounding box
    """

    text: str
    confidence: float
    quad: Quad
    # Cached bbox coordinates (computed in __post_init__)
    _top_left: Tuple[int, int] = None  # type: ignore[assignment]
    _bottom_right: Tuple[int, int] = None  # type: ignore[assignment]
    _center: Tuple[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Compute and cache bounding box coordinates."""
        xs = [pt[0] for pt in self.quad]
        ys = [pt[1] for pt in self.quad]

        x1 = int(math.floor(min(xs)))
        y1 = int(math.floor(min(ys)))
        x2 = int(math.ceil(max(xs)))
        y2 = int(math.ceil(max(ys)))

        # Ensure at least 1x1 size
        x2 = max(x2, x1 + 1)
        y2 = max(y2, y1 + 1)

        center_x = x1 + (x2 - x1) // 2
        center_y = y1 + (y2 - y1) // 2

        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(self, "_top_left", (x1, y1))
        object.__setattr__(self, "_bottom_right", (x2, y2))
        object.__setattr__(self, "_center", (center_x, center_y))

    @property
    def top_left(self) -> Tuple[int, int]:
        """Top-left corner of the axis-aligned bounding box."""
        return self._top_left

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Bottom-right corner of the axis-aligned bounding box."""
        return self._bottom_right

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        return self._center

    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self._bottom_right[0] - self._top_left[0]

    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self._bottom_right[1] - self._top_left[1]

    def to_match_result(self) -> MatchResult:
        """Convert to core MatchResult for API compatibility.

        Returns:
            MatchResult with bounding box coordinates and confidence as score
        """
        return MatchResult(
            top_left=self.top_left,
            bottom_right=self.bottom_right,
            center=self.center,
            score=self.confidence,
        )


class PaddleOCRTemplate(Template):
    """PaddleOCR-based text template for finding text in images.

    This template uses PaddleOCR to detect and recognize text, then
    filters results by pattern matching. When pattern is None, all
    detected text is returned.

    Example:
        >>> import cv2
        >>> screen = cv2.imread("screenshot.png")
        >>>
        >>> # Find specific button text
        >>> template = PaddleOCRTemplate(pattern="确认", lang="ch", threshold=0.8)
        >>> result = template.find(screen)
        >>> if result:
        ...     print(f"Found '{result.text}' at {result.center}")
        >>>
        >>> # Find text matching regex
        >>> template = PaddleOCRTemplate(pattern=r"\\d+", regex=True)
        >>> results = template.find_all(screen)
        >>> for r in results:
        ...     print(f"Number found: {r.text}")
        >>>
        >>> # Extract all text in a region
        >>> template = PaddleOCRTemplate(pattern=None)
        >>> results = template.find_all(screen, roi=[0, 0, 500, 500])

    Attributes:
        pattern: Target text or regex pattern; None matches all text
        regex: Whether to treat pattern as regex
        lang: PaddleOCR language code
        use_angle_cls: Whether to enable text angle classification
        use_gpu: Whether to use GPU acceleration
    """

    def __init__(
        self,
        pattern: Optional[Union[str, Pattern[str]]] = None,
        *,
        regex: bool = False,
        roi: Optional[ROILike] = None,
        threshold: float = 0.5,
        lang: str = "ch",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ) -> None:
        """Initialize OCR template.

        Args:
            pattern: Target text or regex pattern.
                - str: Match if text contains this substring
                - Pattern: Match if regex matches
                - None: Return all detected text (no filtering)
            regex: If True and pattern is str, compile as regex.
                   Ignored if pattern is already a Pattern object.
            roi: Default search region
            threshold: Minimum confidence threshold [0, 1]
            lang: PaddleOCR language code.
                  Common values: "ch" (Chinese+English), "en" (English),
                  "japan", "korean", "german", "french"
            use_angle_cls: Enable text angle classifier for rotated text
            use_gpu: Use GPU acceleration (requires paddlepaddle-gpu)

        Raises:
            ValueError: If threshold is invalid
        """
        super().__init__(roi=roi, threshold=threshold)

        self.pattern = pattern
        self.regex = regex
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu

        # Pre-compile regex pattern if needed
        self._compiled_re: Optional[Pattern[str]] = None
        if isinstance(pattern, Pattern):
            self._compiled_re = pattern
        elif pattern is not None and regex:
            self._compiled_re = re.compile(pattern)

    def find(
        self,
        image: np.ndarray,
        roi: Optional[ROILike] = None,
        threshold: Optional[float] = None,
    ) -> Optional[PaddleOCRMatchResult]:
        """Find the first text matching the pattern.

        Args:
            image: Input image (BGR format, numpy.ndarray)
            roi: Search region, overrides default
            threshold: Confidence threshold, overrides default

        Returns:
            Best matching PaddleOCRMatchResult, or None if not found
        """
        results = self.find_all(image, roi=roi, threshold=threshold, max_count=1)
        return results[0] if results else None

    def find_all(
        self,
        image: np.ndarray,
        roi: Optional[ROILike] = None,
        threshold: Optional[float] = None,
        max_count: Optional[int] = None,
    ) -> List[PaddleOCRMatchResult]:
        """Find all text instances matching the pattern.

        Args:
            image: Input image (BGR format, numpy.ndarray)
            roi: Search region, overrides default
            threshold: Confidence threshold, overrides default
            max_count: Maximum results to return, overrides default

        Returns:
            List of PaddleOCRMatchResult sorted by confidence (descending)
        """
        self._validate_image(image)

        img_height, img_width = image.shape[:2]
        effective_roi = self._resolve_roi(roi, img_width, img_height)
        if effective_roi is None:
            return []

        # Crop ROI region
        roi_image = self._crop_roi(image, effective_roi)

        # Ensure BGR format for PaddleOCR
        roi_bgr = self._ensure_bgr(roi_image)

        # Get OCR client (lazy loaded and cached)
        client = _get_ocr_client(
            lang=self.lang,
            use_angle_cls=self.use_angle_cls,
            use_gpu=self.use_gpu,
        )

        # Run OCR
        ocr_output = client.ocr(roi_bgr, det=True, rec=True, cls=self.use_angle_cls)

        if not ocr_output:
            return []

        # PaddleOCR returns list of results per image
        entries = ocr_output[0] if isinstance(ocr_output, list) else []
        if not entries:
            return []

        effective_threshold = self._resolve_threshold(threshold)
        matches: List[PaddleOCRMatchResult] = []

        for entry in entries:
            if not entry or len(entry) < 2:
                continue

            quad_raw, rec = entry[0], entry[1]
            if not quad_raw or len(quad_raw) != 4:
                continue

            text, score = rec
            confidence = float(score)

            # Filter by confidence
            if confidence < effective_threshold:
                continue

            # Filter by pattern
            if not self._matches_pattern(str(text)):
                continue

            # Offset coordinates back to full image space
            quad = self._offset_quad(quad_raw, effective_roi)
            if quad is None:
                # Skip malformed quad entries
                continue

            matches.append(
                PaddleOCRMatchResult(
                    text=str(text),
                    confidence=confidence,
                    quad=quad,
                )
            )

        if not matches:
            return []

        # Sort by confidence descending
        matches.sort(key=lambda m: m.confidence, reverse=True)

        effective_max_count: Optional[int]
        if max_count is not None:
            effective_max_count = self._resolve_max_count(max_count)
            return matches[:effective_max_count]
        return matches

    def _matches_pattern(self, text: str) -> bool:
        """Check if text matches the configured pattern.

        Args:
            text: Recognized text to check

        Returns:
            True if text matches pattern or pattern is None
        """
        if self.pattern is None:
            return True

        if self._compiled_re is not None:
            return bool(self._compiled_re.search(text))

        # Substring match for non-regex string pattern
        return str(self.pattern) in text

    def _offset_quad(
        self,
        quad: Sequence[Sequence[float]],
        roi: ROI,
    ) -> Optional[Quad]:
        """Offset quadrilateral coordinates from ROI space to image space.

        Args:
            quad: Four points in ROI coordinates
            roi: The ROI that was cropped

        Returns:
            Four points in full image coordinates, or None if quad is malformed
        """
        if len(quad) != 4:
            return None

        points: List[QuadPoint] = []
        for pt in quad:
            if not hasattr(pt, "__len__") or len(pt) < 2:
                return None
            try:
                points.append((float(pt[0]) + roi.x, float(pt[1]) + roi.y))
            except (TypeError, ValueError):
                return None

        return (points[0], points[1], points[2], points[3])

    def _crop_roi(self, image: np.ndarray, roi: ROI) -> np.ndarray:
        """Crop ROI region from image.

        Args:
            image: Source image
            roi: Region to crop

        Returns:
            Cropped image region
        """
        return image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    def _validate_image(self, image: np.ndarray) -> None:
        """Validate input image format.

        Args:
            image: Image to validate

        Raises:
            TypeError: If image is not numpy.ndarray
            ValueError: If image dimensions are invalid
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be numpy.ndarray, got {type(image).__name__}")

        if image.ndim == 2:
            return  # Grayscale OK
        if image.ndim == 3 and image.shape[2] == 3:
            return  # BGR OK

        raise ValueError(
            f"Image must be 2D grayscale or 3-channel BGR, got shape {image.shape}"
        )

    def _ensure_bgr(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is 3-channel BGR as expected by PaddleOCR.

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            3-channel BGR image
        """
        if image.ndim == 2:
            # Convert grayscale to BGR by stacking
            return np.stack([image, image, image], axis=2)
        return image
