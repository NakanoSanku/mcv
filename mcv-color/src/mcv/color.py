"""Multi-point color template matching.

This module provides multi-point color matching using the classic
algorithm inspired by AutoHotkey/按键精灵:
1. Search for the first color in the image
2. For each match, check all offset points
3. Return the position if all colors match
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np

from mcv.base import MatchResult, ROI, ROILike, Template

if TYPE_CHECKING:
    pass

# Color can be hex string "FF5500" or RGB tuple (255, 85, 0)
ColorLike = Union[str, Tuple[int, int, int], Sequence[int]]

# Offset specification: (dx, dy, color)
OffsetSpec = Tuple[int, int, ColorLike]


class MultiColorTemplate(Template):
    """Multi-point color template matching.

    Uses first color + offset colors algorithm for UI element detection.
    More lightweight and noise-resistant than image template matching.

    Example:
        >>> import cv2
        >>> screen = cv2.imread("screenshot.png")
        >>> template = MultiColorTemplate(
        ...     first_color="1E841E",
        ...     offsets=[
        ...         (14, -1, "2E942E"),
        ...         (41, 17, "82C693"),
        ...     ],
        ...     threshold=0.95,
        ... )
        >>> result = template.find(screen)
        >>> if result:
        ...     print(f"Found at: {result.center}")
    """

    def __init__(
        self,
        first_color: ColorLike,
        offsets: Sequence[OffsetSpec],
        *,
        roi: Optional[ROILike] = None,
        threshold: float = 1.0,
    ) -> None:
        """Initialize multi-point color template.

        Args:
            first_color: First point color (hex string "FF5500" or RGB tuple)
            offsets: Offset points [(dx, dy, color), ...]
            roi: Default search region
            threshold: Color similarity threshold [0, 1], 1.0 = exact match

        Raises:
            ValueError: Invalid color format or threshold
        """
        super().__init__(roi=roi, threshold=threshold)

        self._first_color = self._normalize_color(first_color)
        # Pre-convert to int16 for efficient comparison
        self._first_color_i16 = self._first_color.astype(np.int16)
        self._offsets = [
            (int(dx), int(dy), self._normalize_color(color))
            for dx, dy, color in offsets
        ]
        # Pre-convert offset colors to int16
        self._offsets_i16 = [
            (dx, dy, color.astype(np.int16))
            for dx, dy, color in self._offsets
        ]

    @staticmethod
    def _normalize_color(color: ColorLike) -> np.ndarray:
        """Normalize color to RGB numpy array.

        Args:
            color: Hex string "FF5500" or RGB tuple (255, 85, 0)

        Returns:
            RGB numpy array with shape (3,), dtype uint8

        Raises:
            ValueError: Invalid hex string format
            TypeError: Invalid color type
        """
        if isinstance(color, str):
            code = color.strip().lstrip("#")
            if len(code) != 6:
                raise ValueError(
                    f"Color hex string must be 6 characters, got '{color}'"
                )
            try:
                r = int(code[0:2], 16)
                g = int(code[2:4], 16)
                b = int(code[4:6], 16)
            except ValueError as e:
                raise ValueError(f"Invalid hex color '{color}': {e}") from e
            return np.array([r, g, b], dtype=np.uint8)

        if isinstance(color, (tuple, list, np.ndarray)):
            if len(color) != 3:
                raise ValueError(f"Color tuple must have 3 elements, got {len(color)}")
            # Validate RGB values are in range [0, 255]
            for i, val in enumerate(color):
                if not isinstance(val, (int, float, np.integer, np.floating)):
                    raise TypeError(f"Color channel {i} must be numeric, got {type(val).__name__}")
                if val < 0 or val > 255:
                    raise ValueError(f"Color channel {i} must be in [0, 255], got {val}")
            return np.array(color, dtype=np.uint8)

        raise TypeError(
            f"Color must be hex string or RGB tuple, got {type(color).__name__}"
        )

    def find(
        self,
        image: np.ndarray,
        roi: Optional[ROILike] = None,
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        """Find single match.

        Args:
            image: Search image (numpy.ndarray, BGR format)
            roi: Search region
            threshold: Color similarity threshold

        Returns:
            Match result (point), or None if not found
        """
        results = self.find_all(image, roi=roi, threshold=threshold, max_count=1)
        return results[0] if results else None

    def find_all(
        self,
        image: np.ndarray,
        roi: Optional[ROILike] = None,
        threshold: Optional[float] = None,
        max_count: Optional[int] = None,
    ) -> List[MatchResult]:
        """Find all matches.

        Args:
            image: Search image (numpy.ndarray, BGR format)
            roi: Search region
            threshold: Color similarity threshold
            max_count: Maximum result count

        Returns:
            List of match results (points), in search order (top-left to bottom-right)
        """
        self._validate_image(image)

        img_height, img_width = image.shape[:2]
        effective_roi = self._resolve_roi(roi, img_width, img_height)

        if effective_roi is None:
            return []

        effective_threshold = self._resolve_threshold(threshold)
        effective_max_count: Optional[int]
        if max_count is not None:
            effective_max_count = self._resolve_max_count(max_count)
        else:
            effective_max_count = None

        # Calculate tolerance from threshold
        # threshold=1.0 -> tolerance=0 (exact match)
        # threshold=0.9 -> tolerance=25 (allow ~10% difference)
        tolerance = int(round((1.0 - effective_threshold) * 255))

        # Crop ROI region
        roi_image = image[
            effective_roi.y : effective_roi.y + effective_roi.height,
            effective_roi.x : effective_roi.x + effective_roi.width,
        ]

        # Convert BGR to RGB for matching
        roi_rgb = roi_image[:, :, ::-1]

        results: List[MatchResult] = []
        candidates = self._find_first_color_candidates(roi_rgb, tolerance)

        for y, x in candidates:
            score = self._check_offsets(roi_rgb, x, y, tolerance)
            if score is None:
                continue

            # Convert back to absolute coordinates
            abs_x = effective_roi.x + x
            abs_y = effective_roi.y + y
            results.append(MatchResult.from_point(abs_x, abs_y, score=score))

        # Sort by confidence (higher is better) to match Template contract
        results.sort(key=lambda r: r.score, reverse=True)
        if effective_max_count is not None:
            return results[:effective_max_count]
        return results

    def _validate_image(self, image: np.ndarray) -> None:
        """Validate image format.

        Args:
            image: Image to validate

        Raises:
            TypeError: Not a numpy array
            ValueError: Invalid dimensions
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be numpy.ndarray, got {type(image).__name__}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Image must be 3-channel BGR, got shape {image.shape}"
            )

    def _find_first_color_candidates(
        self, rgb_image: np.ndarray, tolerance: int
    ) -> np.ndarray:
        """Find all pixels matching the first color.

        Args:
            rgb_image: RGB image
            tolerance: Color tolerance (0-255)

        Returns:
            Array of (y, x) coordinates
        """
        # Use int16 to avoid overflow in subtraction
        image_i16 = rgb_image.astype(np.int16)

        # Check if all channels are within tolerance (use pre-converted int16)
        diff = np.abs(image_i16 - self._first_color_i16)
        match_mask = np.all(diff <= tolerance, axis=2)

        # Return (y, x) coordinates of matching pixels
        return np.argwhere(match_mask)

    def _check_offsets(
        self, rgb_image: np.ndarray, x: int, y: int, tolerance: int
    ) -> Optional[float]:
        """Check if all offset colors match and return similarity score.

        Args:
            rgb_image: RGB image
            x: First point x coordinate (in ROI)
            y: First point y coordinate (in ROI)
            tolerance: Color tolerance (0-255)

        Returns:
            Match confidence in [0, 1] if all offsets match, otherwise None
        """
        height, width = rgb_image.shape[:2]

        pixel = rgb_image[y, x].astype(np.int16)
        max_diff = int(np.abs(pixel - self._first_color_i16).max())
        if max_diff > tolerance:
            return None

        # Use pre-converted int16 colors
        for dx, dy, target_i16 in self._offsets_i16:
            tx, ty = x + dx, y + dy

            # Check bounds
            if tx < 0 or ty < 0 or tx >= width or ty >= height:
                return None

            # Check color match (use pre-converted int16 target)
            pixel = rgb_image[ty, tx].astype(np.int16)

            channel_diff = np.abs(pixel - target_i16)
            max_diff = max(max_diff, int(channel_diff.max()))

            if channel_diff.max() > tolerance:
                return None

        # Score degrades with the worst channel difference across all points
        return 1.0 - (max_diff / 255.0)
