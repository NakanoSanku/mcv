"""MCV core abstractions.

This module defines core data structures and abstract base classes:
- ROI: Region of Interest
- MatchResult: Match result
- Template: Template abstract base class
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class ROI:
    """Region of Interest.

    Defines a rectangular region using top-left corner (x, y) and dimensions.
    Coordinate system: origin at top-left, x increases rightward, y downward.

    Attributes:
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Region width (must be positive)
        height: Region height (must be positive)
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate ROI parameters."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"ROI dimensions must be positive, got width={self.width}, height={self.height}"
            )

    def clamp(self, image_width: int, image_height: int) -> Optional[ROI]:
        """Clamp ROI to image boundaries.

        Args:
            image_width: Image width
            image_height: Image height

        Returns:
            New ROI clamped to bounds, or None if completely outside
        """
        if image_width <= 0 or image_height <= 0:
            return None

        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(image_width, self.x + self.width)
        y2 = min(image_height, self.y + self.height)

        if x2 <= x1 or y2 <= y1:
            return None

        return ROI(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    @classmethod
    def from_sequence(cls, seq: Sequence[int]) -> ROI:
        """Create ROI from sequence.

        Args:
            seq: 4-element integer sequence [x, y, width, height]

        Returns:
            ROI object

        Raises:
            TypeError: Input is a string
            ValueError: Invalid sequence format
        """
        # Reject strings explicitly (they are Sequences but not valid ROI input)
        if isinstance(seq, (str, bytes)):
            raise TypeError("ROI cannot be created from string or bytes")
        if len(seq) != 4:
            raise ValueError(f"ROI sequence must have 4 elements, got {len(seq)}")
        return cls(x=int(seq[0]), y=int(seq[1]), width=int(seq[2]), height=int(seq[3]))


@dataclass(frozen=True)
class MatchResult:
    """Match result, unified for all template types.

    Attributes:
        top_left: Top-left corner (x, y)
        bottom_right: Bottom-right corner (x, y), exclusive (half-open interval)
        center: Center point (x, y)
        score: Match confidence, range [0, 1]
    """

    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    center: Tuple[int, int]
    score: float

    @property
    def is_point(self) -> bool:
        """Whether this is a point result (multi-color match returns point)."""
        return self.top_left == self.bottom_right

    @property
    def point(self) -> Tuple[int, int]:
        """Use as point, same as center."""
        return self.center

    @property
    def width(self) -> int:
        """Match region width."""
        return self.bottom_right[0] - self.top_left[0]

    @property
    def height(self) -> int:
        """Match region height."""
        return self.bottom_right[1] - self.top_left[1]

    @classmethod
    def from_box(
        cls, x: int, y: int, width: int, height: int, score: float
    ) -> MatchResult:
        """Create MatchResult from bounding box.

        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Region width
            height: Region height
            score: Match score

        Returns:
            MatchResult object
        """
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        center = (x + width // 2, y + height // 2)
        return cls(
            top_left=top_left,
            bottom_right=bottom_right,
            center=center,
            score=score,
        )

    @classmethod
    def from_point(cls, x: int, y: int, score: float = 1.0) -> MatchResult:
        """Create point result (for multi-color matching).

        Args:
            x: Point x coordinate
            y: Point y coordinate
            score: Match score, default 1.0

        Returns:
            MatchResult with top_left == bottom_right == center
        """
        point = (x, y)
        return cls(
            top_left=point,
            bottom_right=point,
            center=point,
            score=score,
        )


ROILike = Union[ROI, Sequence[int]]


class Template(ABC):
    """Template abstract base class.

    All concrete template types (image matching, multi-color, OCR, etc.)
    should inherit from this class. Instances have default parameters,
    method arguments can override defaults.

    Attributes:
        default_roi: Default search region, None means full image
        default_threshold: Default match threshold
        default_max_count: Default maximum result count
    """

    def __init__(
        self,
        roi: Optional[ROILike] = None,
        threshold: float = 0.8,
        max_count: int = 1,
    ) -> None:
        """Initialize template base class.

        Args:
            roi: Default search region, None means full image
            threshold: Default match threshold, range [0, 1]
            max_count: Default maximum result count, must be positive

        Raises:
            ValueError: Invalid threshold or max_count
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if max_count < 1:
            raise ValueError(f"max_count must be positive, got {max_count}")

        self.default_roi = self._normalize_roi(roi)
        self.default_threshold = threshold
        self.default_max_count = max_count

    @abstractmethod
    def find(
        self,
        image: np.ndarray,
        roi: Optional[ROILike] = None,
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        """Find single target.

        Args:
            image: Search image (numpy.ndarray, BGR format)
            roi: Search region, None uses default
            threshold: Match threshold, None uses default

        Returns:
            Match result, or None if not found
        """

    @abstractmethod
    def find_all(
        self,
        image: np.ndarray,
        roi: Optional[ROILike] = None,
        threshold: Optional[float] = None,
        max_count: Optional[int] = None,
    ) -> List[MatchResult]:
        """Find all targets.

        Args:
            image: Search image (numpy.ndarray, BGR format)
            roi: Search region, None uses default
            threshold: Match threshold, None uses default
            max_count: Maximum result count, None uses default

        Returns:
            List of match results, sorted by confidence descending
        """

    @staticmethod
    def _normalize_roi(roi: Optional[ROILike]) -> Optional[ROI]:
        """Normalize ROI input to ROI object.

        Args:
            roi: ROI object, 4-element sequence, or None

        Returns:
            ROI object or None

        Raises:
            ValueError: Invalid input format
        """
        if roi is None:
            return None
        if isinstance(roi, ROI):
            return roi
        if isinstance(roi, Sequence) and len(roi) == 4:
            return ROI.from_sequence(roi)
        raise ValueError("roi must be ROI object or 4-element integer sequence")

    def _resolve_roi(
        self,
        roi: Optional[ROILike],
        image_width: int,
        image_height: int,
    ) -> Optional[ROI]:
        """Resolve and validate ROI parameter.

        Args:
            roi: Input roi parameter
            image_width: Image width
            image_height: Image height

        Returns:
            Valid ROI object, or None if completely outside bounds
        """
        target = self._normalize_roi(roi) if roi is not None else self.default_roi
        if target is None:
            return ROI(x=0, y=0, width=image_width, height=image_height)
        return target.clamp(image_width, image_height)

    def _resolve_threshold(self, threshold: Optional[float]) -> float:
        """Resolve threshold parameter.

        Raises:
            ValueError: Invalid threshold value
        """
        value = threshold if threshold is not None else self.default_threshold
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        return value

    def _resolve_max_count(self, max_count: Optional[int]) -> int:
        """Resolve max_count parameter.

        Raises:
            ValueError: Invalid max_count value
        """
        value = max_count if max_count is not None else self.default_max_count
        if value < 1:
            raise ValueError(f"max_count must be positive, got {value}")
        return value
