"""Coordinate scaling utilities for resolution mapping.

This module provides resolution-independent coordinate transformation,
enabling scripts written for a standard resolution to work across
different device resolutions.
"""

from __future__ import annotations

import math
from typing import Callable, Literal, Tuple

from .base import MatchResult, ROI

RoundingMode = Literal["round", "floor", "ceil", "trunc"]


class ResolutionMapper:
    """Map coordinates and regions between standard and target resolutions.

    This mapper applies independent linear scaling on x/y axes (stretch mode)
    and provides bidirectional transforms for points, ROIs, and MatchResults.

    Typical usage::

        # Script authored at 720x1280, running on 1080x1920 device
        mapper = ResolutionMapper(standard_size=(720, 1280), target_size=(1080, 1920))

        # Map ROI to target resolution before searching
        target_roi = mapper.map_roi(standard_roi)
        result = template.find(image, roi=target_roi)

        # Map result back to standard coordinates
        if result:
            standard_result = mapper.unmap_match(result)

    Attributes:
        standard_size: (width, height) used when authoring scripts.
        target_size: (width, height) of the actual device/image.
        scale_x: Horizontal scale factor (target_width / standard_width).
        scale_y: Vertical scale factor (target_height / standard_height).
    """

    __slots__ = (
        "_standard_width",
        "_standard_height",
        "_target_width",
        "_target_height",
        "_scale_x",
        "_scale_y",
        "_round_fn",
    )

    def __init__(
        self,
        standard_size: Tuple[int, int],
        target_size: Tuple[int, int],
        rounding: RoundingMode = "round",
    ) -> None:
        """Initialize resolution mapper.

        Args:
            standard_size: (width, height) used when authoring scripts.
            target_size: (width, height) of the actual device/image.
            rounding: Strategy for converting float coordinates to int.
                - "round": Round half to even (default, Python's built-in round)
                - "floor": Always round down
                - "ceil": Always round up
                - "trunc": Truncate toward zero

        Raises:
            ValueError: If sizes are non-positive or have wrong format.
        """
        self._standard_width, self._standard_height = self._validate_size(
            standard_size, "standard_size"
        )
        self._target_width, self._target_height = self._validate_size(
            target_size, "target_size"
        )
        self._scale_x = self._target_width / self._standard_width
        self._scale_y = self._target_height / self._standard_height
        self._round_fn = self._get_rounding_function(rounding)

    @property
    def standard_size(self) -> Tuple[int, int]:
        """Standard resolution (width, height)."""
        return (self._standard_width, self._standard_height)

    @property
    def target_size(self) -> Tuple[int, int]:
        """Target resolution (width, height)."""
        return (self._target_width, self._target_height)

    @property
    def scale_x(self) -> float:
        """Horizontal scale factor."""
        return self._scale_x

    @property
    def scale_y(self) -> float:
        """Vertical scale factor."""
        return self._scale_y

    @staticmethod
    def _validate_size(size: Tuple[int, int], name: str) -> Tuple[int, int]:
        """Validate and normalize size tuple.

        Args:
            size: Size tuple to validate
            name: Parameter name for error messages

        Returns:
            Validated (width, height) tuple

        Raises:
            ValueError: If size format is invalid or values non-positive
        """
        if not hasattr(size, "__len__") or len(size) != 2:
            raise ValueError(f"{name} must be (width, height) tuple, got {size!r}")

        try:
            width, height = int(size[0]), int(size[1])
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name} must contain integers: {e}") from e

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{name} dimensions must be positive, got width={width}, height={height}"
            )
        return width, height

    @staticmethod
    def _get_rounding_function(mode: RoundingMode) -> Callable[[float], int]:
        """Get rounding function for the specified mode.

        Args:
            mode: Rounding mode name

        Returns:
            Function that converts float to int

        Raises:
            ValueError: If mode is not recognized
        """
        if mode == "round":
            return lambda v: int(round(v))
        if mode == "floor":
            return lambda v: int(math.floor(v))
        if mode == "ceil":
            return lambda v: int(math.ceil(v))
        if mode == "trunc":
            return int
        raise ValueError(
            f"rounding must be 'round', 'floor', 'ceil', or 'trunc', got {mode!r}"
        )

    def map_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Map point from standard to target coordinates.

        Args:
            point: (x, y) in standard coordinates

        Returns:
            (x, y) in target coordinates
        """
        x, y = point
        return (
            self._round_fn(x * self._scale_x),
            self._round_fn(y * self._scale_y),
        )

    def unmap_point(self, point: Tuple[int, int]) -> Tuple[int, int]:
        """Map point from target back to standard coordinates.

        Args:
            point: (x, y) in target coordinates

        Returns:
            (x, y) in standard coordinates
        """
        x, y = point
        return (
            self._round_fn(x / self._scale_x),
            self._round_fn(y / self._scale_y),
        )

    def map_roi(self, roi: ROI) -> ROI:
        """Map ROI from standard to target coordinates.

        Note:
            Width and height are clamped to minimum of 1 to ensure valid ROI.

        Args:
            roi: ROI in standard coordinates

        Returns:
            ROI in target coordinates
        """
        return self._transform_roi(roi, self._scale_x, self._scale_y)

    def unmap_roi(self, roi: ROI) -> ROI:
        """Map ROI from target back to standard coordinates.

        Note:
            Width and height are clamped to minimum of 1 to ensure valid ROI.

        Args:
            roi: ROI in target coordinates

        Returns:
            ROI in standard coordinates
        """
        return self._transform_roi(roi, 1.0 / self._scale_x, 1.0 / self._scale_y)

    def map_match(self, result: MatchResult) -> MatchResult:
        """Map MatchResult from standard to target coordinates.

        Args:
            result: MatchResult in standard coordinates

        Returns:
            MatchResult in target coordinates with same score
        """
        return self._transform_match(result, self._scale_x, self._scale_y)

    def unmap_match(self, result: MatchResult) -> MatchResult:
        """Map MatchResult from target back to standard coordinates.

        Args:
            result: MatchResult in target coordinates

        Returns:
            MatchResult in standard coordinates with same score
        """
        return self._transform_match(result, 1.0 / self._scale_x, 1.0 / self._scale_y)

    def _transform_roi(self, roi: ROI, scale_x: float, scale_y: float) -> ROI:
        """Apply scale transformation to ROI.

        Args:
            roi: Source ROI
            scale_x: Horizontal scale factor
            scale_y: Vertical scale factor

        Returns:
            Transformed ROI with width/height clamped to minimum of 1
        """
        x = self._round_fn(roi.x * scale_x)
        y = self._round_fn(roi.y * scale_y)
        width = self._round_fn(roi.width * scale_x)
        height = self._round_fn(roi.height * scale_y)

        # Ensure minimum dimension of 1 to avoid invalid ROI
        width = max(1, width)
        height = max(1, height)

        return ROI(x=x, y=y, width=width, height=height)

    def _transform_match(
        self, result: MatchResult, scale_x: float, scale_y: float
    ) -> MatchResult:
        """Apply scale transformation to MatchResult.

        Args:
            result: Source MatchResult
            scale_x: Horizontal scale factor
            scale_y: Vertical scale factor

        Returns:
            Transformed MatchResult with preserved score
        """
        if result.is_point:
            # Point result: just transform the center
            x = self._round_fn(result.center[0] * scale_x)
            y = self._round_fn(result.center[1] * scale_y)
            return MatchResult.from_point(x=x, y=y, score=result.score)

        # Box result: transform as ROI then create new MatchResult
        roi = ROI(
            x=result.top_left[0],
            y=result.top_left[1],
            width=result.width,
            height=result.height,
        )
        scaled_roi = self._transform_roi(roi, scale_x, scale_y)
        return MatchResult.from_box(
            x=scaled_roi.x,
            y=scaled_roi.y,
            width=scaled_roi.width,
            height=scaled_roi.height,
            score=result.score,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ResolutionMapper("
            f"standard_size={self.standard_size}, "
            f"target_size={self.target_size})"
        )
