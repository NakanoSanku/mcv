"""OpenCV-based image template matching.

This module provides image template matching using cv2.matchTemplate,
supporting grayscale conversion, multiple matching methods, NMS,
and optional pyramid acceleration.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from mcv.base import MatchResult, ROI, Template

ROILike = Union[ROI, Sequence[int]]
ImageLike = Union[np.ndarray, str, Path]


class ImageTemplate(Template):
    """OpenCV-based image template matching.

    Uses cv2.matchTemplate for template matching, supporting:
    - Multiple matching methods (default TM_CCOEFF_NORMED)
    - Optional grayscale conversion
    - NMS for overlapping matches

    Example:
        >>> import cv2
        >>> from mcv.image import ImageTemplate
        >>> # Load from numpy array
        >>> screen = cv2.imread("screenshot.png")
        >>> template_img = cv2.imread("button.png")
        >>> template = ImageTemplate(template_img, threshold=0.9)
        >>> result = template.find(screen)
        >>> if result:
        ...     print(f"Found: {result.center}, confidence: {result.score:.2f}")
        >>>
        >>> # Or load directly from path
        >>> template = ImageTemplate("button.png", threshold=0.9)
        >>> result = template.find(screen)
    """

    _SQDIFF_METHODS = (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)

    def __init__(
        self,
        template_image: ImageLike,
        *,
        roi: Optional[ROILike] = None,
        threshold: float = 0.8,
    ) -> None:
        """Initialize image template.

        Args:
            template_image: Template image (numpy.ndarray, BGR or grayscale)
                            OR path to image file (str or pathlib.Path).
                            Images loaded from paths are automatically converted to grayscale.
            roi: Default search region
            threshold: Default match threshold

        Raises:
            TypeError: Invalid image type
            ValueError: Unsupported image format or invalid image file
            FileNotFoundError: Template image file not found
        """
        # max_count is intentionally fixed at 1 for ImageTemplate.
        # find_all() ignores this default and returns all matches unless
        # max_count is explicitly provided by the caller.
        super().__init__(roi=roi, threshold=threshold, max_count=1)

        # Load image from path if needed
        if isinstance(template_image, (str, Path)):
            template_array = self._load_image_from_path(template_image)
        else:
            template_array = template_image

        self._validate_image(template_array)
        self.template_image = template_array
        # Always enable grayscale matching for robustness and performance.
        self.grayscale = True
        # Default matching method and NMS settings are fixed for simplicity.
        self.method = cv2.TM_CCOEFF_NORMED
        self.nms_threshold = 0.5
        # Always enable pyramid matching for single-target search.
        # Advanced users can modify these attributes after initialization
        # if they need to fine-tune performance.
        self.use_pyramid = True
        self.max_pyramid_level = 4
        self.min_pyramid_size = 16

        self._template_processed = self._preprocess(template_array)
        self._template_height, self._template_width = self._template_processed.shape[:2]
        self._template_pyramid: List[np.ndarray] = [self._template_processed]

    def find(
        self,
        image: np.ndarray,
        roi: Optional[ROILike] = None,
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        """Find single best match.

        Args:
            image: Search image (BGR format)
            roi: Search region
            threshold: Match threshold

        Returns:
            Best match result, or None if not found
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
            image: Search image (BGR format)
            roi: Search region
            threshold: Match threshold
            max_count: Maximum result count. If None, return all matches.

        Returns:
            List of match results, sorted by confidence descending
        """
        self._validate_image(image)

        img_height, img_width = image.shape[:2]
        effective_roi = self._resolve_roi(roi, img_width, img_height)

        if effective_roi is None:
            return []

        roi_image = self._crop_roi(image, effective_roi)

        roi_h, roi_w = roi_image.shape[:2]
        if roi_h < self._template_height or roi_w < self._template_width:
            return []

        search_processed = self._preprocess(roi_image)

        effective_threshold = self._resolve_threshold(threshold)
        effective_max_count: Optional[int]
        if max_count is not None:
            effective_max_count = self._resolve_max_count(max_count)
        else:
            effective_max_count = None

        # Only enable pyramid acceleration when caller explicitly requests
        # a single best match (find / find_all with max_count=1).
        if self.use_pyramid and max_count == 1:
            return self._match_with_pyramid(
                search_processed, effective_roi, effective_threshold
            )

        match_result = cv2.matchTemplate(
            search_processed,
            self._template_processed,
            self.method,
        )

        candidates = self._collect_candidates(match_result, effective_threshold)

        if not candidates:
            return []

        matches = [
            MatchResult.from_box(
                x=pt[0] + effective_roi.x,
                y=pt[1] + effective_roi.y,
                width=self._template_width,
                height=self._template_height,
                score=score,
            )
            for pt, score in candidates
        ]

        return self._apply_nms(matches, effective_max_count)

    def _load_image_from_path(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path using np.fromfile + cv2.imdecode.

        Args:
            file_path: Path to image file

        Returns:
            Loaded image as grayscale numpy.ndarray

        Raises:
            FileNotFoundError: If file does not exist or is not a file
            ValueError: If file is not a valid image
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Template image file not found: {file_path}")

        if not path.is_file():
            raise FileNotFoundError(f"Path is not a file: {file_path}")

        try:
            data = np.fromfile(str(path), dtype=np.uint8)
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")

        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(
                f"Failed to decode image from {file_path}. "
                "File may not be a valid image format."
            )

        return image

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

        if image.ndim == 2:
            return
        if image.ndim == 3 and image.shape[2] == 3:
            return

        raise ValueError(
            f"Image must be 2D grayscale or 3-channel BGR, got shape {image.shape}"
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to match grayscale/color configuration.

        Args:
            image: Original image

        Returns:
            Processed image with channel count aligned to matching mode
        """
        if not self.grayscale:
            # Color mode requires 3 channels; expand grayscale inputs to BGR.
            if image.ndim == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image

        if image.ndim == 2:
            return image

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _crop_roi(self, image: np.ndarray, roi: ROI) -> np.ndarray:
        """Crop ROI region.

        Args:
            image: Original image
            roi: Crop region

        Returns:
            Cropped image
        """
        x1, y1 = roi.x, roi.y
        x2, y2 = roi.x + roi.width, roi.y + roi.height
        return image[y1:y2, x1:x2]

    def _collect_candidates(
        self,
        match_result: np.ndarray,
        threshold: float,
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Collect candidate matches above threshold.

        Args:
            match_result: cv2.matchTemplate result matrix
            threshold: Match threshold

        Returns:
            Candidate list [((x, y), score), ...], sorted by score descending
        """
        if self.method in self._SQDIFF_METHODS:
            match_result = 1.0 - match_result

        # Clamp scores to [0, 1] range for consistent API
        match_result = np.clip(match_result, 0.0, 1.0)

        locations = np.where(match_result >= threshold)
        if locations[0].size == 0:
            return []

        candidates: List[Tuple[Tuple[int, int], float]] = []
        for y, x in zip(locations[0], locations[1]):
            score = float(match_result[y, x])
            candidates.append(((int(x), int(y)), score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def _apply_nms(
        self,
        matches: List[MatchResult],
        max_count: Optional[int],
    ) -> List[MatchResult]:
        """Apply Non-Maximum Suppression.

        Args:
            matches: Candidate matches (sorted by score descending)
            max_count: Maximum results to keep; None means no explicit limit

        Returns:
            Filtered match list
        """
        if not matches:
            return []

        boxes = np.array(
            [
                [m.top_left[0], m.top_left[1], m.bottom_right[0], m.bottom_right[1]]
                for m in matches
            ],
            dtype=np.float32,
        )
        scores = np.array([m.score for m in matches], dtype=np.float32)

        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)

            if max_count is not None and len(keep) >= max_count:
                break

            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (
                boxes[order[1:], 3] - boxes[order[1:], 1]
            )
            union = area_i + area_others - intersection

            iou = intersection / (union + 1e-8)

            remaining_mask = iou <= self.nms_threshold
            order = order[1:][remaining_mask]

        return [matches[idx] for idx in keep]

    def _match_with_pyramid(
        self,
        search_processed: np.ndarray,
        effective_roi: ROI,
        threshold: float,
    ) -> List[MatchResult]:
        """Coarse-to-fine pyramid matching for single best result.

        Uses coarse levels to quickly locate candidates, then refines
        at level 0 for accurate score and coordinates.

        Args:
            search_processed: Preprocessed search image
            effective_roi: ROI for coordinate offset
            threshold: Match threshold

        Returns:
            List with single match result, or empty list
        """
        level = self._select_pyramid_level(search_processed)
        if level == 0:
            return self._match_single_scale(
                search_processed, effective_roi, threshold
            )

        search_pyramid = self._build_pyramid(search_processed, level)
        coarse_threshold = max(0.5, threshold - 0.15)

        for lvl in range(level, -1, -1):
            template_lvl = self._get_template_at_level(lvl)
            search_lvl = search_pyramid[lvl]

            if (
                search_lvl.shape[0] < template_lvl.shape[0]
                or search_lvl.shape[1] < template_lvl.shape[1]
            ):
                continue

            result = cv2.matchTemplate(search_lvl, template_lvl, self.method)
            score = self._normalize_score(result)
            _, max_val, _, max_loc = cv2.minMaxLoc(score)

            current_threshold = threshold if lvl == 0 else coarse_threshold

            if max_val >= current_threshold or lvl == 0:
                if lvl > 0:
                    refined = self._refine_at_base_level(
                        search_pyramid[0], max_loc, lvl, threshold
                    )
                    if refined:
                        x = refined[0] + effective_roi.x
                        y = refined[1] + effective_roi.y
                        return [
                            MatchResult.from_box(
                                x=x,
                                y=y,
                                width=self._template_width,
                                height=self._template_height,
                                score=refined[2],
                            )
                        ]
                    # If refinement fails, continue to finer levels instead of
                    # bailing out and missing potential matches.
                    continue

                if max_val < threshold:
                    return []

                return [
                    MatchResult.from_box(
                        x=max_loc[0] + effective_roi.x,
                        y=max_loc[1] + effective_roi.y,
                        width=self._template_width,
                        height=self._template_height,
                        score=float(max_val),
                    )
                ]

        return []

    def _refine_at_base_level(
        self,
        base_image: np.ndarray,
        coarse_loc: Tuple[int, int],
        coarse_level: int,
        threshold: float,
    ) -> Optional[Tuple[int, int, float]]:
        """Refine match at base level around coarse location.

        Args:
            base_image: Full resolution search image
            coarse_loc: Location found at coarse level
            coarse_level: The pyramid level where match was found
            threshold: Match threshold

        Returns:
            Tuple of (x, y, score) if refined match passes threshold, else None
        """
        scale = 2**coarse_level
        margin = scale * 2

        cx = coarse_loc[0] * scale
        cy = coarse_loc[1] * scale

        x1 = max(0, cx - margin)
        y1 = max(0, cy - margin)
        x2 = min(base_image.shape[1], cx + self._template_width + margin)
        y2 = min(base_image.shape[0], cy + self._template_height + margin)

        if x2 - x1 < self._template_width or y2 - y1 < self._template_height:
            return None

        window = base_image[y1:y2, x1:x2]
        result = cv2.matchTemplate(window, self._template_processed, self.method)
        score = self._normalize_score(result)
        _, max_val, _, max_loc = cv2.minMaxLoc(score)

        if max_val < threshold:
            return None

        return (x1 + max_loc[0], y1 + max_loc[1], float(max_val))

    def _normalize_score(self, match_result: np.ndarray) -> np.ndarray:
        """Normalize match result to [0, 1] range.

        Args:
            match_result: Raw matchTemplate result

        Returns:
            Normalized score matrix
        """
        if self.method in self._SQDIFF_METHODS:
            match_result = 1.0 - match_result
        return np.clip(match_result, 0.0, 1.0)

    def _match_single_scale(
        self,
        search_processed: np.ndarray,
        effective_roi: ROI,
        threshold: float,
    ) -> List[MatchResult]:
        """Single-scale matching fallback.

        Args:
            search_processed: Preprocessed search image
            effective_roi: ROI for coordinate offset
            threshold: Match threshold

        Returns:
            List with single match result, or empty list
        """
        result = cv2.matchTemplate(
            search_processed, self._template_processed, self.method
        )

        if self.method in self._SQDIFF_METHODS:
            result = 1.0 - result
        result = np.clip(result, 0.0, 1.0)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < threshold:
            return []

        return [
            MatchResult.from_box(
                x=max_loc[0] + effective_roi.x,
                y=max_loc[1] + effective_roi.y,
                width=self._template_width,
                height=self._template_height,
                score=float(max_val),
            )
        ]

    def _select_pyramid_level(self, search_image: np.ndarray) -> int:
        """Determine pyramid depth based on smallest dimension.

        Args:
            search_image: Search image

        Returns:
            Optimal pyramid level (0 means no pyramid)
        """
        min_dim = min(
            search_image.shape[0],
            search_image.shape[1],
            self._template_height,
            self._template_width,
        )

        if min_dim < 2 * self.min_pyramid_size:
            return 0

        max_level = int(math.log2(min_dim / self.min_pyramid_size))
        return min(self.max_pyramid_level, max(0, max_level))

    def _build_pyramid(self, image: np.ndarray, level: int) -> List[np.ndarray]:
        """Build image pyramid using cv2.pyrDown.

        Args:
            image: Base image
            level: Number of downscale steps

        Returns:
            List of images from original to smallest
        """
        pyramid = [image]
        current = image
        for _ in range(level):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def _get_template_at_level(self, level: int) -> np.ndarray:
        """Get or build template at requested pyramid level.

        Args:
            level: Pyramid level (0 = original)

        Returns:
            Template image at specified level
        """
        while len(self._template_pyramid) <= level:
            last = self._template_pyramid[-1]
            self._template_pyramid.append(cv2.pyrDown(last))
        return self._template_pyramid[level]
