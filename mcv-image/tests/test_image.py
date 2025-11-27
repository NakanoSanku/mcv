"""mcv-image template matching tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from mcv.base import ROI
from mcv.image import ImageTemplate

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def e2e_screenshot() -> np.ndarray:
    """Load real screenshot for end-to-end tests."""
    path = FIXTURES_DIR / "image.png"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None, f"Failed to load screenshot: {path}"
    return image


@pytest.fixture(scope="session")
def e2e_template_image() -> np.ndarray:
    """Load real template image for end-to-end tests."""
    path = FIXTURES_DIR / "template.png"
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert template is not None, f"Failed to load template: {path}"
    return template


class TestImageTemplate:
    """ImageTemplate tests."""

    @pytest.fixture
    def pattern_template(self) -> np.ndarray:
        """Create pattern template (10x10 with checkerboard pattern)."""
        template = np.zeros((10, 10, 3), dtype=np.uint8)
        # Create a simple pattern
        template[0:5, 0:5] = 255
        template[5:10, 5:10] = 255
        return template

    @pytest.fixture
    def search_image_with_match(self) -> np.ndarray:
        """Create search image with match target.

        100x100 gray background with pattern at (45, 45).
        """
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Place pattern at (45, 45)
        image[45:50, 45:50] = 255
        image[50:55, 50:55] = 255
        image[45:50, 50:55] = 0
        image[50:55, 45:50] = 0
        return image

    @pytest.fixture
    def search_image_with_multiple_matches(self) -> np.ndarray:
        """Create search image with multiple match targets."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        
        # Place patterns at different locations
        for y, x in [(20, 20), (60, 60), (100, 100), (150, 150)]:
            image[y:y+5, x:x+5] = 255
            image[y+5:y+10, x+5:x+10] = 255
            image[y:y+5, x+5:x+10] = 0
            image[y+5:y+10, x:x+5] = 0
        return image

    def test_init_valid(self, pattern_template: np.ndarray) -> None:
        """Test valid initialization."""
        template = ImageTemplate(pattern_template, threshold=0.9)
        assert template.default_threshold == 0.9
        assert template.grayscale is True
        assert template.method == cv2.TM_CCOEFF_NORMED

    def test_init_with_grayscale_template(self) -> None:
        """Test initialization with grayscale template."""
        gray_template = np.zeros((10, 10), dtype=np.uint8)
        gray_template[0:5, 0:5] = 255
        template = ImageTemplate(gray_template)
        assert template._template_processed.ndim == 2

    def test_init_invalid_type(self) -> None:
        """Test invalid type raises error."""
        with pytest.raises(TypeError, match="numpy.ndarray"):
            ImageTemplate("not an array")  # type: ignore

    def test_init_invalid_shape(self) -> None:
        """Test invalid shape raises error."""
        invalid = np.ones((10, 10, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="grayscale or 3-channel BGR"):
            ImageTemplate(invalid)

    def test_find_match(
        self, pattern_template: np.ndarray, search_image_with_match: np.ndarray
    ) -> None:
        """Test find finds match."""
        template = ImageTemplate(pattern_template, threshold=0.8)
        result = template.find(search_image_with_match)

        assert result is not None
        assert result.score > 0.8
        # Match should be around (45, 45)
        assert 42 <= result.top_left[0] <= 48
        assert 42 <= result.top_left[1] <= 48

    def test_find_no_match(self, pattern_template: np.ndarray) -> None:
        """Test find returns None when no match."""
        # Create search image with no matching pattern
        search = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # Add different pattern that won't match
        search[45:55, 45:55] = 64
        
        template = ImageTemplate(pattern_template, threshold=0.95)
        result = template.find(search)
        assert result is None

    def test_find_all_multiple(
        self,
        pattern_template: np.ndarray,
        search_image_with_multiple_matches: np.ndarray,
    ) -> None:
        """Test find_all finds multiple matches."""
        template = ImageTemplate(pattern_template, threshold=0.8)
        results = template.find_all(
            search_image_with_multiple_matches,
            max_count=10,
        )

        assert len(results) >= 3
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_find_all_with_max_count(
        self,
        pattern_template: np.ndarray,
        search_image_with_multiple_matches: np.ndarray,
    ) -> None:
        """Test find_all respects max_count."""
        template = ImageTemplate(pattern_template, threshold=0.8)
        results = template.find_all(
            search_image_with_multiple_matches,
            max_count=2,
        )
        assert len(results) <= 2

    def test_find_with_roi(
        self, pattern_template: np.ndarray, search_image_with_match: np.ndarray
    ) -> None:
        """Test find with ROI restriction."""
        template = ImageTemplate(pattern_template, threshold=0.8)

        # ROI doesn't contain target, should not find
        result = template.find(search_image_with_match, roi=ROI(0, 0, 30, 30))
        assert result is None

        # ROI contains target, should find
        result = template.find(search_image_with_match, roi=ROI(40, 40, 30, 30))
        assert result is not None

    def test_find_with_roi_sequence(
        self, pattern_template: np.ndarray, search_image_with_match: np.ndarray
    ) -> None:
        """Test find with sequence ROI."""
        template = ImageTemplate(pattern_template, threshold=0.8)
        result = template.find(search_image_with_match, roi=[40, 40, 30, 30])
        assert result is not None

    def test_find_template_larger_than_roi(self, pattern_template: np.ndarray) -> None:
        """Test template larger than ROI returns empty."""
        search = np.ones((100, 100, 3), dtype=np.uint8) * 128
        template = ImageTemplate(pattern_template, threshold=0.8)

        # ROI smaller than template
        result = template.find(search, roi=ROI(0, 0, 5, 5))
        assert result is None

    def test_find_template_larger_than_image(
        self, pattern_template: np.ndarray
    ) -> None:
        """Test template larger than image returns empty."""
        small_search = np.ones((5, 5, 3), dtype=np.uint8) * 128
        template = ImageTemplate(pattern_template, threshold=0.8)
        result = template.find(small_search)
        assert result is None

    def test_sqdiff_method(
        self, pattern_template: np.ndarray, search_image_with_match: np.ndarray
    ) -> None:
        """Test TM_SQDIFF_NORMED method."""
        template = ImageTemplate(pattern_template, threshold=0.8)
        template.method = cv2.TM_SQDIFF_NORMED
        result = template.find(search_image_with_match)
        assert result is not None
        assert result.score > 0.8

    def test_default_threshold_override(
        self, pattern_template: np.ndarray, search_image_with_match: np.ndarray
    ) -> None:
        """Test overriding default threshold."""
        template = ImageTemplate(pattern_template, threshold=0.99)

        # Use lower threshold
        result_override = template.find(search_image_with_match, threshold=0.8)
        assert result_override is not None

    def test_nms_basic(self) -> None:
        """Test NMS removes overlapping matches."""
        # Create distinct pattern
        template_img = np.zeros((10, 10, 3), dtype=np.uint8)
        template_img[2:8, 2:8] = 255
        
        # Create image with single target
        image = np.ones((100, 100, 3), dtype=np.uint8) * 50
        image[42:52, 42:52] = 0
        image[44:50, 44:50] = 255

        template = ImageTemplate(template_img, threshold=0.7)
        template.nms_threshold = 0.3
        results = template.find_all(image, max_count=100)

        # NMS should keep only a few non-overlapping results
        assert len(results) <= 10

    def test_result_coordinates_in_original_image(
        self, pattern_template: np.ndarray
    ) -> None:
        """Test result coordinates are in original image space."""
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        # Place pattern at (100, 100)
        image[100:105, 100:105] = 255
        image[105:110, 105:110] = 255
        image[100:105, 105:110] = 0
        image[105:110, 100:105] = 0

        template = ImageTemplate(pattern_template, threshold=0.8)
        result = template.find(image, roi=ROI(80, 80, 50, 50))

        assert result is not None
        # Coordinates should be in original image space
        assert result.top_left[0] >= 80
        assert result.top_left[1] >= 80

    def test_invalid_search_image_type(self, pattern_template: np.ndarray) -> None:
        """Test invalid search image type."""
        template = ImageTemplate(pattern_template)
        with pytest.raises(TypeError, match="numpy.ndarray"):
            template.find("not an array")  # type: ignore

    def test_invalid_search_image_shape(self, pattern_template: np.ndarray) -> None:
        """Test invalid search image shape."""
        template = ImageTemplate(pattern_template)
        invalid = np.ones((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="grayscale or 3-channel BGR"):
            template.find(invalid)

    def test_grayscale_with_gray_input(self) -> None:
        """Test grayscale mode with grayscale input."""
        gray_template = np.zeros((10, 10), dtype=np.uint8)
        gray_template[0:5, 0:5] = 255

        gray_search = np.ones((100, 100), dtype=np.uint8) * 128
        gray_search[45:50, 45:50] = 255
        gray_search[50:55, 50:55] = 255

        template = ImageTemplate(gray_template, threshold=0.8)
        result = template.find(gray_search)
        assert result is not None


class TestImageTemplateEdgeCases:
    """Edge case tests."""

    def test_exact_size_match(self) -> None:
        """Test template and search image same size."""
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[0:5, 0:5] = 255
        template = ImageTemplate(image, threshold=0.9)
        result = template.find(image)

        assert result is not None
        assert result.score > 0.99
        assert result.top_left == (0, 0)

    def test_roi_clamping(self) -> None:
        """Test ROI auto-clamped to image bounds."""
        template_img = np.zeros((10, 10, 3), dtype=np.uint8)
        template_img[0:5, 0:5] = 255
        
        search = np.ones((100, 100, 3), dtype=np.uint8) * 128
        search[80:85, 80:85] = 255
        search[85:90, 85:90] = 255

        template = ImageTemplate(template_img, threshold=0.8)
        # ROI extends beyond image bounds
        result = template.find(search, roi=ROI(70, 70, 100, 100))

        assert result is not None

    def test_roi_completely_outside(self) -> None:
        """Test ROI completely outside image."""
        template_img = np.zeros((10, 10, 3), dtype=np.uint8)
        template_img[0:5, 0:5] = 255
        search = np.ones((100, 100, 3), dtype=np.uint8) * 128

        template = ImageTemplate(template_img, threshold=0.8)
        result = template.find(search, roi=ROI(200, 200, 50, 50))

        assert result is None


class TestParameterValidation:
    """Parameter validation tests."""

    def test_invalid_threshold_init(self) -> None:
        """Test invalid threshold in __init__ raises error."""
        template_img = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="threshold must be in"):
            ImageTemplate(template_img, threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be in"):
            ImageTemplate(template_img, threshold=-0.1)

    def test_invalid_threshold_find(self) -> None:
        """Test invalid threshold in find raises error."""
        template_img = np.zeros((10, 10, 3), dtype=np.uint8)
        template = ImageTemplate(template_img, threshold=0.8)
        search = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="threshold must be in"):
            template.find(search, threshold=2.0)

    def test_invalid_max_count_find_all(self) -> None:
        """Test invalid max_count in find_all raises error."""
        template_img = np.zeros((10, 10, 3), dtype=np.uint8)
        template = ImageTemplate(template_img, threshold=0.8)
        search = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="max_count must be positive"):
            template.find_all(search, max_count=0)

    def test_score_clamped_to_range(self) -> None:
        """Test match scores are always in [0, 1] range."""
        # Create simple template
        template_img = np.zeros((10, 10, 3), dtype=np.uint8)
        template_img[0:5, 0:5] = 255

        # Create search image with perfect match
        search = np.zeros((100, 100, 3), dtype=np.uint8)
        search[45:50, 45:50] = 255
        search[50:55, 50:55] = 255

        template = ImageTemplate(template_img, threshold=0.5)
        results = template.find_all(search, max_count=100)

        for result in results:
            assert 0.0 <= result.score <= 1.0


class TestImageTemplatePyramid:
    """Tests for pyramid matching mode."""

    def test_pyramid_finds_same_location(
        self,
        e2e_screenshot: np.ndarray,
        e2e_template_image: np.ndarray,
    ) -> None:
        """Pyramid mode should find same location as normal mode."""
        normal = ImageTemplate(e2e_template_image, threshold=0.8)
        normal.use_pyramid = False
        pyramid = ImageTemplate(e2e_template_image, threshold=0.8)
        pyramid.use_pyramid = True

        r_normal = normal.find(e2e_screenshot)
        r_pyramid = pyramid.find(e2e_screenshot)

        assert r_normal is not None
        assert r_pyramid is not None
        assert abs(r_normal.top_left[0] - r_pyramid.top_left[0]) <= 2
        assert abs(r_normal.top_left[1] - r_pyramid.top_left[1]) <= 2

    def test_pyramid_score_accuracy(
        self,
        e2e_screenshot: np.ndarray,
        e2e_template_image: np.ndarray,
    ) -> None:
        """Pyramid mode score should be close to normal mode."""
        normal = ImageTemplate(e2e_template_image, threshold=0.8)
        normal.use_pyramid = False
        pyramid = ImageTemplate(e2e_template_image, threshold=0.8)
        pyramid.use_pyramid = True

        r_normal = normal.find(e2e_screenshot)
        r_pyramid = pyramid.find(e2e_screenshot)

        assert r_normal is not None
        assert r_pyramid is not None
        assert abs(r_normal.score - r_pyramid.score) < 0.05

    def test_pyramid_with_roi(
        self,
        e2e_screenshot: np.ndarray,
        e2e_template_image: np.ndarray,
    ) -> None:
        """Pyramid mode should work with ROI."""
        template = ImageTemplate(e2e_template_image, threshold=0.8)
        result = template.find(e2e_screenshot, roi=ROI(400, 200, 100, 100))
        assert result is not None

    def test_pyramid_no_match(
        self,
        e2e_screenshot: np.ndarray,
        e2e_template_image: np.ndarray,
    ) -> None:
        """Pyramid mode should return None when no match."""
        template = ImageTemplate(e2e_template_image, threshold=0.8)
        result = template.find(e2e_screenshot, roi=ROI(0, 0, 100, 100))
        assert result is None

    def test_pyramid_refine_failure_falls_back(self) -> None:
        """Refine failure should not abort the entire pyramid search."""

        # Simple checkerboard template reused across tests
        pattern_template = np.zeros((10, 10, 3), dtype=np.uint8)
        pattern_template[0:5, 0:5] = 255
        pattern_template[5:10, 5:10] = 255

        # Build search image with a matching patch at (45, 45)
        search_image_with_match = np.ones((100, 100, 3), dtype=np.uint8) * 128
        search_image_with_match[45:50, 45:50] = 255
        search_image_with_match[50:55, 50:55] = 255
        search_image_with_match[45:50, 50:55] = 0
        search_image_with_match[50:55, 45:50] = 0

        class NoRefineTemplate(ImageTemplate):
            def _refine_at_base_level(self, *args, **kwargs):  # type: ignore[override]
                return None

        template = NoRefineTemplate(pattern_template, threshold=0.8)
        template.use_pyramid = True
        template.min_pyramid_size = 4
        template.max_pyramid_level = 2

        result = template.find(search_image_with_match)
        assert result is not None


class TestImageTemplateE2E:
    """End-to-end tests using real image assets.

    Ground truth for template.png ("探索" lantern) in image.png:
    - top_left: (429, 221)
    - center: (454, 260)
    - size: 51x79
    """

    EXPECTED_TOP_LEFT = (429, 221)
    EXPECTED_CENTER = (454, 260)
    COORDINATE_TOLERANCE = 5

    @pytest.fixture(scope="class")
    def e2e_template(self, e2e_template_image: np.ndarray) -> ImageTemplate:
        """Create ImageTemplate instance for E2E tests."""
        return ImageTemplate(e2e_template_image, threshold=0.8)

    def test_find_template_in_screenshot(
        self,
        e2e_template: ImageTemplate,
        e2e_screenshot: np.ndarray,
        e2e_template_image: np.ndarray,
    ) -> None:
        """Template should be found in the screenshot."""
        result = e2e_template.find(e2e_screenshot)

        assert result is not None, "Template not found in screenshot"
        assert result.score >= 0.8
        assert result.width == e2e_template_image.shape[1]
        assert result.height == e2e_template_image.shape[0]

    def test_find_correct_location(
        self,
        e2e_template: ImageTemplate,
        e2e_screenshot: np.ndarray,
    ) -> None:
        """Found location should match expected coordinates."""
        result = e2e_template.find(e2e_screenshot)

        assert result is not None
        x, y = result.top_left
        cx, cy = result.center

        assert abs(x - self.EXPECTED_TOP_LEFT[0]) <= self.COORDINATE_TOLERANCE
        assert abs(y - self.EXPECTED_TOP_LEFT[1]) <= self.COORDINATE_TOLERANCE
        assert abs(cx - self.EXPECTED_CENTER[0]) <= self.COORDINATE_TOLERANCE
        assert abs(cy - self.EXPECTED_CENTER[1]) <= self.COORDINATE_TOLERANCE

    def test_find_returns_valid_coordinates(
        self,
        e2e_template: ImageTemplate,
        e2e_screenshot: np.ndarray,
    ) -> None:
        """Found coordinates should be within image bounds."""
        result = e2e_template.find(e2e_screenshot)

        assert result is not None
        img_h, img_w = e2e_screenshot.shape[:2]
        x, y = result.top_left
        assert 0 <= x < img_w
        assert 0 <= y < img_h
        assert x + result.width <= img_w
        assert y + result.height <= img_h

    def test_roi_excludes_target(
        self,
        e2e_template: ImageTemplate,
        e2e_screenshot: np.ndarray,
        e2e_template_image: np.ndarray,
    ) -> None:
        """ROI not covering target should not find match."""
        baseline = e2e_template.find(e2e_screenshot)
        assert baseline is not None, "Baseline match required for ROI test"

        tmpl_h, tmpl_w = e2e_template_image.shape[:2]
        img_h, img_w = e2e_screenshot.shape[:2]
        match_x, match_y = baseline.top_left

        # Create ROI in opposite corner from match, with bounds protection
        if match_x < img_w // 2:
            roi_x = max(0, img_w - tmpl_w - 10)
        else:
            roi_x = 0
        if match_y < img_h // 2:
            roi_y = max(0, img_h - tmpl_h - 10)
        else:
            roi_y = 0

        roi_away = ROI(roi_x, roi_y, tmpl_w + 10, tmpl_h + 10)
        result = e2e_template.find(e2e_screenshot, roi=roi_away)
        assert result is None, "Should not find template in ROI away from target"

    def test_roi_includes_target(
        self,
        e2e_template: ImageTemplate,
        e2e_screenshot: np.ndarray,
    ) -> None:
        """ROI covering target should find match."""
        baseline = e2e_template.find(e2e_screenshot)
        assert baseline is not None, "Baseline match required for ROI test"

        x, y = baseline.top_left
        w, h = baseline.width, baseline.height

        roi_covering = ROI(x, y, w, h)
        result = e2e_template.find(e2e_screenshot, roi=roi_covering)
        assert result is not None, "Should find template in ROI covering target"

    def test_low_threshold_accepts(
        self,
        e2e_template: ImageTemplate,
        e2e_screenshot: np.ndarray,
    ) -> None:
        """Lower threshold should accept match."""
        result = e2e_template.find(e2e_screenshot, threshold=0.7)
        assert result is not None, "Should accept at lower threshold"

    def test_find_all_returns_single_match(
        self,
        e2e_template: ImageTemplate,
        e2e_screenshot: np.ndarray,
    ) -> None:
        """find_all should return exactly one match for unique template."""
        results = e2e_template.find_all(e2e_screenshot, max_count=10)
        assert len(results) == 1, "Should find exactly one match"
        assert results[0].top_left == e2e_template.find(e2e_screenshot).top_left
