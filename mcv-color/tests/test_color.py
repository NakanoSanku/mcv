"""Tests for MultiColorTemplate."""

import numpy as np
import pytest

from mcv.color import MultiColorTemplate


class TestColorNormalization:
    """Test color format normalization."""

    def test_hex_color_uppercase(self):
        """Test uppercase hex color."""
        template = MultiColorTemplate(first_color="FF5500", offsets=[])
        assert np.array_equal(template._first_color, [255, 85, 0])

    def test_hex_color_lowercase(self):
        """Test lowercase hex color."""
        template = MultiColorTemplate(first_color="ff5500", offsets=[])
        assert np.array_equal(template._first_color, [255, 85, 0])

    def test_hex_color_with_hash(self):
        """Test hex color with # prefix."""
        template = MultiColorTemplate(first_color="#FF5500", offsets=[])
        assert np.array_equal(template._first_color, [255, 85, 0])

    def test_rgb_tuple(self):
        """Test RGB tuple color."""
        template = MultiColorTemplate(first_color=(255, 85, 0), offsets=[])
        assert np.array_equal(template._first_color, [255, 85, 0])

    def test_rgb_list(self):
        """Test RGB list color."""
        template = MultiColorTemplate(first_color=[255, 85, 0], offsets=[])
        assert np.array_equal(template._first_color, [255, 85, 0])

    def test_invalid_hex_length(self):
        """Test invalid hex string length."""
        with pytest.raises(ValueError, match="6 characters"):
            MultiColorTemplate(first_color="FFF", offsets=[])

    def test_invalid_hex_chars(self):
        """Test invalid hex characters."""
        with pytest.raises(ValueError, match="Invalid hex"):
            MultiColorTemplate(first_color="GGGGGG", offsets=[])

    def test_invalid_tuple_length(self):
        """Test invalid tuple length."""
        with pytest.raises(ValueError, match="3 elements"):
            MultiColorTemplate(first_color=(255, 85), offsets=[])

    def test_invalid_color_type(self):
        """Test invalid color type."""
        with pytest.raises(TypeError, match="hex string or RGB tuple"):
            MultiColorTemplate(first_color=12345, offsets=[])

    def test_rgb_value_out_of_range(self):
        """Test RGB values out of valid range."""
        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            MultiColorTemplate(first_color=(300, 0, 0), offsets=[])

        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            MultiColorTemplate(first_color=(0, -5, 0), offsets=[])

    def test_rgb_non_numeric_value(self):
        """Test non-numeric RGB values."""
        with pytest.raises(TypeError, match="numeric"):
            MultiColorTemplate(first_color=("red", 0, 0), offsets=[])


class TestMultiColorTemplate:
    """Test MultiColorTemplate matching."""

    def create_test_image(self, width: int = 100, height: int = 100) -> np.ndarray:
        """Create a test image with known colors (BGR format)."""
        # Black background
        image = np.zeros((height, width, 3), dtype=np.uint8)
        return image

    def test_find_exact_match(self):
        """Test finding exact color match."""
        # Create image with specific color at (50, 50)
        image = self.create_test_image()
        # Set pixel at (50, 50) to RGB(30, 132, 30) = BGR(30, 132, 30)
        # Note: Our template uses RGB, image is BGR
        image[50, 50] = [30, 132, 30]  # BGR for RGB(30, 132, 30)

        template = MultiColorTemplate(first_color="1E841E", offsets=[])
        result = template.find(image)

        assert result is not None
        assert result.center == (50, 50)
        assert result.is_point

    def test_find_with_offsets(self):
        """Test finding with offset colors."""
        image = self.create_test_image()
        # First point at (50, 50): RGB(30, 132, 30)
        image[50, 50] = [30, 132, 30]  # BGR
        # Offset (+10, +5): RGB(46, 148, 46)
        image[55, 60] = [46, 148, 46]  # BGR

        template = MultiColorTemplate(
            first_color="1E841E",
            offsets=[(10, 5, "2E942E")],
        )
        result = template.find(image)

        assert result is not None
        assert result.center == (50, 50)

    def test_find_no_match(self):
        """Test when no match exists."""
        image = self.create_test_image()  # All black

        template = MultiColorTemplate(first_color="FF0000", offsets=[])
        result = template.find(image)

        assert result is None

    def test_find_offset_mismatch(self):
        """Test when first color matches but offset doesn't."""
        image = self.create_test_image()
        image[50, 50] = [30, 132, 30]  # First color matches
        # Offset color doesn't match (black)

        template = MultiColorTemplate(
            first_color="1E841E",
            offsets=[(10, 5, "2E942E")],
        )
        result = template.find(image)

        assert result is None

    def test_find_with_tolerance(self):
        """Test finding with color tolerance."""
        image = self.create_test_image()
        # Slightly different color: RGB(35, 137, 35) instead of (30, 132, 30)
        image[50, 50] = [35, 137, 35]  # BGR

        # Exact match should fail
        template_exact = MultiColorTemplate(first_color="1E841E", offsets=[], threshold=1.0)
        assert template_exact.find(image) is None

        # With tolerance should succeed
        template_tolerant = MultiColorTemplate(first_color="1E841E", offsets=[], threshold=0.95)
        result = template_tolerant.find(image)
        assert result is not None
        assert result.center == (50, 50)

    def test_find_all(self):
        """Test finding multiple matches."""
        image = self.create_test_image()
        # Multiple matching points
        image[20, 30] = [30, 132, 30]
        image[50, 50] = [30, 132, 30]
        image[70, 80] = [30, 132, 30]

        template = MultiColorTemplate(first_color="1E841E", offsets=[], max_count=10)
        results = template.find_all(image)

        assert len(results) == 3
        # Results should be in scan order (top-left to bottom-right)
        assert results[0].center == (30, 20)
        assert results[1].center == (50, 50)
        assert results[2].center == (80, 70)

    def test_find_all_with_max_count(self):
        """Test find_all respects max_count."""
        image = self.create_test_image()
        image[20, 30] = [30, 132, 30]
        image[50, 50] = [30, 132, 30]
        image[70, 80] = [30, 132, 30]

        template = MultiColorTemplate(first_color="1E841E", offsets=[])
        results = template.find_all(image, max_count=2)

        assert len(results) == 2

    def test_scores_and_sorting_with_tolerance(self):
        """Matches should carry confidence and be sorted by score."""
        image = self.create_test_image()
        # Exact match
        image[20, 30] = [30, 132, 30]
        # Slight deviation (max diff 10)
        image[40, 40] = [40, 142, 40]
        # Larger deviation (max diff 20)
        image[60, 60] = [50, 152, 50]

        template = MultiColorTemplate(
            first_color="1E841E",
            offsets=[],
            threshold=0.85,
            max_count=5,
        )

        results = template.find_all(image)

        assert [r.center for r in results] == [(30, 20), (40, 40), (60, 60)]
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(1.0)
        assert scores[-1] < 1.0

    def test_find_with_roi(self):
        """Test finding within ROI."""
        image = self.create_test_image()
        image[20, 20] = [30, 132, 30]  # Outside ROI
        image[60, 60] = [30, 132, 30]  # Inside ROI

        template = MultiColorTemplate(first_color="1E841E", offsets=[])
        # ROI: x=50, y=50, width=40, height=40
        result = template.find(image, roi=[50, 50, 40, 40])

        assert result is not None
        assert result.center == (60, 60)

    def test_offset_out_of_bounds(self):
        """Test offset pointing outside image bounds."""
        image = self.create_test_image()
        # Color at edge
        image[5, 95] = [30, 132, 30]

        template = MultiColorTemplate(
            first_color="1E841E",
            offsets=[(10, 0, "2E942E")],  # Would be x=105, out of bounds
        )
        result = template.find(image)

        assert result is None  # Should not match because offset is out of bounds


class TestValidation:
    """Test input validation."""

    def test_invalid_image_type(self):
        """Test with non-numpy array."""
        template = MultiColorTemplate(first_color="FF0000", offsets=[])
        with pytest.raises(TypeError, match="numpy.ndarray"):
            template.find([[1, 2, 3]])

    def test_invalid_image_dimensions(self):
        """Test with wrong image dimensions."""
        template = MultiColorTemplate(first_color="FF0000", offsets=[])
        # Grayscale image
        with pytest.raises(ValueError, match="3-channel"):
            template.find(np.zeros((100, 100), dtype=np.uint8))

    def test_invalid_threshold(self):
        """Test invalid threshold value."""
        with pytest.raises(ValueError, match="threshold"):
            MultiColorTemplate(first_color="FF0000", offsets=[], threshold=1.5)

    def test_invalid_max_count(self):
        """Test invalid max_count value."""
        with pytest.raises(ValueError, match="max_count"):
            MultiColorTemplate(first_color="FF0000", offsets=[], max_count=0)
