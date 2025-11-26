"""mcv-base core module tests."""

import pytest

from mcv.base import MatchResult, ROI, Template


class TestROI:
    """ROI dataclass tests."""

    def test_create_valid_roi(self) -> None:
        """Test creating valid ROI."""
        roi = ROI(x=10, y=20, width=100, height=50)
        assert roi.x == 10
        assert roi.y == 20
        assert roi.width == 100
        assert roi.height == 50

    def test_create_invalid_roi_zero_width(self) -> None:
        """Test creating ROI with zero width raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            ROI(x=0, y=0, width=0, height=10)

    def test_create_invalid_roi_negative_height(self) -> None:
        """Test creating ROI with negative height raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            ROI(x=0, y=0, width=10, height=-5)

    def test_to_tuple(self) -> None:
        """Test to_tuple method."""
        roi = ROI(x=10, y=20, width=30, height=40)
        assert roi.to_tuple() == (10, 20, 30, 40)

    def test_from_sequence(self) -> None:
        """Test from_sequence class method."""
        roi = ROI.from_sequence([10, 20, 30, 40])
        assert roi == ROI(x=10, y=20, width=30, height=40)

    def test_from_sequence_tuple(self) -> None:
        """Test creating ROI from tuple."""
        roi = ROI.from_sequence((5, 10, 15, 20))
        assert roi == ROI(x=5, y=10, width=15, height=20)

    def test_from_sequence_invalid_length(self) -> None:
        """Test sequence with wrong length raises error."""
        with pytest.raises(ValueError, match="4 elements"):
            ROI.from_sequence([1, 2, 3])

    def test_clamp_within_bounds(self) -> None:
        """Test ROI within image bounds unchanged."""
        roi = ROI(x=10, y=10, width=50, height=50)
        clamped = roi.clamp(100, 100)
        assert clamped == roi

    def test_clamp_partial_overlap(self) -> None:
        """Test ROI partially outside is clamped."""
        roi = ROI(x=80, y=80, width=50, height=50)
        clamped = roi.clamp(100, 100)
        assert clamped is not None
        assert clamped.x == 80
        assert clamped.y == 80
        assert clamped.width == 20
        assert clamped.height == 20

    def test_clamp_negative_start(self) -> None:
        """Test ROI with negative start is clamped."""
        roi = ROI(x=-10, y=-10, width=50, height=50)
        clamped = roi.clamp(100, 100)
        assert clamped is not None
        assert clamped.x == 0
        assert clamped.y == 0
        assert clamped.width == 40
        assert clamped.height == 40

    def test_clamp_completely_outside(self) -> None:
        """Test ROI completely outside returns None."""
        roi = ROI(x=200, y=200, width=50, height=50)
        clamped = roi.clamp(100, 100)
        assert clamped is None

    def test_clamp_zero_image_size(self) -> None:
        """Test zero image size returns None."""
        roi = ROI(x=0, y=0, width=50, height=50)
        assert roi.clamp(0, 100) is None
        assert roi.clamp(100, 0) is None

    def test_frozen(self) -> None:
        """Test ROI is immutable."""
        roi = ROI(x=10, y=20, width=30, height=40)
        with pytest.raises(AttributeError):
            roi.x = 100  # type: ignore


class TestMatchResult:
    """MatchResult dataclass tests."""

    def test_create_match_result(self) -> None:
        """Test creating MatchResult."""
        result = MatchResult(
            top_left=(10, 20),
            bottom_right=(60, 70),
            center=(35, 45),
            score=0.95,
        )
        assert result.top_left == (10, 20)
        assert result.bottom_right == (60, 70)
        assert result.center == (35, 45)
        assert result.score == 0.95

    def test_is_point_false(self) -> None:
        """Test area result has is_point False."""
        result = MatchResult(
            top_left=(10, 20),
            bottom_right=(60, 70),
            center=(35, 45),
            score=0.95,
        )
        assert result.is_point is False

    def test_is_point_true(self) -> None:
        """Test point result has is_point True."""
        result = MatchResult(
            top_left=(35, 45),
            bottom_right=(35, 45),
            center=(35, 45),
            score=1.0,
        )
        assert result.is_point is True

    def test_point_property(self) -> None:
        """Test point property equals center."""
        result = MatchResult(
            top_left=(10, 20),
            bottom_right=(60, 70),
            center=(35, 45),
            score=0.95,
        )
        assert result.point == result.center

    def test_width_height(self) -> None:
        """Test width and height properties."""
        result = MatchResult(
            top_left=(10, 20),
            bottom_right=(60, 70),
            center=(35, 45),
            score=0.95,
        )
        assert result.width == 50
        assert result.height == 50

    def test_from_box(self) -> None:
        """Test from_box class method."""
        result = MatchResult.from_box(x=10, y=20, width=50, height=40, score=0.9)
        assert result.top_left == (10, 20)
        assert result.bottom_right == (60, 60)
        assert result.center == (35, 40)
        assert result.score == 0.9

    def test_from_point(self) -> None:
        """Test from_point class method."""
        result = MatchResult.from_point(x=100, y=200, score=0.85)
        assert result.top_left == (100, 200)
        assert result.bottom_right == (100, 200)
        assert result.center == (100, 200)
        assert result.score == 0.85
        assert result.is_point is True

    def test_from_point_default_score(self) -> None:
        """Test from_point default score is 1.0."""
        result = MatchResult.from_point(x=50, y=50)
        assert result.score == 1.0

    def test_frozen(self) -> None:
        """Test MatchResult is immutable."""
        result = MatchResult(
            top_left=(10, 20),
            bottom_right=(60, 70),
            center=(35, 45),
            score=0.95,
        )
        with pytest.raises(AttributeError):
            result.score = 0.5  # type: ignore


class TestTemplate:
    """Template abstract base class tests."""

    def test_cannot_instantiate(self) -> None:
        """Test cannot instantiate abstract class."""
        with pytest.raises(TypeError):
            Template()  # type: ignore

    def test_normalize_roi_none(self) -> None:
        """Test _normalize_roi handles None."""
        assert Template._normalize_roi(None) is None

    def test_normalize_roi_instance(self) -> None:
        """Test _normalize_roi handles ROI instance."""
        roi = ROI(x=10, y=20, width=30, height=40)
        assert Template._normalize_roi(roi) is roi

    def test_normalize_roi_sequence(self) -> None:
        """Test _normalize_roi handles sequence."""
        result = Template._normalize_roi([10, 20, 30, 40])
        assert result == ROI(x=10, y=20, width=30, height=40)

    def test_normalize_roi_invalid(self) -> None:
        """Test _normalize_roi handles invalid input."""
        with pytest.raises(ValueError):
            Template._normalize_roi([1, 2, 3])  # type: ignore

    def test_normalize_roi_string_rejected(self) -> None:
        """Test _normalize_roi rejects string input."""
        with pytest.raises(TypeError, match="string or bytes"):
            Template._normalize_roi("abcd")  # type: ignore


class TestROIStringRejection:
    """Test ROI.from_sequence rejects strings."""

    def test_from_sequence_rejects_string(self) -> None:
        """Test from_sequence rejects string."""
        with pytest.raises(TypeError, match="string or bytes"):
            ROI.from_sequence("1234")  # type: ignore

    def test_from_sequence_rejects_bytes(self) -> None:
        """Test from_sequence rejects bytes."""
        with pytest.raises(TypeError, match="string or bytes"):
            ROI.from_sequence(b"1234")  # type: ignore
