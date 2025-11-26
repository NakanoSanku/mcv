"""ResolutionMapper tests."""

import pytest

from mcv.base import MatchResult, ROI
from mcv.scaling import ResolutionMapper


class TestResolutionMapperInit:
    """Initialization validation tests."""

    def test_valid_sizes(self) -> None:
        """Valid sizes should create mapper."""
        mapper = ResolutionMapper((720, 1280), (1080, 1920))
        assert mapper.standard_size == (720, 1280)
        assert mapper.target_size == (1080, 1920)
        assert mapper.scale_x == 1.5
        assert mapper.scale_y == 1.5

    def test_invalid_size_zero_width(self) -> None:
        """Zero width should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ResolutionMapper((0, 1280), (1080, 1920))

    def test_invalid_size_negative_height(self) -> None:
        """Negative height should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            ResolutionMapper((720, -1), (1080, 1920))

    def test_invalid_size_wrong_format(self) -> None:
        """Non-tuple size should raise ValueError."""
        with pytest.raises(ValueError):
            ResolutionMapper(720, (1080, 1920))  # type: ignore

    def test_invalid_size_wrong_length(self) -> None:
        """Wrong tuple length should raise ValueError."""
        with pytest.raises(ValueError, match="width, height"):
            ResolutionMapper((720,), (1080, 1920))  # type: ignore

    def test_invalid_rounding_mode(self) -> None:
        """Unsupported rounding mode should raise ValueError."""
        with pytest.raises(ValueError, match="rounding"):
            ResolutionMapper((100, 200), (200, 400), rounding="invalid")  # type: ignore

    def test_repr(self) -> None:
        """Test string representation."""
        mapper = ResolutionMapper((720, 1280), (1080, 1920))
        assert "720" in repr(mapper)
        assert "1080" in repr(mapper)


class TestPointMapping:
    """Point mapping and round-trip tests."""

    def test_map_point_basic(self) -> None:
        """Basic point mapping works correctly."""
        mapper = ResolutionMapper((720, 1280), (1080, 1920))
        # scale is 1.5 in both directions
        assert mapper.map_point((100, 200)) == (150, 300)

    def test_unmap_point_basic(self) -> None:
        """Basic point unmapping works correctly."""
        mapper = ResolutionMapper((720, 1280), (1080, 1920))
        assert mapper.unmap_point((150, 300)) == (100, 200)

    def test_map_unmap_round_trip(self) -> None:
        """Mapping to target and back returns original point."""
        mapper = ResolutionMapper((720, 1280), (1080, 1920))
        point = (120, 480)
        mapped = mapper.map_point(point)
        assert mapped == (180, 720)
        assert mapper.unmap_point(mapped) == point

    def test_one_to_one_mapping(self) -> None:
        """1:1 mapping keeps coordinates unchanged."""
        mapper = ResolutionMapper((500, 500), (500, 500))
        assert mapper.map_point((10, 20)) == (10, 20)
        assert mapper.unmap_point((300, 400)) == (300, 400)

    def test_non_uniform_scale(self) -> None:
        """Non-uniform scaling applies independent factors."""
        mapper = ResolutionMapper((100, 200), (200, 300))
        # x scale: 2.0, y scale: 1.5
        assert mapper.map_point((50, 100)) == (100, 150)
        assert mapper.unmap_point((100, 150)) == (50, 100)

    def test_origin_point(self) -> None:
        """Origin point always maps to origin."""
        mapper = ResolutionMapper((720, 1280), (1080, 1920))
        assert mapper.map_point((0, 0)) == (0, 0)
        assert mapper.unmap_point((0, 0)) == (0, 0)


class TestROIMapping:
    """ROI mapping and round-trip tests."""

    def test_map_roi_basic(self) -> None:
        """Basic ROI mapping works correctly."""
        mapper = ResolutionMapper((200, 400), (300, 600))
        # scale: 1.5 in both directions
        roi = ROI(x=10, y=20, width=50, height=80)
        mapped = mapper.map_roi(roi)
        assert mapped.x == 15
        assert mapped.y == 30
        assert mapped.width == 75
        assert mapped.height == 120

    def test_unmap_roi_basic(self) -> None:
        """Basic ROI unmapping works correctly."""
        mapper = ResolutionMapper((200, 400), (300, 600))
        roi = ROI(x=15, y=30, width=75, height=120)
        unmapped = mapper.unmap_roi(roi)
        assert unmapped.x == 10
        assert unmapped.y == 20
        assert unmapped.width == 50
        assert unmapped.height == 80

    def test_map_unmap_roi_round_trip(self) -> None:
        """ROI mapping preserves geometry after round-trip."""
        mapper = ResolutionMapper((200, 400), (300, 600))
        roi = ROI(x=10, y=20, width=50, height=80)
        mapped = mapper.map_roi(roi)
        unmapped = mapper.unmap_roi(mapped)
        assert unmapped == roi

    def test_small_roi_minimum_dimension(self) -> None:
        """Very small ROI maintains minimum dimension of 1."""
        # Scaling down from 1000 to 10 with a 1x1 ROI
        mapper = ResolutionMapper((1000, 1000), (10, 10))
        roi = ROI(x=0, y=0, width=5, height=5)
        # width/height would be 0.05, but should be at least 1
        mapped = mapper.map_roi(roi)
        assert mapped.width >= 1
        assert mapped.height >= 1


class TestMatchResultMapping:
    """MatchResult mapping tests."""

    def test_map_point_match(self) -> None:
        """Point MatchResult mapping works correctly."""
        mapper = ResolutionMapper((200, 400), (300, 600))
        # scale: 1.5 in both directions
        result = MatchResult.from_point(x=20, y=40, score=0.9)
        mapped = mapper.map_match(result)
        assert mapped.center == (30, 60)
        assert mapped.is_point is True
        assert mapped.score == 0.9

    def test_unmap_point_match(self) -> None:
        """Point MatchResult unmapping works correctly."""
        mapper = ResolutionMapper((200, 400), (300, 600))
        result = MatchResult.from_point(x=30, y=60, score=0.9)
        unmapped = mapper.unmap_match(result)
        assert unmapped.center == (20, 40)
        assert unmapped.is_point is True
        assert unmapped.score == 0.9

    def test_map_unmap_point_match_round_trip(self) -> None:
        """Point MatchResult round-trip preserves data."""
        mapper = ResolutionMapper((200, 400), (300, 600))
        result = MatchResult.from_point(x=20, y=40, score=0.85)
        mapped = mapper.map_match(result)
        unmapped = mapper.unmap_match(mapped)
        assert unmapped == result

    def test_map_box_match(self) -> None:
        """Box MatchResult mapping works correctly."""
        mapper = ResolutionMapper((100, 200), (200, 400))
        # scale: 2.0 in both directions
        result = MatchResult.from_box(x=10, y=20, width=30, height=40, score=0.8)
        mapped = mapper.map_match(result)
        assert mapped.top_left == (20, 40)
        assert mapped.bottom_right == (80, 120)
        assert mapped.width == 60
        assert mapped.height == 80
        assert mapped.score == 0.8

    def test_unmap_box_match(self) -> None:
        """Box MatchResult unmapping works correctly."""
        mapper = ResolutionMapper((100, 200), (200, 400))
        result = MatchResult.from_box(x=20, y=40, width=60, height=80, score=0.8)
        unmapped = mapper.unmap_match(result)
        assert unmapped.top_left == (10, 20)
        assert unmapped.bottom_right == (40, 60)
        assert unmapped.width == 30
        assert unmapped.height == 40
        assert unmapped.score == 0.8

    def test_map_unmap_box_match_round_trip(self) -> None:
        """Box MatchResult round-trip preserves data."""
        mapper = ResolutionMapper((100, 200), (200, 400))
        result = MatchResult.from_box(x=10, y=20, width=30, height=40, score=0.75)
        mapped = mapper.map_match(result)
        unmapped = mapper.unmap_match(mapped)
        assert unmapped == result


class TestRoundingModes:
    """Rounding mode behavior tests."""

    def test_round_mode_default(self) -> None:
        """Default round mode uses Python's round (round-half-to-even)."""
        mapper = ResolutionMapper((2, 2), (3, 3), rounding="round")
        # 1 * 1.5 = 1.5, round(1.5) = 2 (round-half-to-even)
        assert mapper.map_point((1, 1)) == (2, 2)

    def test_floor_mode(self) -> None:
        """Floor mode always rounds down."""
        mapper = ResolutionMapper((2, 2), (3, 3), rounding="floor")
        # 1 * 1.5 = 1.5, floor(1.5) = 1
        assert mapper.map_point((1, 1)) == (1, 1)

    def test_ceil_mode(self) -> None:
        """Ceil mode always rounds up."""
        mapper = ResolutionMapper((2, 2), (3, 3), rounding="ceil")
        # 1 * 1.5 = 1.5, ceil(1.5) = 2
        assert mapper.map_point((1, 1)) == (2, 2)

    def test_trunc_mode(self) -> None:
        """Trunc mode truncates toward zero."""
        mapper = ResolutionMapper((2, 2), (3, 3), rounding="trunc")
        # 1 * 1.5 = 1.5, int(1.5) = 1
        assert mapper.map_point((1, 1)) == (1, 1)

    def test_rounding_affects_roi(self) -> None:
        """Rounding mode affects ROI dimensions."""
        # scale: 1.5
        mapper_floor = ResolutionMapper((10, 10), (15, 15), rounding="floor")
        mapper_ceil = ResolutionMapper((10, 10), (15, 15), rounding="ceil")

        roi = ROI(x=1, y=1, width=3, height=3)
        # width: 3 * 1.5 = 4.5
        assert mapper_floor.map_roi(roi).width == 4
        assert mapper_ceil.map_roi(roi).width == 5


class TestEdgeCases:
    """Edge case tests."""

    def test_scale_up_large_factor(self) -> None:
        """Large scale up factor works correctly."""
        mapper = ResolutionMapper((100, 100), (4000, 4000))
        assert mapper.scale_x == 40.0
        assert mapper.scale_y == 40.0
        assert mapper.map_point((50, 50)) == (2000, 2000)

    def test_scale_down_large_factor(self) -> None:
        """Large scale down factor works correctly."""
        mapper = ResolutionMapper((4000, 4000), (100, 100))
        assert mapper.scale_x == 0.025
        assert mapper.scale_y == 0.025
        assert mapper.map_point((2000, 2000)) == (50, 50)

    def test_asymmetric_scaling(self) -> None:
        """Asymmetric scaling (different x/y factors) works."""
        mapper = ResolutionMapper((100, 200), (400, 300))
        # x scale: 4.0, y scale: 1.5
        assert mapper.scale_x == 4.0
        assert mapper.scale_y == 1.5
        point = (10, 20)
        mapped = mapper.map_point(point)
        assert mapped == (40, 30)
