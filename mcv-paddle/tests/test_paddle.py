"""mcv-paddle OCR template tests."""

import builtins
import re
import sys
import types
from typing import List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from mcv.base import MatchResult, ROI
from mcv.paddle import (
    PaddleOCRMatchResult,
    PaddleOCRTemplate,
    PaddleOCRNotInstalledError,
    clear_ocr_cache,
)


class TestPaddleOCRMatchResult:
    """PaddleOCRMatchResult dataclass tests."""

    def test_properties_basic(self) -> None:
        """Test basic property computation."""
        quad = ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))
        result = PaddleOCRMatchResult(text="hello", confidence=0.95, quad=quad)

        assert result.text == "hello"
        assert result.confidence == 0.95
        assert result.top_left == (0, 0)
        assert result.bottom_right == (10, 5)
        assert result.center == (5, 2)
        assert result.width == 10
        assert result.height == 5

    def test_properties_with_float_coords(self) -> None:
        """Test property computation with fractional coordinates."""
        quad = ((1.2, 2.8), (5.7, 2.0), (5.0, 7.9), (1.0, 8.0))
        result = PaddleOCRMatchResult(text="ok", confidence=0.9, quad=quad)

        # top_left uses floor, bottom_right uses ceil
        assert result.top_left == (1, 2)
        assert result.bottom_right == (6, 8)
        assert result.center == (3, 5)
        assert result.width == 5
        assert result.height == 6

    def test_to_match_result(self) -> None:
        """Test conversion to core MatchResult."""
        quad = ((10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0))
        ocr_result = PaddleOCRMatchResult(text="test", confidence=0.85, quad=quad)

        match_result = ocr_result.to_match_result()

        assert isinstance(match_result, MatchResult)
        assert match_result.top_left == ocr_result.top_left
        assert match_result.bottom_right == ocr_result.bottom_right
        assert match_result.center == ocr_result.center
        assert match_result.score == pytest.approx(0.85)

    def test_minimum_size_guarantee(self) -> None:
        """Test that tiny quads still produce at least 1x1 bounding box."""
        # Very small quad
        quad = ((5.1, 5.1), (5.2, 5.1), (5.2, 5.2), (5.1, 5.2))
        result = PaddleOCRMatchResult(text=".", confidence=0.9, quad=quad)

        # Should be at least 1x1
        assert result.width >= 1
        assert result.height >= 1


class DummyOCR:
    """Dummy PaddleOCR replacement for testing."""

    def __init__(self, outputs: List) -> None:
        self.outputs = outputs
        self.calls: List[Tuple[int, int]] = []

    def ocr(self, image, det=True, rec=True, cls=True):
        if hasattr(image, "shape"):
            self.calls.append((image.shape[0], image.shape[1]))
        return self.outputs


class TestPaddleOCRTemplate:
    """PaddleOCRTemplate behavior tests with mocked PaddleOCR."""

    @pytest.fixture(autouse=True)
    def clear_cache_fixture(self) -> None:
        """Clear OCR cache before and after each test."""
        clear_ocr_cache()
        yield
        clear_ocr_cache()

    @pytest.fixture
    def sample_ocr_output(self) -> List:
        """Sample PaddleOCR output with two text entries."""
        return [
            [
                (
                    [[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]],
                    ("确认", 0.95),
                ),
                (
                    [[30.0, 0.0], [60.0, 0.0], [60.0, 10.0], [30.0, 10.0]],
                    ("金币:12345", 0.85),
                ),
                (
                    [[0.0, 20.0], [40.0, 20.0], [40.0, 30.0], [0.0, 30.0]],
                    ("low confidence", 0.3),
                ),
            ]
        ]

    def test_init_validation_threshold(self) -> None:
        """Test invalid threshold raises error."""
        with pytest.raises(ValueError, match="threshold must be in"):
            PaddleOCRTemplate(pattern="x", threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be in"):
            PaddleOCRTemplate(pattern="x", threshold=-0.1)

    def test_init_validation_max_count(self) -> None:
        """Test invalid max_count raises error."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern="x", threshold=0.5)
        with pytest.raises(ValueError, match="max_count must be positive"):
            template.find_all(img, max_count=0)

    def test_find_exact_match(self, monkeypatch, sample_ocr_output) -> None:
        """Test finding text with exact substring match."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern="确认", threshold=0.5)

        result = template.find(img)

        assert result is not None
        assert result.text == "确认"
        assert result.confidence == pytest.approx(0.95)

    def test_find_all_with_pattern(self, monkeypatch, sample_ocr_output) -> None:
        """Test find_all filters by pattern."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern="金币", threshold=0.5)

        results = template.find_all(img)

        assert len(results) == 1
        assert results[0].text == "金币:12345"

    def test_pattern_none_returns_all(self, monkeypatch, sample_ocr_output) -> None:
        """Test pattern=None returns all text above threshold."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern=None, threshold=0.5)

        results = template.find_all(img)

        # Should return 2 (excluding low confidence one)
        assert len(results) == 2
        texts = {r.text for r in results}
        assert "确认" in texts
        assert "金币:12345" in texts

    def test_regex_matching(self, monkeypatch, sample_ocr_output) -> None:
        """Test regex pattern matching."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Match numbers
        template = PaddleOCRTemplate(pattern=r"\d+", regex=True, threshold=0.5)

        results = template.find_all(img)

        assert len(results) == 1
        assert results[0].text == "金币:12345"

    def test_compiled_regex_pattern(self, monkeypatch, sample_ocr_output) -> None:
        """Test pre-compiled regex pattern."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        pattern = re.compile(r"金币:\d+")
        template = PaddleOCRTemplate(pattern=pattern, threshold=0.5)

        result = template.find(img)

        assert result is not None
        assert result.text == "金币:12345"

    def test_threshold_filtering(self, monkeypatch, sample_ocr_output) -> None:
        """Test confidence threshold filtering."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # High threshold excludes most results
        template = PaddleOCRTemplate(pattern=None, threshold=0.9)
        results = template.find_all(img)
        assert len(results) == 1
        assert results[0].text == "确认"

        # Low threshold includes more
        results = template.find_all(img, threshold=0.2)
        assert len(results) == 3

    def test_threshold_override(self, monkeypatch, sample_ocr_output) -> None:
        """Test threshold can be overridden per call."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern=None, threshold=0.99)

        # Default threshold too high
        results = template.find_all(img)
        assert len(results) == 0

        # Override with lower threshold
        results = template.find_all(img, threshold=0.5)
        assert len(results) == 2

    def test_roi_crop_and_coordinate_offset(self, monkeypatch) -> None:
        """Test ROI cropping and coordinate offset."""
        output = [
            [
                (
                    [[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]],
                    ("roi_text", 0.9),
                )
            ]
        ]
        dummy = DummyOCR(output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((200, 200, 3), dtype=np.uint8)
        roi = ROI(x=50, y=100, width=80, height=60)

        template = PaddleOCRTemplate(pattern=None, threshold=0.5)
        results = template.find_all(img, roi=roi)

        assert len(results) == 1
        result = results[0]

        # Coordinates should be offset by ROI position
        assert result.top_left == (55, 105)  # 5+50, 5+100
        assert result.bottom_right == (65, 115)  # 15+50, 15+100

        # Verify dummy received cropped image dimensions
        assert dummy.calls == [(60, 80)]

    def test_roi_sequence_format(self, monkeypatch) -> None:
        """Test ROI as list/tuple."""
        output = [
            [
                (
                    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                    ("test", 0.9),
                )
            ]
        ]
        dummy = DummyOCR(output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern=None, threshold=0.5)

        # ROI as list
        results = template.find_all(img, roi=[10, 20, 30, 40])
        assert len(results) == 1
        assert results[0].top_left == (10, 20)

    def test_max_count_limit(self, monkeypatch, sample_ocr_output) -> None:
        """Test max_count limits results."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern=None, threshold=0.2)

        results = template.find_all(img, max_count=2)
        assert len(results) == 2

        # Override max_count
        results = template.find_all(img, max_count=1)
        assert len(results) == 1

    def test_results_sorted_by_confidence(self, monkeypatch, sample_ocr_output) -> None:
        """Test results are sorted by confidence descending."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern=None, threshold=0.2)

        results = template.find_all(img)

        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_grayscale_image(self, monkeypatch, sample_ocr_output) -> None:
        """Test grayscale image is accepted and converted."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100), dtype=np.uint8)  # Grayscale
        template = PaddleOCRTemplate(pattern="确认", threshold=0.5)

        result = template.find(img)
        assert result is not None

    def test_invalid_image_type(self) -> None:
        """Test invalid image type raises error."""
        template = PaddleOCRTemplate(pattern="test")

        with pytest.raises(TypeError, match="numpy.ndarray"):
            template.find("not an array")  # type: ignore

    def test_invalid_image_shape(self) -> None:
        """Test invalid image shape raises error."""
        template = PaddleOCRTemplate(pattern="test")

        # 4-channel image
        invalid = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D grayscale or 3-channel BGR"):
            template.find(invalid)

    def test_empty_ocr_output(self, monkeypatch) -> None:
        """Test handling of empty OCR output."""
        dummy = DummyOCR([])
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern=None, threshold=0.5)

        results = template.find_all(img)
        assert results == []

        result = template.find(img)
        assert result is None

    def test_no_match_found(self, monkeypatch, sample_ocr_output) -> None:
        """Test when no text matches pattern."""
        dummy = DummyOCR(sample_ocr_output)
        monkeypatch.setattr("mcv.paddle._get_ocr_client", lambda **_: dummy)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        template = PaddleOCRTemplate(pattern="不存在的文字", threshold=0.5)

        result = template.find(img)
        assert result is None

        results = template.find_all(img)
        assert results == []


class TestDependencyAndCache:
    """Dependency and cache handling tests."""

    @pytest.fixture(autouse=True)
    def clear_cache_fixture(self) -> None:
        """Clear cache before and after each test."""
        clear_ocr_cache()
        yield
        clear_ocr_cache()

    def test_missing_paddleocr_raises_clear_error(self, monkeypatch) -> None:
        """Test clear error when paddleocr not installed."""
        # Import the actual function to test
        from mcv.paddle import _get_ocr_client

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "paddleocr":
                raise ImportError("No module named 'paddleocr'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(PaddleOCRNotInstalledError, match="paddleocr is not installed"):
            _get_ocr_client(lang="ch", use_angle_cls=True, use_gpu=False)

    def test_cache_reuse(self, monkeypatch) -> None:
        """Test OCR client is cached and reused."""
        from mcv.paddle import _get_ocr_client

        created = []

        class FakePaddleOCR:
            def __init__(self, **kwargs) -> None:
                created.append(kwargs)

        fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
        monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

        # First call creates client
        _get_ocr_client(lang="ch", use_angle_cls=True, use_gpu=False)
        assert len(created) == 1

        # Second call reuses cached client
        _get_ocr_client(lang="ch", use_angle_cls=True, use_gpu=False)
        assert len(created) == 1

        # Different config creates new client
        _get_ocr_client(lang="en", use_angle_cls=True, use_gpu=False)
        assert len(created) == 2

    def test_clear_cache(self, monkeypatch) -> None:
        """Test clear_ocr_cache forces new client creation."""
        from mcv.paddle import _get_ocr_client

        created = []

        class FakePaddleOCR:
            def __init__(self, **kwargs) -> None:
                created.append(kwargs)

        fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
        monkeypatch.setitem(sys.modules, "paddleocr", fake_module)

        _get_ocr_client(lang="ch", use_angle_cls=True, use_gpu=False)
        assert len(created) == 1

        clear_ocr_cache()

        _get_ocr_client(lang="ch", use_angle_cls=True, use_gpu=False)
        assert len(created) == 2
