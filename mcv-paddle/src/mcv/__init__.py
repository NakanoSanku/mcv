"""MCV PaddleOCR integration module.

This module provides OCR-based text matching using PaddleOCR.
"""

from pkgutil import extend_path

from .paddle import (
    OCRMatchResult,
    OCRTemplate,
    PaddleOCRNotInstalledError,
    clear_ocr_cache,
)

__path__ = extend_path(__path__, __name__)
__all__ = [
    "OCRTemplate",
    "OCRMatchResult",
    "PaddleOCRNotInstalledError",
    "clear_ocr_cache",
]
