"""MCV PaddleOCR integration module.

This module provides OCR-based text matching using PaddleOCR.
"""

from pkgutil import extend_path

from .paddle import (
    PaddleOCRMatchResult,
    PaddleOCRTemplate,
    PaddleOCRNotInstalledError,
    clear_ocr_cache,
)

__path__ = extend_path(__path__, __name__)
__all__ = [
    "PaddleOCRTemplate",
    "PaddleOCRMatchResult",
    "PaddleOCRNotInstalledError",
    "clear_ocr_cache",
]
