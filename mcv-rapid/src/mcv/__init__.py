"""MCV RapidOCR backend.

This module provides RapidOCR-based text recognition for the MCV framework.

Example:
    >>> from mcv.rapid import RapidOCRTemplate
    >>> template = RapidOCRTemplate(pattern="Hello")
    >>> result = template.find(image)
"""

from .rapid import (
    RapidOCRMatchResult,
    RapidOCRNotInstalledError,
    RapidOCRTemplate,
    clear_ocr_cache,
)

__all__ = [
    "RapidOCRTemplate",
    "RapidOCRMatchResult",
    "RapidOCRNotInstalledError",
    "clear_ocr_cache",
]
