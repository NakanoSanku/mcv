"""MCV image template matching module.

Exports image template matching classes.
"""

from pkgutil import extend_path

from .image import ImageTemplate

__path__ = extend_path(__path__, __name__)
__all__ = ["ImageTemplate"]
