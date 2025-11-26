"""MCV multi-point color matching module.

Exports multi-point color template matching classes.
"""

from pkgutil import extend_path

from .color import MultiColorTemplate

__path__ = extend_path(__path__, __name__)
__all__ = ["MultiColorTemplate"]
