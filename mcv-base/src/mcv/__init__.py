"""MCV core module.

Exports core data structures and abstract base classes.
"""

from pkgutil import extend_path

from .base import MatchResult, ROI, ROILike, Template
from .scaling import ResolutionMapper

__path__ = extend_path(__path__, __name__)
__all__ = ["ROI", "ROILike", "MatchResult", "Template", "ResolutionMapper"]
