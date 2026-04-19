"""Layout detection module."""

from typing import Optional

from .base import BaseLayoutDetector

_layout_import_error: Optional[BaseException] = None
_layout_import_error_is_dependency: bool = False

try:
    from .layout_detector import PPDocLayoutDetector
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    PPDocLayoutDetector = None  # type: ignore
    _layout_import_error = e
    _layout_import_error_is_dependency = True
except Exception as e:  # pragma: no cover
    # Dependencies may already be installed; importing the detector can still fail
    # due to version incompatibilities or other runtime errors.
    PPDocLayoutDetector = None  # type: ignore
    _layout_import_error = e
    _layout_import_error_is_dependency = False


def _raise_layout_import_error() -> None:
    err_name = type(_layout_import_error).__name__
    err_msg = str(_layout_import_error)

    if _layout_import_error_is_dependency:
        message = (
            f"Layout detector dependencies missing. Try: pip install 'glmocr[layout]'. "
            f"Original error: {err_name}: {err_msg}"
        )
    else:
        message = (
            "Layout detector failed to import. "
            f"Original error: {err_name}: {err_msg}"
        )

    raise ImportError(message) from _layout_import_error


__all__ = ["BaseLayoutDetector", "PPDocLayoutDetector"]
