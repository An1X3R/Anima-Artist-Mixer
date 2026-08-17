import logging

from .anima_mixer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
# TEMP_SEMANTIC_DIAG_HOOK: remove with semantic_diagnostics.py after diagnosis.
from .anima_mixer.semantic_diagnostics import setup_comfy_file_logging


setup_comfy_file_logging()

# TEMP_CONDITIONING_DIAG_HOOK: this optional import is the only installation
# site for conditioning_diagnostics.py. Deleting that module is fail-open;
# removing this block restores the original plugin entrypoint exactly.
try:
    from .anima_mixer.conditioning_diagnostics import (
        install as _install_conditioning_diagnostics,
    )
except Exception as _conditioning_diagnostic_error:
    _install_conditioning_diagnostics = None
    logging.getLogger(__name__).warning(
        "[AnimaConditionDiag] event=optional_import_error error_type=%s error=%s",
        type(_conditioning_diagnostic_error).__name__,
        _conditioning_diagnostic_error,
    )

if _install_conditioning_diagnostics is not None:
    try:
        _install_conditioning_diagnostics()
    except Exception as _conditioning_diagnostic_error:
        logging.getLogger(__name__).warning(
            "[AnimaConditionDiag] event=optional_install_error error_type=%s error=%s",
            type(_conditioning_diagnostic_error).__name__,
            _conditioning_diagnostic_error,
        )

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
