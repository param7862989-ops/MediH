"""Data-mode guards that keep research mode from touching the synthetic demo pipeline.

In DATA_MODE=mimic the application must never seed synthetic patients, train the synthetic
models, or serve synthetic predictions as if they were real. Until serving integration for
research data exists, those entry points refuse to run instead of silently falling back.
"""

from src.utils.config import Config


class ResearchModeNotSupportedError(RuntimeError):
    """Raised when a synthetic-only operation is attempted while DATA_MODE=mimic."""


RESEARCH_MODE_HELP = (
    "Research mode (DATA_MODE=mimic) never seeds synthetic patients, trains the synthetic models, "
    "or serves synthetic predictions, and the web application cannot run on research data until "
    "serving integration is implemented. To build the MIMIC-IV research dataset run: "
    "python -m src.tasks.build_mimic_t1 --mimic-dir <MIMIC-IV demo folder> --output-dir <output folder>. "
    "To run the synthetic demo, unset DATA_MODE or set DATA_MODE=synthetic."
)


def is_research_mode() -> bool:
    return Config.DATA_MODE == "mimic"


def require_synthetic_mode(action: str) -> None:
    """Raises ResearchModeNotSupportedError if a synthetic-only action runs in research mode."""
    if is_research_mode():
        raise ResearchModeNotSupportedError(f"{action} is disabled because DATA_MODE=mimic. {RESEARCH_MODE_HELP}")
