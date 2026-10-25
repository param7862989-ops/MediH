"""Location safety checks for research data (MIMIC-IV inputs and generated datasets).

Credentialed clinical data, and anything derived from it, must stay out of the Git
repository and out of cloud-synchronized folders such as OneDrive. This module has no
project imports so that the configuration module can use it.
"""

import os
import re
from pathlib import Path
from typing import List, Union

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ONEDRIVE_ENV_VARS = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
# Matches folder names such as "OneDrive" or "OneDrive - Contoso"
ONEDRIVE_FOLDER_PATTERN = re.compile(r"^onedrive(\s*-\s*.+)?$", re.IGNORECASE)

PathLike = Union[str, Path]


class UnsafeResearchPathError(ValueError):
    """Raised when a research input/output path is inside the repository or a synced folder."""


def _normalized(path: PathLike) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def is_within(path: PathLike, parent: PathLike) -> bool:
    """Returns True if path equals parent or is nested inside it (case-insensitive on Windows)."""
    child, root = _normalized(path), _normalized(parent)
    try:
        return os.path.commonpath([child, root]) == root
    except ValueError:  # different drives on Windows
        return False


def onedrive_roots() -> List[Path]:
    """Returns the OneDrive roots announced by the operating system environment."""
    return [Path(os.environ[name]) for name in ONEDRIVE_ENV_VARS if os.environ.get(name, "").strip()]


def unsafe_reasons(path: PathLike, repo_root: PathLike = REPO_ROOT) -> List[str]:
    """Lists why a path is unsafe for research data; an empty list means it is acceptable."""
    reasons = []
    if is_within(path, repo_root):
        reasons.append(f"it is inside the Git repository ({Path(repo_root)})")

    synced_root = next((root for root in onedrive_roots() if is_within(path, root)), None)
    if synced_root is not None:
        reasons.append(f"it is inside the OneDrive-synchronized folder {synced_root}")
    elif any(ONEDRIVE_FOLDER_PATTERN.match(part) for part in Path(path).expanduser().resolve().parts):
        reasons.append("its path contains a OneDrive folder")
    return reasons


def require_safe_research_path(path: PathLike, purpose: str) -> Path:
    """Returns the resolved path, or raises UnsafeResearchPathError if it is not safe to use."""
    resolved = Path(path).expanduser().resolve()
    reasons = unsafe_reasons(resolved)
    if reasons:
        raise UnsafeResearchPathError(
            f"Refusing to use {purpose} at '{resolved}' because " + " and ".join(reasons) + ". "
            "Keep MIMIC-IV source files and generated research datasets in a local folder outside "
            "the repository and outside OneDrive (for example C:\\research\\mimic-iv-demo and "
            "C:\\research\\medihaven-output)."
        )
    return resolved
