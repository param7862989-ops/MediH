"""Checksums of the persistent (non-test) database and artifact directories.

Used by conftest.py to prove a test run leaves them untouched. Deliberately imports nothing
from src/ so it can run before the test configuration is applied.
"""

import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def persistent_database_paths() -> List[Path]:
    """The demo database the app would use outside tests (.env value or default), plus WAL files."""
    configured = dotenv_values(PROJECT_ROOT / ".env").get("DATABASE_PATH") or "database/medihaven.db"
    databases = {PROJECT_ROOT / "database" / "medihaven.db", _resolve(configured)}
    return sorted(p.with_name(p.name + suffix) for p in databases for suffix in ("", "-wal", "-shm"))


def persistent_artifact_dirs() -> List[Path]:
    """Model and processed-data directories the app would use outside tests."""
    env = dotenv_values(PROJECT_ROOT / ".env")
    return sorted({
        PROJECT_ROOT / "models_store",
        PROJECT_ROOT / "data" / "processed",
        _resolve(env.get("MODELS_STORE_DIR") or "models_store"),
        _resolve(env.get("PROCESSED_DATA_DIR") or "data/processed"),
    })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_persistent_artifacts() -> Dict[str, Optional[str]]:
    """Maps every persistent file (or expected file) to its SHA-256, or None if absent."""
    snapshot: Dict[str, Optional[str]] = {}
    for db_file in persistent_database_paths():
        snapshot[str(db_file)] = _sha256(db_file) if db_file.is_file() else None
    for directory in persistent_artifact_dirs():
        snapshot[f"{directory}{os.sep}"] = "present" if directory.is_dir() else None
        if directory.is_dir():
            for file in sorted(p for p in directory.rglob("*") if p.is_file()):
                snapshot[str(file)] = _sha256(file)
    return snapshot


def describe_changes(before: Dict[str, Optional[str]], after: Dict[str, Optional[str]]) -> List[str]:
    """Lists files that were created, deleted or modified between two snapshots."""
    changes = []
    for path in sorted(set(before) | set(after)):
        old, new = before.get(path), after.get(path)
        if old != new:
            state = "created" if old is None else "deleted" if new is None else "modified"
            changes.append(f"{state}: {path}")
    return changes
