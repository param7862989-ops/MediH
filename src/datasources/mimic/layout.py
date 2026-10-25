"""MIMIC-IV directory layout discovery and schema validation.

Expects the official PhysioNet layout (MIMIC-IV or the MIMIC-IV Clinical Database Demo):

    <MIMIC_IV_DIR>/hosp/patients.csv.gz, admissions.csv.gz, labevents.csv.gz, d_labitems.csv.gz
    <MIMIC_IV_DIR>/icu/icustays.csv.gz, chartevents.csv.gz, d_items.csv.gz

Plain .csv files are accepted too. Only header lines are read here; no data rows.
"""

import csv
import gzip
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


class MimicDataError(RuntimeError):
    """Raised for missing/invalid MIMIC-IV files, with an actionable explanation."""


# Source tables required by the Task 1 builder, with the columns it reads
REQUIRED_TABLES: Dict[str, List[str]] = {
    "hosp/patients": ["subject_id", "gender", "anchor_age", "anchor_year", "dod"],
    "hosp/admissions": [
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
        "discharge_location", "hospital_expire_flag",
    ],
    "hosp/labevents": ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"],
    "hosp/d_labitems": ["itemid", "label", "fluid", "category"],
    "icu/icustays": ["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"],
    "icu/chartevents": [
        "subject_id", "hadm_id", "stay_id", "charttime", "storetime", "itemid", "valuenum", "valueuom",
    ],
    "icu/d_items": ["itemid", "label", "linksto"],
}

# Tables the contract can expose when present, not needed for Task 1
OPTIONAL_TABLES: Dict[str, List[str]] = {
    "hosp/prescriptions": ["subject_id", "hadm_id", "starttime", "stoptime", "drug", "route"],
}

SETUP_HINT = (
    "Download the MIMIC-IV Clinical Database Demo from PhysioNet, extract it to a local folder "
    "outside the repository and outside OneDrive, and point MIMIC_IV_DIR (or --mimic-dir) at the "
    "folder that contains the 'hosp' and 'icu' subfolders."
)


@dataclass
class MimicLayout:
    """Resolved MIMIC-IV source files and release metadata."""

    root: Path
    files: Dict[str, Path] = field(default_factory=dict)
    optional_files: Dict[str, Path] = field(default_factory=dict)
    release: str = "unknown"
    checksum_manifest_sha256: Optional[str] = None

    def describe(self) -> Dict[str, object]:
        """Release and file inventory (names and sizes only) for the dataset manifest."""
        return {
            "release": self.release,
            "root_folder_name": self.root.name,
            "checksum_manifest_sha256": self.checksum_manifest_sha256,
            "files": {
                table: {"file": path.relative_to(self.root).as_posix(), "bytes": path.stat().st_size}
                for table, path in sorted({**self.files, **self.optional_files}.items())
            },
        }


def read_header(path: Path) -> List[str]:
    """Reads only the header row of a CSV or CSV.GZ file."""
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, EOFError, UnicodeDecodeError) as e:
        raise MimicDataError(f"Could not read the header of {path.name}: {type(e).__name__}. Is the file complete?") from e
    return [col.strip() for col in header]


def _find_table_file(root: Path, table: str) -> Optional[Path]:
    # Prefer the official compressed file when both forms exist
    for suffix in (".csv.gz", ".csv"):
        candidate = root / f"{table}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _locate_root(root: Path) -> Path:
    if (root / "hosp").is_dir() and (root / "icu").is_dir():
        return root
    # Accept a parent folder holding exactly one extracted release
    children = [c for c in root.iterdir() if c.is_dir() and (c / "hosp").is_dir() and (c / "icu").is_dir()]
    if len(children) == 1:
        return children[0]
    raise MimicDataError(f"'{root}' does not contain the 'hosp' and 'icu' subfolders. {SETUP_HINT}")


def _detect_release(root: Path) -> str:
    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", root.name)
    if not match:
        return "unknown"
    kind = "mimic-iv-demo" if "demo" in root.name.lower() else "mimic-iv"
    return f"{kind}-{match.group(1)}"


def resolve_layout(root_dir: Path) -> MimicLayout:
    """Finds and validates every required MIMIC-IV table under root_dir."""
    root_dir = Path(root_dir).expanduser()
    if not root_dir.is_dir():
        raise MimicDataError(f"MIMIC-IV directory '{root_dir}' does not exist. {SETUP_HINT}")
    root = _locate_root(root_dir)

    files, missing = {}, []
    for table in REQUIRED_TABLES:
        found = _find_table_file(root, table)
        if found is None:
            missing.append(f"{table}.csv.gz")
        else:
            files[table] = found
    if missing:
        raise MimicDataError(f"Required MIMIC-IV files are missing under '{root}': {', '.join(missing)}. {SETUP_HINT}")

    column_problems = []
    for table, path in files.items():
        header = set(read_header(path))
        absent = [col for col in REQUIRED_TABLES[table] if col not in header]
        if absent:
            column_problems.append(f"{path.name} lacks {absent}")
    if column_problems:
        raise MimicDataError(
            "MIMIC-IV files do not have the expected columns: " + "; ".join(column_problems)
            + ". Check that the files come from an official MIMIC-IV release and are not modified."
        )

    optional_files = {}
    for table, columns in OPTIONAL_TABLES.items():
        found = _find_table_file(root, table)
        if found is not None and all(col in read_header(found) for col in columns):
            optional_files[table] = found

    checksum_file = root / "SHA256SUMS.txt"
    checksum_sha = hashlib.sha256(checksum_file.read_bytes()).hexdigest() if checksum_file.is_file() else None

    return MimicLayout(
        root=root,
        files=files,
        optional_files=optional_files,
        release=_detect_release(root),
        checksum_manifest_sha256=checksum_sha,
    )
