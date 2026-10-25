"""DuckDB-backed reader for a local MIMIC-IV directory.

Event tables (chartevents, labevents) are scanned in place by DuckDB and filtered to the
cohort's observation windows and mapped item IDs before anything reaches pandas, so the
full tables are never loaded into memory. All columns are read as text and cast explicitly
so type inference cannot silently change values.

DuckDB may spill intermediate rows to disk. Its default spill folder is ".tmp" under the
current working directory (which can be the repository), so the reader always sets an
explicit spill directory that passes the research-path safety check.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd

from src.datasources.mimic.layout import MimicLayout, resolve_layout
from src.utils.research_paths import require_safe_research_path


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _to_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="ISO8601")


class MimicIVReader:
    """Reads MIMIC-IV tables needed by the dataset builders."""

    def __init__(self, root_dir: Path, temp_directory: Optional[Path] = None):
        """temp_directory: DuckDB spill folder; defaults to a private folder in the system temp dir."""
        self.layout: MimicLayout = resolve_layout(root_dir)
        self._owns_temp = temp_directory is None
        if temp_directory is None:
            temp_directory = Path(tempfile.mkdtemp(prefix="medihaven_duckdb_"))
        self.temp_directory = require_safe_research_path(temp_directory, "the DuckDB spill directory")
        self.temp_directory.mkdir(parents=True, exist_ok=True)

        self._con = duckdb.connect(database=":memory:")
        spill = self.temp_directory.as_posix().replace("'", "''")
        self._con.execute(f"SET temp_directory = '{spill}'")
        configured = Path(self._con.execute("SELECT current_setting('temp_directory')").fetchone()[0]).resolve()
        if configured != self.temp_directory:
            self.close()
            raise RuntimeError("DuckDB did not accept the safe spill directory; refusing to read MIMIC-IV data.")

    def close(self) -> None:
        self._con.close()
        if self._owns_temp:
            shutil.rmtree(self.temp_directory, ignore_errors=True)

    def __enter__(self) -> "MimicIVReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _scan(self, table: str) -> str:
        path = self.layout.files[table].as_posix().replace("'", "''")
        return f"read_csv('{path}', header=true, all_varchar=true)"

    def _select(self, table: str, columns: Iterable[str]) -> pd.DataFrame:
        cols = ", ".join(f'"{c}"' for c in columns)
        return self._con.execute(f"SELECT {cols} FROM {self._scan(table)}").df()

    # ------------------------------------------------------------------
    # Small reference tables (loaded whole)
    # ------------------------------------------------------------------
    def load_patients(self) -> pd.DataFrame:
        df = self._select("hosp/patients", ["subject_id", "gender", "anchor_age", "anchor_year", "dod"])
        return pd.DataFrame({
            "subject_id": _to_int(df["subject_id"]),
            "gender": df["gender"].astype("string").str.strip().str.upper(),
            "anchor_age": _to_int(df["anchor_age"]),
            "anchor_year": _to_int(df["anchor_year"]),
            "dod": _to_time(df["dod"]),
        })

    def load_admissions(self) -> pd.DataFrame:
        df = self._select("hosp/admissions", [
            "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
            "discharge_location", "hospital_expire_flag",
        ])
        return pd.DataFrame({
            "subject_id": _to_int(df["subject_id"]),
            "hadm_id": _to_int(df["hadm_id"]),
            "admittime": _to_time(df["admittime"]),
            "dischtime": _to_time(df["dischtime"]),
            "deathtime": _to_time(df["deathtime"]),
            "discharge_location": df["discharge_location"].astype("string").str.strip().str.upper(),
            "hospital_expire_flag": _to_int(df["hospital_expire_flag"]),
        })

    def load_icustays(self) -> pd.DataFrame:
        df = self._select("icu/icustays", ["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime"])
        return pd.DataFrame({
            "subject_id": _to_int(df["subject_id"]),
            "hadm_id": _to_int(df["hadm_id"]),
            "stay_id": _to_int(df["stay_id"]),
            "first_careunit": df["first_careunit"].astype("string"),
            "intime": _to_time(df["intime"]),
            "outtime": _to_time(df["outtime"]),
        })

    def load_d_items(self) -> pd.DataFrame:
        df = self._select("icu/d_items", ["itemid", "label", "linksto"])
        df["itemid"] = _to_int(df["itemid"])
        return df.dropna(subset=["itemid"]).drop_duplicates("itemid")

    def load_d_labitems(self) -> pd.DataFrame:
        df = self._select("hosp/d_labitems", ["itemid", "label", "fluid", "category"])
        df["itemid"] = _to_int(df["itemid"])
        return df.dropna(subset=["itemid"]).drop_duplicates("itemid")

    # ------------------------------------------------------------------
    # Event tables (filtered inside DuckDB)
    # ------------------------------------------------------------------
    def extract_window_events(self, cohort: pd.DataFrame, chart_itemids, lab_itemids) -> pd.DataFrame:
        """Returns mapped measurements charted inside each stay's [window_start, index_time].

        cohort needs stay_id, subject_id, window_start, index_time. The storetime rule is
        applied afterwards in pandas so its effect can be counted.
        """
        columns = ["stay_id", "source_table", "itemid", "charttime", "storetime", "valuenum", "valueuom"]
        if cohort.empty:
            return pd.DataFrame(columns=columns)

        window = cohort[["stay_id", "subject_id", "window_start", "index_time"]].copy()
        window["stay_id"] = window["stay_id"].astype("int64")
        window["subject_id"] = window["subject_id"].astype("int64")
        self._con.register("cohort_window", window)
        try:
            parts = []
            if chart_itemids:
                ids = ", ".join(str(int(i)) for i in sorted(chart_itemids))
                parts.append(f"""
                    SELECT w.stay_id, 'chartevents' AS source_table, e.itemid, e.charttime, e.storetime,
                           e.valuenum, e.valueuom
                    FROM (
                        SELECT TRY_CAST(stay_id AS BIGINT) AS stay_id, TRY_CAST(itemid AS INTEGER) AS itemid,
                               TRY_CAST(charttime AS TIMESTAMP) AS charttime, TRY_CAST(storetime AS TIMESTAMP) AS storetime,
                               TRY_CAST(valuenum AS DOUBLE) AS valuenum, valueuom
                        FROM {self._scan('icu/chartevents')}
                    ) e
                    JOIN cohort_window w ON e.stay_id = w.stay_id
                    WHERE e.itemid IN ({ids}) AND e.charttime >= w.window_start AND e.charttime <= w.index_time
                """)
            if lab_itemids:
                ids = ", ".join(str(int(i)) for i in sorted(lab_itemids))
                # labevents has no stay_id and often no hadm_id: attach by person and time window
                parts.append(f"""
                    SELECT w.stay_id, 'labevents' AS source_table, e.itemid, e.charttime, e.storetime,
                           e.valuenum, e.valueuom
                    FROM (
                        SELECT TRY_CAST(subject_id AS BIGINT) AS subject_id, TRY_CAST(itemid AS INTEGER) AS itemid,
                               TRY_CAST(charttime AS TIMESTAMP) AS charttime, TRY_CAST(storetime AS TIMESTAMP) AS storetime,
                               TRY_CAST(valuenum AS DOUBLE) AS valuenum, valueuom
                        FROM {self._scan('hosp/labevents')}
                    ) e
                    JOIN cohort_window w ON e.subject_id = w.subject_id
                    WHERE e.itemid IN ({ids}) AND e.charttime >= w.window_start AND e.charttime <= w.index_time
                """)
            if not parts:
                return pd.DataFrame(columns=columns)
            events = self._con.execute(" UNION ALL ".join(parts)).df()
        finally:
            self._con.unregister("cohort_window")

        return pd.DataFrame({
            "stay_id": _to_int(events["stay_id"]),
            "source_table": events["source_table"].astype("string"),
            "itemid": _to_int(events["itemid"]),
            "charttime": pd.to_datetime(events["charttime"]),
            "storetime": pd.to_datetime(events["storetime"]),
            "valuenum": pd.to_numeric(events["valuenum"], errors="coerce"),
            "valueuom": events["valueuom"].astype("string"),
        })
