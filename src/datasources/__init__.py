"""Clinical data-source adapters mapping each source onto MediHaven's common data contract.

- contract.py   documented canonical tables (encounters, ICU stays, observations, medications)
- synthetic.py  adapter over the existing synthetic SQLite demo (behavior unchanged)
- mimic/        adapter over a local MIMIC-IV (Demo) CSV / CSV.GZ directory, queried with DuckDB
"""

from src.datasources.contract import (
    CANONICAL_TABLES,
    CANONICAL_UNITS,
    DataContractError,
    validate_table,
)

__all__ = ["CANONICAL_TABLES", "CANONICAL_UNITS", "DataContractError", "validate_table"]
