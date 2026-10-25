"""MIMIC-IV adapter: local CSV/CSV.GZ layout validation, dictionary-checked item mappings,
and DuckDB extraction filtered to cohort observation windows."""

from src.datasources.mimic.layout import MimicDataError, MimicLayout, resolve_layout
from src.datasources.mimic.mappings import FEATURE_SPEC_VERSION, MappingError
from src.datasources.mimic.reader import MimicIVReader

__all__ = [
    "FEATURE_SPEC_VERSION",
    "MappingError",
    "MimicDataError",
    "MimicIVReader",
    "MimicLayout",
    "resolve_layout",
]
