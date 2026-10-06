"""Clinical Data Ingestion and Database Extraction Layer for MediHaven.

Provides robust, parameterized extraction of clinical records from the relational
database into structured Pandas DataFrames. Supports full cohort extraction for batch
model training as well as single-patient extraction for real-time REST API inference.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import pandas as pd
import sqlite3

from src.database.db import get_connection, get_db_connection
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.data.loader")


def load_patients(
    patient_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """Loads patient demographic and admission records.

    Args:
        patient_id: Optional patient ID to filter by.
        conn: Optional active database connection.

    Returns:
        DataFrame containing patient records.
    """
    query = """
        SELECT patient_id, mrn, full_name, age, gender, 
               admission_date, discharge_date, status, ward, bed_number
        FROM patients
    """
    params = []
    if patient_id is not None:
        query += " WHERE patient_id = ?"
        params.append(patient_id)

    query += " ORDER BY patient_id ASC"

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        df = pd.read_sql_query(
            sql=query,
            con=conn,
            params=params,
            parse_dates=["admission_date", "discharge_date"],
        )
        return df
    finally:
        if should_close and conn:
            conn.close()


def load_vitals(
    patient_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """Loads longitudinal vitals time series.

    Args:
        patient_id: Optional patient ID to filter by.
        conn: Optional active database connection.

    Returns:
        DataFrame containing chronological vitals readings.
    """
    query = """
        SELECT vital_id, patient_id, recorded_at, heart_rate, 
               blood_pressure_sys, blood_pressure_dia, temperature, 
               oxygen_saturation, respiratory_rate
        FROM vitals
    """
    params = []
    if patient_id is not None:
        query += " WHERE patient_id = ?"
        params.append(patient_id)

    query += " ORDER BY patient_id ASC, recorded_at ASC"

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        df = pd.read_sql_query(
            sql=query,
            con=conn,
            params=params,
            parse_dates=["recorded_at"],
        )
        return df
    finally:
        if should_close and conn:
            conn.close()


def load_lab_results(
    patient_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """Loads laboratory biomarker results.

    Args:
        patient_id: Optional patient ID to filter by.
        conn: Optional active database connection.

    Returns:
        DataFrame containing laboratory test records.
    """
    query = """
        SELECT lab_id, patient_id, recorded_at, glucose, creatinine, 
               cholesterol, wbc_count, bun, hemoglobin, platelets
        FROM lab_results
    """
    params = []
    if patient_id is not None:
        query += " WHERE patient_id = ?"
        params.append(patient_id)

    query += " ORDER BY patient_id ASC, recorded_at ASC"

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        df = pd.read_sql_query(
            sql=query,
            con=conn,
            params=params,
            parse_dates=["recorded_at"],
        )
        return df
    finally:
        if should_close and conn:
            conn.close()


def load_medications(
    patient_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """Loads active and past medication orders.

    Args:
        patient_id: Optional patient ID to filter by.
        conn: Optional active database connection.

    Returns:
        DataFrame containing medication records.
    """
    query = """
        SELECT medication_id, patient_id, drug_name, dosage, 
               route, frequency, start_date, end_date, is_active
        FROM medications
    """
    params = []
    if patient_id is not None:
        query += " WHERE patient_id = ?"
        params.append(patient_id)

    query += " ORDER BY patient_id ASC, start_date ASC"

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        df = pd.read_sql_query(
            sql=query,
            con=conn,
            params=params,
            parse_dates=["start_date", "end_date"],
        )
        return df
    finally:
        if should_close and conn:
            conn.close()


def load_outcomes(
    patient_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """Loads clinical outcomes and ground-truth diagnosis data.

    Args:
        patient_id: Optional patient ID to filter by.
        conn: Optional active database connection.

    Returns:
        DataFrame containing patient outcomes and discharge flags.
    """
    query = """
        SELECT outcome_id, patient_id, primary_diagnosis, 
               readmitted_30d, icu_transfer, outcome_date
        FROM outcomes
    """
    params = []
    if patient_id is not None:
        query += " WHERE patient_id = ?"
        params.append(patient_id)

    query += " ORDER BY patient_id ASC"

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        df = pd.read_sql_query(
            sql=query,
            con=conn,
            params=params,
            parse_dates=["outcome_date"],
        )
        return df
    finally:
        if should_close and conn:
            conn.close()


def load_predictions(
    patient_id: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> pd.DataFrame:
    """Loads historical model predictions for audit and validation.

    Args:
        patient_id: Optional patient ID to filter by.
        conn: Optional active database connection.

    Returns:
        DataFrame containing logged predictions.
    """
    query = """
        SELECT prediction_id, patient_id, predicted_at, risk_tier, 
               risk_score, model_used, decision_tree_path, 
               similar_patient_ids, recommended_protocol, readmission_30d_risk
        FROM predictions
    """
    params = []
    if patient_id is not None:
        query += " WHERE patient_id = ?"
        params.append(patient_id)

    query += " ORDER BY patient_id ASC, predicted_at DESC"

    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        df = pd.read_sql_query(
            sql=query,
            con=conn,
            params=params,
            parse_dates=["predicted_at"],
        )
        return df
    finally:
        if should_close and conn:
            conn.close()


def load_raw_multimodal_dataset(
    patient_id: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Ingests all clinical tables into a consolidated dictionary of DataFrames.

    Args:
        patient_id: Optional filter for a specific patient.

    Returns:
        Dictionary mapping table name to DataFrame:
        {'patients', 'vitals', 'labs', 'medications', 'outcomes', 'predictions'}
    """
    logger.info(
        f"Loading multimodal dataset"
        + (f" for patient_id={patient_id}" if patient_id else " for all cohorts")
    )

    with get_connection() as conn:
        dataset = {
            "patients": load_patients(patient_id, conn=conn),
            "vitals": load_vitals(patient_id, conn=conn),
            "labs": load_lab_results(patient_id, conn=conn),
            "medications": load_medications(patient_id, conn=conn),
            "outcomes": load_outcomes(patient_id, conn=conn),
            "predictions": load_predictions(patient_id, conn=conn),
        }

    logger.info(
        f"Dataset loaded: {len(dataset['patients'])} patients, "
        f"{len(dataset['vitals'])} vitals readings, "
        f"{len(dataset['labs'])} lab results, "
        f"{len(dataset['medications'])} medications, "
        f"{len(dataset['outcomes'])} outcomes."
    )
    return dataset
