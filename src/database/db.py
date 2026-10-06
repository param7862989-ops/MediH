"""Database Access Layer and Connection Manager for MediHaven.

Provides thread-safe SQLite connection handling, Write-Ahead Logging (WAL) mode,
automatic foreign-key enforcement, transaction context managers, and dictionary-like
row serialization.
"""

import sqlite3
import datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.db")

# Register clean ISO-8601 datetime adapters for modern Python (3.12+)
sqlite3.register_adapter(datetime.datetime, lambda dt: dt.isoformat())
sqlite3.register_adapter(datetime.date, lambda d: d.isoformat())


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Creates and configures a SQLite connection.

    Args:
        db_path: Path to database file. Defaults to Config.DATABASE_PATH.

    Returns:
        sqlite3.Connection configured with WAL mode, foreign keys, and dictionary rows.
    """
    target_path = db_path or Config.DATABASE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        database=str(target_path),
        timeout=30.0,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        check_same_thread=False,
    )

    # Enable SQLite Pragmas for concurrency and integrity
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    # Enable dictionary-like row access
    conn.row_factory = sqlite3.Row

    return conn


@contextmanager
def get_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for acquiring and safely closing a database connection.

    Yields:
        sqlite3.Connection instance.
    """
    conn = get_db_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction(db_path: Optional[Path] = None) -> Generator[sqlite3.Cursor, None, None]:
    """Context manager for atomic database transactions with automatic commit and rollback.

    Yields:
        sqlite3.Cursor instance within an active transaction.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction failed and was rolled back: {e}", exc_info=True)
        raise
    finally:
        conn.close()


def init_db(force_recreate: bool = False, db_path: Optional[Path] = None) -> None:
    """Initializes the database schema using database/schema.sql.

    Args:
        force_recreate: If True, deletes existing database before executing schema.
        db_path: Optional override for database path.
    """
    target_path = db_path or Config.DATABASE_PATH
    schema_path = Config.DATABASE_DIR / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")

    if force_recreate and target_path.exists():
        logger.warning(f"force_recreate=True: Removing existing database at {target_path}")
        target_path.unlink()

    logger.info(f"Applying schema from {schema_path} to {target_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with get_connection(target_path) as conn:
        conn.executescript(schema_sql)

    logger.info("Database schema initialized successfully.")


def execute_query(
    sql: str, params: Optional[tuple] = None, db_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Executes a SELECT query and returns rows as a list of Python dictionaries.

    Args:
        sql: SQL query string.
        params: Tuple of query parameters.
        db_path: Optional override for database path.

    Returns:
        List of row dictionaries.
    """
    params = params or ()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def execute_write(
    sql: str, params: Optional[tuple] = None, db_path: Optional[Path] = None
) -> int:
    """Executes an INSERT, UPDATE, or DELETE query atomically.

    Args:
        sql: SQL statement string.
        params: Tuple of parameters.
        db_path: Optional database path.

    Returns:
        The last inserted row ID.
    """
    params = params or ()
    with transaction(db_path) as cursor:
        cursor.execute(sql, params)
        return cursor.lastrowid
