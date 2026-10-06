"""CLI script to initialize or reset the MediHaven database schema.

Usage:
    python database/init_db.py
    python database/init_db.py --force
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.db import init_db
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("medihaven.init_db")


def main():
    parser = argparse.ArgumentParser(description="Initialize MediHaven database schema.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate the database from scratch (WARNING: drops existing data).",
    )
    args = parser.parse_args()

    print(f"Initializing MediHaven Database at: {Config.DATABASE_PATH}")
    if args.force:
        print("Note: --force flag detected. Any existing database will be recreated.")

    try:
        init_db(force_recreate=args.force)
        print("Success: Database schema created and verified.")
    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
