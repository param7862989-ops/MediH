
"""Database seeding engine proxy for MediHaven."""
from database.seed_data import seed_database, seed_database as seed_clinical_database

__all__ = ["seed_database", "seed_clinical_database"]
