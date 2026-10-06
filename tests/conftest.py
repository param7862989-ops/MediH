"""Shared PyTest fixtures and environment configuration for MediHaven tests."""

import pytest
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Config


@pytest.fixture(scope="session")
def project_config():
    """Provides access to the master configuration object."""
    return Config
