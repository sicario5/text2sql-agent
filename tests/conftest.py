from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from app.sql_validator import SQLValidator  # noqa: E402
from fakes import APPROVED, make_settings, make_vanna, sample_schema_info  # noqa: E402


@pytest.fixture
def validator() -> SQLValidator:
    return SQLValidator(APPROVED, default_schema="public", max_rows=1000)


@pytest.fixture
def schema_info():
    return sample_schema_info()


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def vn(tmp_path):
    return make_vanna(tmp_path)


@pytest.fixture
def knowledge_dir() -> Path:
    return ROOT / "knowledge"


# ---------------------------------------------------------------- integration (real PostgreSQL)
def _test_db_url() -> str | None:
    return os.getenv("TEST_DATABASE_URL")


@pytest.fixture(scope="session")
def db_url() -> str:
    url = _test_db_url()
    if not url:
        pytest.skip("TEST_DATABASE_URL not set (integration tests need the sample PostgreSQL database)")
    return url


@pytest.fixture(scope="session")
def engine(db_url):
    from app.db import check_connection, create_db_engine

    eng = create_db_engine(db_url, query_timeout_seconds=10)
    status = check_connection(eng)
    if not status.ok:
        pytest.skip(f"Test database unreachable: {status.error}")
    yield eng
    eng.dispose()
