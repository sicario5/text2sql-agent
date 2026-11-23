"""Headless UI test of the Streamlit app (needs the sample database)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def app_env(db_url, monkeypatch, tmp_path):
    from fake_llm_server import start_server

    server, url = start_server()
    env = {
        "DATABASE_URL": db_url,
        "LLM_PROVIDER": "openai_compatible",
        "LLM_BASE_URL": url,
        "LLM_API_KEY": "fake",
        "LLM_MODEL": "fake",
        "EMBEDDING_PROVIDER": "hashing",
        "VECTOR_DB_PATH": str(tmp_path / "chroma"),
        "AUTO_TRAIN": "true",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    server.shutdown()


def test_app_answers_question_and_follow_up(app_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("Connected to" in s.value for s in at.sidebar.success)
    assert at.sidebar.selectbox[0].value == "public"

    at.chat_input[0].set_value("Show total revenue by month.").run()
    assert not at.exception, at.exception
    assert len(at.dataframe) >= 1
    assert any("SELECT" in c.value for c in at.code)

    at.chat_input[0].set_value("Now only for 2025.").run()
    assert not at.exception, at.exception
    assert any("Interpreted as" in c.value for c in at.caption)


def test_app_shows_blocked_query(app_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.chat_input[0].set_value("Delete all orders").run()
    assert not at.exception, at.exception
    assert any("safety layer" in e.value for e in at.error)
