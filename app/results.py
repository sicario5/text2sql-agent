"""Result processing: DataFrame normalisation and natural-language summaries."""

from __future__ import annotations

import datetime as dt
import decimal
import logging
import uuid
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .executor import QueryResult

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM = """You are a precise data analyst. Answer the user's question in plain English using ONLY the query result provided.
- Lead with the direct answer. Use 1-4 short sentences, or a short bullet list for rankings.
- Quote numbers exactly as they appear in the data, formatted with thousands separators (e.g. 12,345.67).
- Do not invent numbers, causes or trends that the data does not show.
- If the result is empty, say that no matching data was found.
- Do not describe the SQL."""

MAX_SUMMARY_CHARS = 8000


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert driver types into pandas-friendly types for display and charting.

    * Decimal  -> float  (NUMERIC columns arrive as Decimal objects)
    * date     -> datetime64 (so charts get a real time axis)
    * UUID / dict / list (json) -> str
    """
    df = df.copy()
    for col in df.columns:
        series = df[col]
        if series.dtype != object:
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        sample = non_null.iloc[:200]
        if all(isinstance(v, decimal.Decimal) for v in sample):
            df[col] = pd.to_numeric(series.map(lambda v: float(v) if v is not None else None), errors="coerce")
        elif all(isinstance(v, dt.date) and not isinstance(v, dt.datetime) for v in sample):
            df[col] = pd.to_datetime(series, errors="coerce")
        elif all(isinstance(v, dt.datetime) for v in sample):
            df[col] = pd.to_datetime(series, errors="coerce", utc=any(v.tzinfo for v in sample))
        elif any(isinstance(v, (uuid.UUID, dict, list, bytes, memoryview)) for v in sample):
            df[col] = series.map(lambda v: None if v is None else str(v))
    return df


def summarize_result(vn, question: str, sql: str, result: QueryResult, max_rows: int = 50) -> str:
    """Ask the LLM for a short natural-language answer; falls back to a rule-based summary."""
    if result.row_count == 0:
        return "The query ran successfully but returned no rows."
    sample = result.df.head(max_rows)
    csv = sample.to_csv(index=False, float_format="%.2f")
    if len(csv) > MAX_SUMMARY_CHARS:
        csv = csv[:MAX_SUMMARY_CHARS] + "\n... (truncated)"
    shown = (
        f"first {len(sample)} of {result.row_count}" if result.row_count > len(sample) else f"all {result.row_count}"
    )
    note = " The result was truncated at the row limit." if result.truncated else ""
    user = f"Question: {question}\n\nQuery result ({shown} rows shown).{note}\n```csv\n{csv}\n```"
    try:
        text = vn.submit_prompt([vn.system_message(SUMMARY_SYSTEM), vn.user_message(user)])
        if text and text.strip():
            return text.strip()
    except Exception as exc:  # noqa: BLE001 - never fail the whole answer because of the summary
        logger.warning("Summary generation failed: %s", exc)
    return fallback_summary(result)


def fallback_summary(result: QueryResult) -> str:
    """Deterministic summary used when LLM summaries are disabled or fail."""
    df = result.df
    if result.row_count == 0:
        return "The query ran successfully but returned no rows."
    if result.row_count == 1 and len(df.columns) <= 4:
        parts = [f"{col.replace('_', ' ')}: {format_value(df.iloc[0][col])}" for col in df.columns]
        return "Result — " + "; ".join(parts) + "."
    text = f"Returned {result.row_count:,} rows across {len(df.columns)} columns."
    if result.truncated:
        text += f" Results were limited to the first {result.row_count:,} rows."
    return text


def format_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) or (isinstance(value, float) and float(value).is_integer() and abs(value) < 1e15):
        return f"{int(value):,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else value.strftime("%Y-%m-%d %H:%M")
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return f"{int(value):,}"
        if isinstance(value, np.floating):
            return format_value(float(value))
    except ImportError:  # pragma: no cover
        pass
    return str(value)
