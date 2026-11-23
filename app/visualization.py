"""Automatic chart recommendation and Plotly figure building.

The recommendation is rule-based (deterministic and unit-tested), driven by
the shape and types of the result:

* one row of 1-4 numbers                  -> KPI cards
* time column + numeric column(s)         -> line chart (one line per series/category)
* category + numeric, "share"-style question, few positive values -> pie chart
* category + numeric                      -> bar chart (horizontal when many/long labels)
* two numeric columns only                -> scatter plot
* anything else                           -> table only
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_TYPES = ("kpi", "line", "bar", "pie", "scatter", "table")

PROPORTION_WORDS = re.compile(
    r"\b(share|shares|proportion|percentage|percent|pct|breakdown|split|distribution|mix|composition|ratio)\b|%",
    re.IGNORECASE,
)
TIME_NAME = re.compile(r"(date|time|day|week|month|quarter|year|period|_at$|^dt$)", re.IGNORECASE)
TIME_GRAIN_ORDER = ["date", "day", "week", "month", "quarter", "year"]  # finest first
MONEY_NAME = re.compile(r"(revenue|sales|amount|total|profit|value|spend|income|gmv)", re.IGNORECASE)
RATIO_NAME = re.compile(r"(pct|percent|share|ratio|rate|margin)", re.IGNORECASE)
ID_NAME = re.compile(r"(^id$|_id$|^id_)", re.IGNORECASE)
MAX_PIE_SLICES = 8
MAX_BAR_CATEGORIES = 50
MAX_COLOR_GROUPS = 12


@dataclass
class ChartSpec:
    chart_type: str
    x: str | None = None
    y: list[str] = field(default_factory=list)
    color: str | None = None
    title: str = ""
    reason: str = ""
    horizontal: bool = False

    @property
    def is_chart(self) -> bool:
        return self.chart_type not in ("table", "kpi")


def recommend_chart(df: pd.DataFrame, question: str = "") -> ChartSpec:
    if df is None or df.empty:
        return ChartSpec("table", reason="No rows to visualise.")

    numeric = [c for c in df.columns if _is_numeric(df[c]) and not ID_NAME.search(str(c))]
    temporal = _temporal_columns(df)
    numeric_measures = [c for c in numeric if c not in temporal]
    categorical = [c for c in df.columns if c not in numeric and c not in temporal and not ID_NAME.search(str(c))]

    # KPI cards
    if len(df) == 1 and 1 <= len(numeric_measures) <= 4 and len(df.columns) <= 5:
        return ChartSpec("kpi", y=numeric_measures, reason="Single-row result with numeric values.")

    # Time series
    if temporal and numeric_measures and len(df) > 1:
        x = temporal[0]
        color = None
        # remaining low-cardinality temporal (e.g. 'year' when x is 'month_num') or categorical -> series
        for candidate in temporal[1:] + categorical:
            if 1 < df[candidate].nunique() <= MAX_COLOR_GROUPS:
                color = candidate
                break
        y = _comparable_measures(df, numeric_measures)
        if color is not None:
            y = y[:1]
        return ChartSpec(
            "line",
            x=x,
            y=y[:5],
            color=color,
            title=_title(y[:5], x, color),
            reason=f"'{x}' is a time dimension, so a line chart shows the trend.",
        )

    # Categorical comparisons
    if categorical and numeric_measures:
        x = categorical[0]
        n_categories = df[x].nunique()
        if n_categories > MAX_BAR_CATEGORIES:
            return ChartSpec("table", reason=f"Too many categories ({n_categories}) to chart clearly.")
        y = _comparable_measures(df, numeric_measures)
        color = categorical[1] if len(categorical) > 1 and df[categorical[1]].nunique() <= MAX_COLOR_GROUPS else None
        values = df[y[0]]
        if (
            PROPORTION_WORDS.search(question or "")
            and color is None
            and 2 <= n_categories <= MAX_PIE_SLICES
            and (values.dropna() >= 0).all()
            and len(df) == n_categories
        ):
            measure = _pick_pie_measure(y)
            return ChartSpec(
                "pie",
                x=x,
                y=[measure],
                title=f"{_label(measure)} by {_label(x)}",
                reason="The question asks about proportions across a few categories.",
            )
        longest_label = df[x].astype(str).str.len().max()
        return ChartSpec(
            "bar",
            x=x,
            y=y[:1] if color else y[:3],
            color=color,
            horizontal=n_categories > 12 or longest_label > 18,
            title=_title(y[:1] if color else y[:3], x, color),
            reason=f"Comparing values across '{x}' categories.",
        )

    # Numeric relationships
    if len(numeric_measures) >= 2 and len(df) > 2:
        return ChartSpec(
            "scatter",
            x=numeric_measures[0],
            y=[numeric_measures[1]],
            title=f"{_label(numeric_measures[1])} vs {_label(numeric_measures[0])}",
            reason="Two numeric measures: a scatter plot shows their relationship.",
        )
    return ChartSpec("table", reason="No chart fits this result shape; showing the table.")


def compatible_chart_types(df: pd.DataFrame) -> list[str]:
    """Chart types the user may switch to in the UI for this result."""
    if df is None or df.empty:
        return ["table"]
    numeric = [c for c in df.columns if _is_numeric(df[c])]
    options = ["table"]
    if len(df) == 1 and numeric:
        options.append("kpi")
    if numeric and len(df.columns) >= 2:
        options += ["bar", "line"]
        if len(df) <= 20:
            options.append("pie")
        if len(numeric) >= 2:
            options.append("scatter")
    return options


def spec_for_type(df: pd.DataFrame, chart_type: str, base: ChartSpec) -> ChartSpec:
    """Adapt a recommended spec to a chart type the user picked."""
    if chart_type == base.chart_type:
        return base
    numeric = [c for c in df.columns if _is_numeric(df[c]) and not ID_NAME.search(str(c))]
    others = [c for c in df.columns if c not in numeric]
    x = base.x if base.x in df.columns else (others[0] if others else df.columns[0])
    y = [c for c in (base.y or numeric) if c in numeric and c != x] or [c for c in numeric if c != x][:1]
    if chart_type == "kpi":
        return ChartSpec("kpi", y=numeric[:4])
    if chart_type == "scatter" and len(numeric) >= 2:
        return ChartSpec("scatter", x=numeric[0], y=[numeric[1]], title=f"{_label(numeric[1])} vs {_label(numeric[0])}")
    if chart_type == "pie":
        return ChartSpec("pie", x=x, y=y[:1], title=f"{_label(y[0]) if y else ''} by {_label(x)}")
    return ChartSpec(
        chart_type,
        x=x,
        y=y[:3],
        color=base.color if chart_type in ("bar", "line") else None,
        title=base.title or _title(y[:3], x, None),
    )


def build_figure(df: pd.DataFrame, spec: ChartSpec) -> go.Figure | None:
    """Build a Plotly figure for ``spec`` (None for table / KPI)."""
    if not spec.is_chart or df is None or df.empty or not spec.y:
        return None
    try:
        if spec.chart_type == "line":
            data = df.sort_values(spec.x) if spec.x in df.columns else df
            if spec.color:
                fig = px.line(data, x=spec.x, y=spec.y[0], color=data[spec.color].astype(str), markers=True)
                fig.update_layout(legend_title_text=_label(spec.color))
                fig.update_yaxes(title_text=_label(spec.y[0]))
            else:
                long = data.melt(id_vars=[spec.x], value_vars=spec.y, var_name="series", value_name="value")
                long["series"] = long["series"].map(_label)
                fig = px.line(long, x=spec.x, y="value", color="series" if len(spec.y) > 1 else None, markers=True)
                fig.update_yaxes(title_text=_label(spec.y[0]) if len(spec.y) == 1 else "")
                fig.update_layout(legend_title_text="")
        elif spec.chart_type == "bar":
            if spec.color:
                data = df.copy()
                data[spec.color] = data[spec.color].astype(str)
                data[spec.x] = data[spec.x].astype(str)
                # If every bar has exactly one colour value (e.g. product -> category), colour is an
                # attribute: draw full-width bars. Otherwise colour is a series: group the bars.
                one_per_bar = data.groupby(spec.x)[spec.color].nunique().max() <= 1
                barmode = "relative" if one_per_bar else "group"
                if spec.horizontal:
                    fig = px.bar(data, y=spec.x, x=spec.y[0], color=spec.color, barmode=barmode, orientation="h")
                    fig.update_yaxes(
                        autorange="reversed", categoryorder="array", categoryarray=list(dict.fromkeys(data[spec.x]))
                    )
                else:
                    fig = px.bar(data, x=spec.x, y=spec.y[0], color=spec.color, barmode=barmode)
                    fig.update_xaxes(categoryorder="array", categoryarray=list(dict.fromkeys(data[spec.x])))
                fig.update_layout(legend_title_text=_label(spec.color))
                value_axis = fig.update_xaxes if spec.horizontal else fig.update_yaxes
                value_axis(title_text=_label(spec.y[0]))
                (fig.update_yaxes if spec.horizontal else fig.update_xaxes)(title_text=_label(spec.x))
            else:
                long = df.melt(id_vars=[spec.x], value_vars=spec.y, var_name="series", value_name="value")
                long[spec.x] = long[spec.x].astype(str)
                long["series"] = long["series"].map(_label)
                color = "series" if len(spec.y) > 1 else None
                if spec.horizontal:
                    fig = px.bar(long, y=spec.x, x="value", color=color, barmode="group", orientation="h")
                    fig.update_yaxes(autorange="reversed", title_text=_label(spec.x))
                    fig.update_xaxes(title_text=_label(spec.y[0]) if len(spec.y) == 1 else "")
                else:
                    fig = px.bar(long, x=spec.x, y="value", color=color, barmode="group")
                    fig.update_yaxes(title_text=_label(spec.y[0]) if len(spec.y) == 1 else "")
                    fig.update_xaxes(title_text=_label(spec.x))
                if color:
                    fig.update_layout(legend_title_text="")
        elif spec.chart_type == "pie":
            fig = px.pie(df, names=df[spec.x].astype(str), values=spec.y[0], hole=0.35)
            fig.update_traces(textposition="inside", textinfo="percent+label")
        elif spec.chart_type == "scatter":
            fig = px.scatter(df, x=spec.x, y=spec.y[0])
        else:
            return None
    except Exception:  # noqa: BLE001 - a chart must never break the answer
        return None
    fig.update_layout(
        title=spec.title or None,
        margin=dict(l=10, r=10, t=50 if spec.title else 20, b=10),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if spec.chart_type == "line" and spec.x:
        fig.update_xaxes(title_text=_label(spec.x))
    return fig


# --------------------------------------------------------------------------- helpers
def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _temporal_columns(df: pd.DataFrame) -> list[str]:
    """Columns that represent time, finest grain first."""
    found: list[tuple[int, str]] = []
    for col in df.columns:
        series = df[col]
        name = str(col).lower()
        if pd.api.types.is_datetime64_any_dtype(series):
            found.append((0, col))
            continue
        if not TIME_NAME.search(name) or ID_NAME.search(name):
            continue
        rank = next((i for i, g in enumerate(TIME_GRAIN_ORDER) if g in name), 1)
        if pd.api.types.is_integer_dtype(series) or (_is_numeric(series) and (series.dropna() % 1 == 0).all()):
            values = series.dropna()
            plausible = (
                ("year" in name and values.between(1900, 2200).all())
                or ("month" in name and values.between(1, 12).all())
                or ("quarter" in name and values.between(1, 4).all())
                or ("week" in name and values.between(0, 53).all())
                or ("day" in name and values.between(0, 366).all())
            )
            if plausible:
                found.append((rank, col))
        elif series.dtype == object or pd.api.types.is_string_dtype(series):
            parsed = pd.to_datetime(series.dropna().astype(str).head(20), errors="coerce", format="mixed")
            if parsed.notna().all():
                found.append((rank, col))
    return [c for _, c in sorted(found, key=lambda t: t[0])]


def _comparable_measures(df: pd.DataFrame, measures: list[str]) -> list[str]:
    """Primary measure first, plus other measures on a similar scale (so bars/lines stay readable)."""
    if len(measures) <= 1:
        return list(measures)
    absolute = [m for m in measures if not RATIO_NAME.search(m)] or list(measures)
    primary = next((m for m in absolute if MONEY_NAME.search(m)), absolute[0])
    scale = float(df[primary].abs().max() or 0)
    result = [primary]
    for m in absolute:
        if m == primary:
            continue
        other = float(df[m].abs().max() or 0)
        if scale and other and max(scale, other) / min(scale, other) <= 10:
            result.append(m)
    return result


def _pick_pie_measure(measures: list[str]) -> str:
    for m in measures:
        if not re.search(r"(pct|percent|share|ratio|rate)", m, re.IGNORECASE):
            return m
    return measures[0]


def _label(name: str | None) -> str:
    return str(name or "").replace("_", " ").strip().capitalize()


def _title(y: list[str], x: str | None, color: str | None) -> str:
    measures = ", ".join(_label(c) for c in y)
    title = f"{measures} by {_label(x)}" if x else measures
    if color:
        title += f" and {_label(color)}"
    return title
