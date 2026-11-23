"""Tests for automatic chart recommendation."""

from __future__ import annotations

import pandas as pd

from app.visualization import build_figure, compatible_chart_types, recommend_chart, spec_for_type


def test_single_value_is_kpi():
    spec = recommend_chart(pd.DataFrame({"total_revenue": [12345.6]}))
    assert spec.chart_type == "kpi" and spec.y == ["total_revenue"]
    assert build_figure(pd.DataFrame({"total_revenue": [1.0]}), spec) is None


def test_single_row_multiple_metrics_is_kpi():
    df = pd.DataFrame({"orders": [10], "revenue": [99.5], "aov": [9.95]})
    assert recommend_chart(df).chart_type == "kpi"


def test_time_series_is_line():
    df = pd.DataFrame({"month": pd.date_range("2025-01-01", periods=12, freq="MS"), "revenue": range(12)})
    spec = recommend_chart(df)
    assert spec.chart_type == "line" and spec.x == "month" and spec.y == ["revenue"]
    assert build_figure(df, spec) is not None


def test_string_dates_detected_as_time():
    df = pd.DataFrame({"day": ["2025-01-01", "2025-01-02", "2025-01-03"], "orders": [3, 4, 5]})
    assert recommend_chart(df).chart_type == "line"


def test_year_over_year_month_number_with_year_series():
    df = pd.DataFrame({"month_num": list(range(1, 13)) * 2, "year": [2024] * 12 + [2025] * 12, "revenue": range(24)})
    spec = recommend_chart(df)
    assert spec.chart_type == "line" and spec.x == "month_num" and spec.color == "year"
    assert build_figure(df, spec) is not None


def test_wide_year_comparison_has_two_lines():
    df = pd.DataFrame({"month_num": range(1, 13), "revenue_2024": range(12), "revenue_2025": range(1, 13)})
    spec = recommend_chart(df)
    assert spec.chart_type == "line" and spec.y == ["revenue_2024", "revenue_2025"]


def test_category_comparison_is_bar():
    df = pd.DataFrame({"category": ["Books", "Toys", "Beauty"], "revenue": [300.0, 200.0, 100.0]})
    spec = recommend_chart(df, "Revenue by category")
    assert spec.chart_type == "bar" and spec.x == "category"
    assert build_figure(df, spec) is not None


def test_many_or_long_categories_use_horizontal_bar():
    df = pd.DataFrame({"customer_name": [f"Customer number {i}" for i in range(15)], "revenue": range(15)})
    spec = recommend_chart(df)
    assert spec.chart_type == "bar" and spec.horizontal


def test_share_question_is_pie():
    df = pd.DataFrame(
        {"sales_channel": ["web", "mobile", "store"], "orders": [50, 30, 20], "pct_of_orders": [50.0, 30.0, 20.0]}
    )
    spec = recommend_chart(df, "What share of orders comes from each channel?")
    assert spec.chart_type == "pie" and spec.y == ["orders"]
    assert build_figure(df, spec) is not None


def test_share_question_with_too_many_slices_is_bar():
    df = pd.DataFrame({"state": [f"S{i}" for i in range(12)], "revenue": range(1, 13)})
    assert recommend_chart(df, "revenue share by state").chart_type == "bar"


def test_negative_values_never_pie():
    df = pd.DataFrame({"segment": ["a", "b"], "profit": [10, -5]})
    assert recommend_chart(df, "profit distribution by segment").chart_type == "bar"


def test_mixed_scale_measures_keep_primary_money_measure():
    df = pd.DataFrame({"state": ["CA", "TX", "NY"], "customers": [60, 50, 40], "revenue": [300000.0, 250000.0, 1.0]})
    assert recommend_chart(df).y == ["revenue"]


def test_ratio_columns_not_mixed_with_amounts():
    df = pd.DataFrame(
        {
            "category": ["a", "b"],
            "revenue": [100.0, 90.0],
            "gross_profit": [40.0, 30.0],
            "gross_margin_pct": [40.0, 33.3],
        }
    )
    assert recommend_chart(df).y == ["revenue", "gross_profit"]


def test_two_category_columns_use_color():
    df = pd.DataFrame({"segment": ["A", "A", "B", "B"], "channel": ["web", "store"] * 2, "revenue": [1, 2, 3, 4]})
    spec = recommend_chart(df)
    assert spec.chart_type == "bar" and spec.color == "channel"
    assert build_figure(df, spec) is not None


def test_two_numeric_columns_scatter():
    df = pd.DataFrame({"unit_price": [1.0, 2.0, 3.0, 4.0], "units_sold": [10, 8, 5, 2]})
    assert recommend_chart(df).chart_type == "scatter"


def test_id_columns_are_not_measures():
    df = pd.DataFrame({"customer_id": [1, 2], "customer_name": ["a", "b"], "email": ["x", "y"]})
    assert recommend_chart(df).chart_type == "table"


def test_empty_result_is_table():
    assert recommend_chart(pd.DataFrame()).chart_type == "table"


def test_too_many_categories_is_table():
    df = pd.DataFrame({"product": [f"p{i}" for i in range(80)], "revenue": range(80)})
    assert recommend_chart(df).chart_type == "table"


def test_user_can_switch_chart_type():
    df = pd.DataFrame({"category": ["a", "b", "c"], "revenue": [3.0, 2.0, 1.0]})
    base = recommend_chart(df)
    assert {"bar", "line", "pie", "table"} <= set(compatible_chart_types(df))
    pie = spec_for_type(df, "pie", base)
    assert pie.chart_type == "pie" and build_figure(df, pie) is not None
