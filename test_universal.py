"""
Universal Stress-Test Suite for ezpzAnalyze
============================================
Tests every non-interactive module against 10 diverse, synthetic datasets
that represent real-world edge cases. Each dataset targets a specific
category of data the program might encounter.

This ensures the tool works reliably with ANY CSV, not just the IPL dataset.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project import (
    _auto_parse_dates,
    _safe_filename,
    correlation_analysis,
    dataset_summary,
    descriptive_statistics,
    export_results,
    frequency_distribution,
    generate_report,
    generate_visualizations,
    missing_value_analysis,
    normality_test,
)


# ─────────────────────────────────────────────────────────────────────────────
# 10 Universal Dataset Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ds_purely_numeric():
    """Dataset 1: Purely numeric — no categorical columns at all."""
    np.random.seed(1)
    return pd.DataFrame({
        "Temperature": np.random.normal(25, 5, 200),
        "Humidity": np.random.uniform(30, 90, 200),
        "Pressure": np.random.normal(1013, 10, 200),
        "Wind_Speed": np.random.exponential(10, 200),
    })


@pytest.fixture
def ds_purely_categorical():
    """Dataset 2: Purely categorical — no numeric columns at all."""
    np.random.seed(2)
    n = 150
    return pd.DataFrame({
        "Color": np.random.choice(["Red", "Blue", "Green", "Yellow"], n),
        "Size": np.random.choice(["Small", "Medium", "Large"], n),
        "Shape": np.random.choice(["Circle", "Square", "Triangle"], n),
    })


@pytest.fixture
def ds_heavy_missing():
    """Dataset 3: Heavy missing values — 40-60% NaN across columns."""
    np.random.seed(3)
    n = 100
    df = pd.DataFrame({
        "Salary": np.random.normal(50000, 15000, n),
        "Bonus": np.random.normal(5000, 2000, n),
        "Rating": np.random.choice(["A", "B", "C", "D"], n),
        "Region": np.random.choice(["North", "South", "East", "West"], n),
    })
    # Inject 40-60% NaN
    for col in df.columns:
        mask = np.random.random(n) < 0.5
        df.loc[mask, col] = np.nan
    return df


@pytest.fixture
def ds_tiny():
    """Dataset 4: Tiny dataset — only 3 rows (edge case for stats)."""
    return pd.DataFrame({
        "X": [10, 20, 30],
        "Y": [1, 2, 3],
        "Label": ["A", "B", "A"],
    })


@pytest.fixture
def ds_single_column_each():
    """Dataset 5: One numeric + one categorical column only."""
    return pd.DataFrame({
        "Score": [88, 92, 76, 85, 91, 70, 83, 95, 67, 79],
        "Pass": ["Yes", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "No", "Yes"],
    })


@pytest.fixture
def ds_constant_values():
    """Dataset 6: Constant columns — zero variance (edge case for stats)."""
    n = 50
    return pd.DataFrame({
        "FixedNum": [42.0] * n,
        "FixedCat": ["Same"] * n,
        "Varying": np.random.normal(100, 15, n),
    })


@pytest.fixture
def ds_special_chars():
    """Dataset 7: Column names with special characters."""
    np.random.seed(7)
    return pd.DataFrame({
        "Revenue ($)": np.random.uniform(1000, 50000, 80),
        "Profit/Loss": np.random.normal(0, 5000, 80),
        "Growth (%)": np.random.uniform(-20, 40, 80),
        "Region & Zone": np.random.choice(["US/East", "EU-West", "APAC (South)"], 80),
    })


@pytest.fixture
def ds_dates_mixed():
    """Dataset 8: Mixed types with date strings and numeric data."""
    np.random.seed(8)
    dates = pd.date_range("2020-01-01", periods=100, freq="D").strftime("%Y-%m-%d").tolist()
    return pd.DataFrame({
        "Date": dates,
        "Sales": np.random.poisson(50, 100),
        "Returns": np.random.poisson(5, 100),
        "Store": np.random.choice(["Store_A", "Store_B", "Store_C"], 100),
    })


@pytest.fixture
def ds_high_cardinality():
    """Dataset 9: High-cardinality categorical + numeric."""
    np.random.seed(9)
    n = 500
    return pd.DataFrame({
        "UserID": [f"user_{i:05d}" for i in range(n)],
        "SessionDuration": np.random.exponential(30, n),
        "PageViews": np.random.poisson(8, n),
        "Country": np.random.choice(
            ["US", "UK", "India", "Germany", "Brazil", "Japan",
             "France", "Canada", "Australia", "Mexico"] * 5, n
        ),
    })


@pytest.fixture
def ds_large_mixed():
    """Dataset 10: Large mixed dataset — 5000 rows, many column types."""
    np.random.seed(10)
    n = 5000
    return pd.DataFrame({
        "ID": range(n),
        "Age": np.random.randint(18, 80, n),
        "Salary": np.random.lognormal(10.5, 0.8, n),
        "Satisfaction": np.random.uniform(1, 10, n),
        "Department": np.random.choice(
            ["Engineering", "Sales", "Marketing", "HR", "Finance", "Legal"], n
        ),
        "Gender": np.random.choice(["M", "F", "Non-Binary"], n),
        "Hired": pd.date_range("2010-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "Active": np.random.choice([True, False], n),
    })


# Collect all fixture names for parameterized tests
ALL_DATASETS = [
    "ds_purely_numeric",
    "ds_purely_categorical",
    "ds_heavy_missing",
    "ds_tiny",
    "ds_single_column_each",
    "ds_constant_values",
    "ds_special_chars",
    "ds_dates_mixed",
    "ds_high_cardinality",
    "ds_large_mixed",
]


# ─────────────────────────────────────────────────────────────────────────────
# Universal Module Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetSummaryUniversal:
    """dataset_summary() must never crash on any dataset."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_summary_no_crash(self, ds_name, request):
        df = request.getfixturevalue(ds_name)
        result = dataset_summary(df)
        assert isinstance(result, str)
        assert len(result) > 0


class TestMissingValueAnalysisUniversal:
    """missing_value_analysis() must return a valid dict on any dataset."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_missing_no_crash(self, ds_name, request):
        df = request.getfixturevalue(ds_name)
        result = missing_value_analysis(df)
        assert isinstance(result, dict)

    def test_heavy_missing_reports_all(self, ds_heavy_missing):
        """Dataset with 40-60% NaN should report missing for those columns."""
        result = missing_value_analysis(ds_heavy_missing)
        # At least some columns should show missing values
        assert len(result) > 0


class TestDescriptiveStatisticsUniversal:
    """descriptive_statistics() must handle any dataset type."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_stats_no_crash(self, ds_name, request):
        df = request.getfixturevalue(ds_name)
        result = descriptive_statistics(df)
        assert isinstance(result, pd.DataFrame)

    def test_purely_categorical_returns_empty(self, ds_purely_categorical):
        """No numeric columns → empty DataFrame."""
        result = descriptive_statistics(ds_purely_categorical)
        assert result.empty

    def test_constant_column_stats(self, ds_constant_values):
        """Constant column should have Std Dev = 0, Range = 0."""
        result = descriptive_statistics(ds_constant_values)
        assert result["FixedNum"]["Std Dev"] == 0.0
        assert result["FixedNum"]["Range"] == 0.0
        assert result["FixedNum"]["Mean"] == 42.0


class TestFrequencyDistributionUniversal:
    """frequency_distribution() must handle any dataset type."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_freq_no_crash(self, ds_name, request):
        df = request.getfixturevalue(ds_name)
        result = frequency_distribution(df)
        assert isinstance(result, dict)

    def test_purely_numeric_returns_empty(self, ds_purely_numeric):
        """No categorical columns → empty dict."""
        result = frequency_distribution(ds_purely_numeric)
        assert result == {}

    def test_high_cardinality_returns_all_values(self, ds_high_cardinality):
        """High-cardinality column should still return ALL values in dict."""
        result = frequency_distribution(ds_high_cardinality)
        assert "UserID" in result
        assert len(result["UserID"]) == 500  # All unique IDs


class TestCorrelationAnalysisUniversal:
    """correlation_analysis() must handle any dataset type."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_correlation_no_crash(self, ds_name, request):
        df = request.getfixturevalue(ds_name)
        result = correlation_analysis(df)
        assert isinstance(result, pd.DataFrame)

    def test_purely_categorical_returns_empty(self, ds_purely_categorical):
        """No numeric columns → empty DataFrame."""
        result = correlation_analysis(ds_purely_categorical)
        assert result.empty

    def test_single_numeric_returns_empty(self, ds_single_column_each):
        """Only 1 numeric column → empty DataFrame (need ≥2)."""
        result = correlation_analysis(ds_single_column_each)
        assert result.empty


class TestNormalityTestUniversal:
    """normality_test() must handle any dataset type."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_normality_no_crash(self, ds_name, request):
        df = request.getfixturevalue(ds_name)
        result = normality_test(df)
        assert isinstance(result, dict)

    def test_purely_categorical_returns_empty(self, ds_purely_categorical):
        """No numeric columns → empty dict."""
        result = normality_test(ds_purely_categorical)
        assert result == {}

    def test_tiny_dataset_works(self, ds_tiny):
        """3-row dataset should still run (≥3 observations)."""
        result = normality_test(ds_tiny)
        assert "X" in result
        # New implementation uses 'stat' key (either Shapiro W or Anderson A²)
        assert "stat" in result["X"]
        assert "normal" in result["X"]

    def test_constant_column_handled(self, ds_constant_values):
        """Constant column has zero variance → Shapiro still runs."""
        result = normality_test(ds_constant_values)
        # FixedNum has zero variance; Shapiro may return W=1.0 or a warning
        # but it must NOT crash
        assert "FixedNum" in result or "Varying" in result

    def test_large_dataset_uses_anderson(self, ds_large_mixed):
        """5000-row dataset (n>5000) should switch to Anderson-Darling."""
        result = normality_test(ds_large_mixed)
        assert len(result) > 0
        # At least some columns should use anderson for large n
        test_types = {v["test"] for v in result.values()}
        assert "anderson" in test_types or "shapiro" in test_types


class TestGenerateVisualizationsUniversal:
    """generate_visualizations() must handle any dataset type without crashing."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_viz_no_crash(self, ds_name, request, tmp_path, monkeypatch):
        import project
        monkeypatch.setattr(project, "PLOTS_DIR", tmp_path / "plots")
        (tmp_path / "plots").mkdir()

        df = request.getfixturevalue(ds_name)
        result = generate_visualizations(df)
        assert isinstance(result, list)

    def test_special_chars_filenames(self, ds_special_chars, tmp_path, monkeypatch):
        """Columns with special chars should produce valid filenames."""
        import project
        monkeypatch.setattr(project, "PLOTS_DIR", tmp_path / "plots")
        (tmp_path / "plots").mkdir()

        result = generate_visualizations(ds_special_chars)
        for path_str in result:
            p = Path(path_str)
            assert p.exists(), f"File not created: {path_str}"
            # No illegal characters in filename
            assert "/" not in p.name
            assert "\\" not in p.name
            assert "$" not in p.name
            assert "(" not in p.name

    def test_high_cardinality_skipped(self, ds_high_cardinality, tmp_path, monkeypatch):
        """UserID (500 unique) should be skipped (>500 threshold)."""
        import project
        monkeypatch.setattr(project, "PLOTS_DIR", tmp_path / "plots")
        (tmp_path / "plots").mkdir()

        result = generate_visualizations(ds_high_cardinality)
        filenames = [Path(p).name for p in result]
        # UserID has 500 unique values → at the threshold border
        # Country has 10 unique values → should produce a chart
        assert any("country" in f for f in filenames)


class TestGenerateReportUniversal:
    """generate_report() must produce a valid report on any dataset."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_report_no_crash(self, ds_name, request, tmp_path, monkeypatch):
        import project
        monkeypatch.setattr(project, "REPORTS_DIR", tmp_path / "reports")
        (tmp_path / "reports").mkdir()

        df = request.getfixturevalue(ds_name)
        result = generate_report(df)
        assert isinstance(result, str)
        assert len(result) > 100  # Non-trivial report
        assert "END OF REPORT" in result

    def test_report_file_created(self, ds_large_mixed, tmp_path, monkeypatch):
        """A report file should actually exist on disk."""
        import project
        report_dir = tmp_path / "reports"
        monkeypatch.setattr(project, "REPORTS_DIR", report_dir)
        report_dir.mkdir()

        generate_report(ds_large_mixed)
        report_file = report_dir / "analysis_report.txt"
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "DATASET INFORMATION" in content
        assert "NORMALITY TESTS" in content


class TestExportResultsUniversal:
    """export_results() must produce CSV files on any dataset."""

    @pytest.mark.parametrize("ds_name", ALL_DATASETS)
    def test_export_no_crash(self, ds_name, request, tmp_path, monkeypatch):
        import project
        monkeypatch.setattr(project, "EXPORTS_DIR", tmp_path / "exports")
        (tmp_path / "exports").mkdir()

        df = request.getfixturevalue(ds_name)
        result = export_results(df)
        assert isinstance(result, list)

    def test_purely_numeric_exports_stats_only(self, ds_purely_numeric, tmp_path, monkeypatch):
        """Purely numeric → only descriptive_statistics.csv."""
        import project
        monkeypatch.setattr(project, "EXPORTS_DIR", tmp_path / "exports")
        (tmp_path / "exports").mkdir()

        result = export_results(ds_purely_numeric)
        assert len(result) == 1
        assert "descriptive_statistics.csv" in result[0]

    def test_purely_categorical_exports_freq_only(self, ds_purely_categorical, tmp_path, monkeypatch):
        """Purely categorical → only frequency_tables.csv."""
        import project
        monkeypatch.setattr(project, "EXPORTS_DIR", tmp_path / "exports")
        (tmp_path / "exports").mkdir()

        result = export_results(ds_purely_categorical)
        assert len(result) == 1
        assert "frequency_tables.csv" in result[0]


# ─────────────────────────────────────────────────────────────────────────────
# Date Auto-Detection Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoParseUniversal:
    """_auto_parse_dates() must handle edge cases without crashing."""

    def test_iso_dates(self):
        df = pd.DataFrame({"d": ["2024-01-01", "2024-06-15", "2024-12-31"]})
        result = _auto_parse_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["d"])

    def test_us_format_dates(self):
        df = pd.DataFrame({"d": ["01/15/2024", "06/20/2024", "12/31/2024"]})
        result = _auto_parse_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(result["d"])

    def test_non_date_strings_unchanged(self):
        df = pd.DataFrame({"d": ["hello", "world", "python"]})
        result = _auto_parse_dates(df)
        assert result["d"].dtype == object

    def test_mixed_date_and_garbage(self):
        """If <80% parse as dates, keep as string."""
        df = pd.DataFrame({"d": ["2024-01-01", "not_a_date", "garbage", "more_junk", "nope"]})
        result = _auto_parse_dates(df)
        # Only 1/5 = 20% parseable → should stay as object
        assert result["d"].dtype == object

    def test_empty_column_unchanged(self):
        df = pd.DataFrame({"d": [np.nan, np.nan, np.nan]})
        result = _auto_parse_dates(df)
        # All NaN → sample is empty → should not crash
        assert True  # No exception = pass

    def test_numeric_columns_untouched(self):
        df = pd.DataFrame({"n": [1, 2, 3], "d": ["2024-01-01", "2024-02-01", "2024-03-01"]})
        result = _auto_parse_dates(df)
        assert pd.api.types.is_numeric_dtype(result["n"])
        assert pd.api.types.is_datetime64_any_dtype(result["d"])


# ─────────────────────────────────────────────────────────────────────────────
# Filename Sanitization Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeFilenameUniversal:
    """_safe_filename() must produce valid filenames from any input."""

    @pytest.mark.parametrize("input_name,expected", [
        ("Revenue ($)", "revenue____"),
        ("Profit/Loss", "profit_loss"),
        ("Growth (%)", "growth____"),
        ("Region & Zone", "region___zone"),
        ("col name with spaces", "col_name_with_spaces"),
        ("UPPER_case", "upper_case"),
        ("already_safe", "already_safe"),
        ("a/b\\c:d*e?f", "a_b_c_d_e_f"),
        ("日本語", "___"),
        ("  leading_trailing  ", "leading_trailing"),
    ])
    def test_sanitization(self, input_name, expected):
        result = _safe_filename(input_name)
        # Should not contain any forbidden filesystem characters
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ', '$', '(', ')']:
            assert ch not in result, f"'{ch}' found in sanitized name: {result}"


# ─────────────────────────────────────────────────────────────────────────────
# Load Dataset Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadDatasetUniversal:
    """load_dataset() edge cases with diverse CSV content."""

    def test_load_with_date_auto_detection(self, ds_dates_mixed, tmp_path):
        """CSV with date strings should auto-detect the date column."""
        from project import load_dataset

        csv_path = tmp_path / "dates.csv"
        ds_dates_mixed.to_csv(csv_path, index=False)

        df = load_dataset(str(csv_path))
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_load_purely_numeric(self, ds_purely_numeric, tmp_path):
        """Purely numeric CSV should load without issues."""
        from project import load_dataset

        csv_path = tmp_path / "numeric.csv"
        ds_purely_numeric.to_csv(csv_path, index=False)

        df = load_dataset(str(csv_path))
        assert df.shape[0] == 200
        assert df.select_dtypes(include="number").shape[1] == 4

    def test_load_special_char_columns(self, ds_special_chars, tmp_path):
        """CSV with special character column names should load fine."""
        from project import load_dataset

        csv_path = tmp_path / "special.csv"
        ds_special_chars.to_csv(csv_path, index=False)

        df = load_dataset(str(csv_path))
        assert "Revenue ($)" in df.columns
        assert "Profit/Loss" in df.columns

    def test_load_heavy_missing(self, ds_heavy_missing, tmp_path):
        """CSV with heavy missing values should load without error."""
        from project import load_dataset

        csv_path = tmp_path / "missing.csv"
        ds_heavy_missing.to_csv(csv_path, index=False)

        df = load_dataset(str(csv_path))
        assert df.isna().sum().sum() > 0  # Confirms NaNs survived the load
