import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project import (
    calculate_mean,
    correlation_analysis,
    dataset_summary,
    descriptive_statistics,
    export_results,
    frequency_distribution,
    load_dataset,
    missing_value_analysis,
)



# Fixtures — Reusable Mock Data

@pytest.fixture
def sample_df():
    """A small DataFrame that mimics a typical CSV input."""
    return pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "Diana", "Edward"],
        "Age": [21, 23, 22, 20, 25],
        "Gender": ["Female", "Male", "Male", "Female", "Male"],
        "Income": [32000, 28000, 35000, np.nan, 45000],
        "GPA": [3.5, 3.1, 3.8, 3.9, 2.9],
        "Department": ["CS", "BUS", "CS", "PSY", "BUS"],
        "Smoking": ["No", "Yes", "No", "No", "Yes"],
    })


@pytest.fixture
def numeric_only_df():
    """DataFrame with only numeric columns — for focused stats tests."""
    return pd.DataFrame({
        "X": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Y": [2.0, 4.0, 6.0, 8.0, 10.0],
    })


@pytest.fixture
def missing_df():
    """DataFrame with known missing values for testing."""
    return pd.DataFrame({
        "A": [1, 2, np.nan, 4, np.nan],
        "B": ["x", np.nan, "z", "w", "v"],
        "C": [10, 20, 30, 40, 50],
    })


@pytest.fixture
def temp_csv(sample_df):
    """Create a temporary CSV file from sample_df."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        sample_df.to_csv(f, index=False)
        path = f.name
    yield path
    os.unlink(path)  # Cleanup after test



# Tests: load_dataset()

class TestLoadDataset:
    """Tests for the dataset loading function."""

    def test_load_valid_csv(self, temp_csv):
        """Loading a valid CSV returns a DataFrame with correct shape."""
        df = load_dataset(temp_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 7)

    def test_load_file_not_found(self):
        """A missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dataset("this_file_does_not_exist.csv")

    def test_load_invalid_format(self):
        """A non-CSV file raises ValueError."""
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx", delete=False
        ) as f:
            f.write(b"fake data")
            path = f.name
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_dataset(path)
        finally:
            os.unlink(path)

    def test_load_empty_csv(self):
        """An empty CSV raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as f:
            f.write("")  # Empty file
            path = f.name
        try:
            with pytest.raises(ValueError):
                load_dataset(path)
        finally:
            os.unlink(path)



# Tests: missing_value_analysis()

class TestMissingValueAnalysis:
    """Tests for the missing value analysis function."""

    def test_correct_missing_counts(self, missing_df):
        """Verify correct count and percentage for known NaNs."""
        result = missing_value_analysis(missing_df)

        assert "A" in result
        assert result["A"][0] == 2           # 2 missing in column A
        assert result["A"][1] == 40.0        # 2/5 = 40%

        assert "B" in result
        assert result["B"][0] == 1           # 1 missing in column B
        assert result["B"][1] == 20.0        # 1/5 = 20%

    def test_no_missing_values(self, numeric_only_df):
        """A DataFrame with no NaNs returns an empty dict."""
        result = missing_value_analysis(numeric_only_df)
        assert result == {}

    def test_complete_column_excluded(self, missing_df):
        """Column C (no NaNs) should not appear in the result."""
        result = missing_value_analysis(missing_df)
        assert "C" not in result



#  Tests: descriptive_statistics()

class TestDescriptiveStatistics:
    """Tests for the descriptive statistics function."""

    def test_mean_calculation(self, numeric_only_df):
        """Mean of [10, 20, 30, 40, 50] should be 30.0."""
        result = descriptive_statistics(numeric_only_df)
        assert result["X"]["Mean"] == 30.0

    def test_median_calculation(self, numeric_only_df):
        """Median of [10, 20, 30, 40, 50] should be 30.0."""
        result = descriptive_statistics(numeric_only_df)
        assert result["X"]["Median"] == 30.0

    def test_std_dev_uses_ddof1(self, numeric_only_df):
        """Standard deviation must use ddof=1 (sample SD)."""
        result = descriptive_statistics(numeric_only_df)
        expected_sd = round(numeric_only_df["X"].std(ddof=1), 4)
        assert result["X"]["Std Dev"] == expected_sd

    def test_variance_uses_ddof1(self, numeric_only_df):
        """Variance must use ddof=1 (sample variance)."""
        result = descriptive_statistics(numeric_only_df)
        expected_var = round(numeric_only_df["X"].var(ddof=1), 4)
        assert result["X"]["Variance"] == expected_var

    def test_range_calculation(self, numeric_only_df):
        """Range of [10, 20, 30, 40, 50] should be 40.0."""
        result = descriptive_statistics(numeric_only_df)
        assert result["X"]["Range"] == 40.0

    def test_min_max(self, numeric_only_df):
        """Min and Max should match known values."""
        result = descriptive_statistics(numeric_only_df)
        assert result["X"]["Min"] == 10.0
        assert result["X"]["Max"] == 50.0



#  Tests: frequency_distribution()

class TestFrequencyDistribution:
    """Tests for the frequency distribution function."""

    def test_correct_counts(self, sample_df):
        """Gender frequency should match known distribution."""
        result = frequency_distribution(sample_df)
        assert "Gender" in result

        gender_df = result["Gender"]
        male_row = gender_df[gender_df["Value"] == "Male"]
        female_row = gender_df[gender_df["Value"] == "Female"]

        assert int(male_row["Frequency"].values[0]) == 3
        assert int(female_row["Frequency"].values[0]) == 2

    def test_percentages_sum_to_100(self, sample_df):
        """Percentages within a variable should sum to ~100%."""
        result = frequency_distribution(sample_df)
        for col, freq_df in result.items():
            total_pct = freq_df["Percentage"].sum()
            assert abs(total_pct - 100.0) < 0.1, f"{col} percentages sum to {total_pct}"

    def test_no_categorical_columns(self, numeric_only_df):
        """A purely numeric DataFrame returns an empty dict."""
        result = frequency_distribution(numeric_only_df)
        assert result == {}



#  Tests: correlation_analysis()

class TestCorrelationAnalysis:
    """Tests for the Pearson correlation function."""

    def test_perfect_correlation(self):
        """Perfectly correlated variables should have r = 1.0."""
        df = pd.DataFrame({
            "X": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Y": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        corr_matrix = correlation_analysis(df)
        assert abs(corr_matrix.loc["X", "Y"] - 1.0) < 1e-10

    def test_negative_correlation(self):
        """Inversely correlated variables should have r ≈ -1.0."""
        df = pd.DataFrame({
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        corr_matrix = correlation_analysis(df)
        assert abs(corr_matrix.loc["A", "B"] - (-1.0)) < 1e-10

    def test_single_numeric_col_returns_empty(self):
        """With only 1 numeric column, return empty DataFrame."""
        df = pd.DataFrame({"X": [1, 2, 3]})
        result = correlation_analysis(df)
        assert result.empty



#  Tests: dataset_summary()

class TestDatasetSummary:
    """Tests for the dataset summary function."""

    def test_contains_variable_names(self, sample_df, capsys):
        """The summary text should contain all column names."""
        summary = dataset_summary(sample_df)
        for col in sample_df.columns:
            assert col in summary

    def test_contains_observation_count(self, sample_df):
        """The summary should report the correct row count."""
        summary = dataset_summary(sample_df)
        assert "5" in summary  # 5 rows

    def test_numeric_type_label(self, sample_df):
        """Numeric columns should be labelled 'Numeric'."""
        summary = dataset_summary(sample_df)
        assert "Numeric" in summary

    def test_categorical_type_label(self, sample_df):
        """Categorical columns should be labelled 'Categorical'."""
        summary = dataset_summary(sample_df)
        assert "Categorical" in summary



#  Tests: calculate_mean()

class TestCalculateMean:
    """Tests for the standalone calculate_mean utility."""

    def test_basic_mean(self):
        assert calculate_mean([10, 20, 30]) == 20.0

    def test_single_value(self):
        assert calculate_mean([42]) == 42.0

    def test_floats(self):
        result = calculate_mean([1.5, 2.5, 3.0])
        assert abs(result - 2.3333) < 0.01

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            calculate_mean([])



#  Tests: export_results()

class TestExportResults:
    """Tests for the CSV export function."""

    def test_exports_files(self, sample_df, tmp_path, monkeypatch):
        """Verify that export creates the expected CSV files."""
        # Redirect EXPORTS_DIR to a temp directory
        import project
        monkeypatch.setattr(project, "EXPORTS_DIR", tmp_path / "exports")
        (tmp_path / "exports").mkdir()

        saved = export_results(sample_df)
        assert len(saved) == 2  # descriptive_statistics.csv + frequency_tables.csv

        for path_str in saved:
            assert os.path.exists(path_str)
            assert path_str.endswith(".csv")
