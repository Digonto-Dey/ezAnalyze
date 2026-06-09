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
    normality_test,
    one_way_anova,
    linear_regression,
    multiple_regression_test,
    logistic_regression,
    data_cleaning_menu,
    interactive_filter,
    _safe_filename,
    _auto_parse_dates,
    _remove_outliers_iqr,
    _remove_outliers_zscore,
    _dummy_encode,
    _sigmoid,
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


@pytest.fixture
def anova_df():
    """DataFrame suitable for one-way ANOVA testing."""
    return pd.DataFrame({
        "Score": [85, 90, 78, 92, 88, 60, 65, 70, 55, 72, 95, 98, 100, 88, 92],
        "Group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C"],
    })


@pytest.fixture
def date_df():
    """DataFrame with a date-like string column for auto-detection testing."""
    return pd.DataFrame({
        "event": ["game1", "game2", "game3"],
        "date": ["2024-01-15", "2024-02-20", "2024-03-10"],
        "score": [100, 200, 150],
    })



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

    def test_high_cardinality_still_returns_all(self):
        """Frequency distribution returns ALL values in dict, even if display is capped."""
        many_cats = pd.DataFrame({
            "Cat": [f"value_{i}" for i in range(100)]
        })
        result = frequency_distribution(many_cats)
        assert "Cat" in result
        assert len(result["Cat"]) == 100  # Full result returned



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



#  Tests: normality_test()

class TestNormalityTest:
    """Tests for the Shapiro-Wilk normality test function."""

    def test_normal_data_detected(self):
        """Data drawn from a normal distribution should be detected as normal."""
        np.random.seed(42)
        df = pd.DataFrame({"Normal": np.random.normal(100, 15, 200)})
        result = normality_test(df)
        assert "Normal" in result
        assert result["Normal"]["normal"] == True

    def test_uniform_data_detected_non_normal(self):
        """Uniform data should be detected as non-normal."""
        np.random.seed(42)
        df = pd.DataFrame({"Uniform": np.random.uniform(0, 100, 200)})
        result = normality_test(df)
        assert "Uniform" in result
        assert result["Uniform"]["normal"] == False

    def test_no_numeric_columns(self):
        """A DataFrame with no numeric columns returns an empty dict."""
        df = pd.DataFrame({"Name": ["Alice", "Bob", "Charlie"]})
        result = normality_test(df)
        assert result == {}

    def test_too_few_observations_skipped(self):
        """Columns with fewer than 3 observations should be skipped."""
        df = pd.DataFrame({"Short": [1.0, 2.0]})
        result = normality_test(df)
        assert "Short" not in result

    def test_result_keys(self):
        """Each result entry should have stat, p_value, normal, test, skewness, kurtosis keys."""
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = normality_test(df)
        assert "stat" in result["X"]       # W (Shapiro) or A² (Anderson)
        assert "p_value" in result["X"]
        assert "normal" in result["X"]
        assert "test" in result["X"]       # 'shapiro' or 'anderson'
        assert "skewness" in result["X"]
        assert "kurtosis" in result["X"]



#  Tests: one_way_anova()

class TestOneWayAnova:
    """Tests for the one-way ANOVA function."""

    def test_significant_anova(self, anova_df, monkeypatch):
        """Groups with clearly different means should yield significant F-test."""
        # Simulate user input: "Score" for numeric, "Group" for grouping
        inputs = iter(["Score", "Group"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = one_way_anova(anova_df)
        assert result["significant"] == True
        assert result["f_stat"] > 0
        assert len(result["groups"]) == 3

    def test_non_significant_anova(self, monkeypatch):
        """Groups with similar means should not be significant."""
        df = pd.DataFrame({
            "Value": [10, 11, 10, 11, 10, 11, 10, 11, 10, 11],
            "Cat": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        })
        inputs = iter(["Value", "Cat"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = one_way_anova(df)
        assert result["significant"] == False

    def test_no_numeric_columns(self):
        """Returns empty dict when no numeric columns exist."""
        df = pd.DataFrame({"A": ["x", "y"], "B": ["a", "b"]})
        result = one_way_anova(df)
        assert result == {}



#  Tests: data_cleaning_menu()

class TestDataCleaningMenu:
    """Tests for the data cleaning tools."""

    def test_drop_na(self, missing_df, monkeypatch):
        """Option 1 should drop rows with missing values."""
        inputs = iter(["1", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = data_cleaning_menu(missing_df.copy())
        assert result.isna().sum().sum() == 0
        assert len(result) == 2  # 5 rows, 3 have at least 1 NaN → 2 remain

    def test_fill_mean(self, missing_df, monkeypatch):
        """Option 2 should fill numeric NaNs with column mean."""
        inputs = iter(["2", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = data_cleaning_menu(missing_df.copy())
        # Column A had NaN — now filled with mean of [1,2,4] = 2.333...
        assert result["A"].isna().sum() == 0

    def test_fill_median(self, missing_df, monkeypatch):
        """Option 3 should fill numeric NaNs with column median."""
        inputs = iter(["3", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = data_cleaning_menu(missing_df.copy())
        assert result["A"].isna().sum() == 0

    def test_remove_duplicates(self, monkeypatch):
        """Option 4 should remove duplicate rows."""
        df = pd.DataFrame({"X": [1, 1, 2, 3], "Y": ["a", "a", "b", "c"]})
        inputs = iter(["4", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = data_cleaning_menu(df.copy())
        assert len(result) == 3

    def test_drop_column(self, sample_df, monkeypatch):
        """Option 5 should drop the specified column."""
        inputs = iter(["5", "Name", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = data_cleaning_menu(sample_df.copy())
        assert "Name" not in result.columns



#  Tests: _safe_filename()

class TestSafeFilename:
    """Tests for the filename sanitization helper."""

    def test_spaces_replaced(self):
        assert _safe_filename("my column") == "my_column"

    def test_special_chars_replaced(self):
        assert _safe_filename("col/name\\test") == "col_name_test"

    def test_uppercase_lowered(self):
        assert _safe_filename("MyColumn") == "mycolumn"

    def test_already_safe(self):
        assert _safe_filename("simple_name") == "simple_name"



#  Tests: _auto_parse_dates()

class TestAutoParseDate:
    """Tests for the automatic date column detection."""

    def test_date_column_detected(self, date_df):
        """A column with ISO date strings should be converted to datetime."""
        result = _auto_parse_dates(date_df.copy())
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_non_date_column_unchanged(self, date_df):
        """Non-date string columns should remain as object dtype."""
        result = _auto_parse_dates(date_df.copy())
        assert result["event"].dtype == object

    def test_numeric_columns_unchanged(self, date_df):
        """Numeric columns should not be touched."""
        result = _auto_parse_dates(date_df.copy())
        assert pd.api.types.is_numeric_dtype(result["score"])


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: normality_test() — extended
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalityTestExtended:
    """Extended tests for the robust normality_test function."""

    def test_result_structure(self):
        """Each result must have required keys."""
        df = pd.DataFrame({"X": np.random.normal(0, 1, 50)})
        result = normality_test(df)
        assert "X" in result
        for key in ("test", "stat", "p_value", "normal", "skewness", "kurtosis"):
            assert key in result["X"], f"Missing key: {key}"

    def test_shapiro_used_for_small_n(self):
        """n ≤ 5000 should use Shapiro-Wilk test."""
        np.random.seed(0)
        df = pd.DataFrame({"V": np.random.normal(10, 2, 100)})
        result = normality_test(df)
        assert result["V"]["test"] == "shapiro"

    def test_anderson_used_for_large_n(self):
        """n > 5000 should use Anderson-Darling test."""
        np.random.seed(1)
        df = pd.DataFrame({"V": np.random.normal(0, 1, 6000)})
        result = normality_test(df)
        assert result["V"]["test"] == "anderson"

    def test_non_normal_skewed_data(self):
        """Heavily skewed exponential data should be flagged as non-normal."""
        np.random.seed(42)
        df = pd.DataFrame({"Exp": np.random.exponential(1, 500)})
        result = normality_test(df)
        assert result["Exp"]["normal"] is False

    def test_skewness_and_kurtosis_returned(self):
        """Skewness and kurtosis values must be numeric floats."""
        df = pd.DataFrame({"Z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
        result = normality_test(df)
        assert isinstance(result["Z"]["skewness"], float)
        assert isinstance(result["Z"]["kurtosis"], float)

    def test_column_error_isolated(self):
        """A bad column should not prevent other columns from being tested."""
        np.random.seed(3)
        df = pd.DataFrame({
            "Good": np.random.normal(0, 1, 50),
            "AllSame": [5.0] * 50,   # zero variance — Shapiro may warn
        })
        result = normality_test(df)
        assert "Good" in result


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: linear_regression() — simple & multiple
# ─────────────────────────────────────────────────────────────────────────────

class TestLinearRegression:
    """Tests for the robust linear_regression function."""

    @pytest.fixture
    def lr_df(self):
        np.random.seed(7)
        x = np.arange(1, 51, dtype=float)
        return pd.DataFrame({
            "X":  x,
            "X2": x + np.random.normal(0, 0.5, 50),
            "Y":  2.5 * x + 10 + np.random.normal(0, 2, 50),
        })

    def test_simple_regression_significant(self, lr_df, monkeypatch):
        """Simple regression on perfectly-correlated data should be significant."""
        inputs = iter(["Y", "X"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = linear_regression(lr_df)
        assert result["significant"] is True
        assert result["r_squared"] > 0.9
        assert result["k"] == 1

    def test_result_has_required_keys(self, lr_df, monkeypatch):
        """Result dict must contain all documented keys."""
        inputs = iter(["Y", "X"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = linear_regression(lr_df)
        for key in ("y_var", "x_vars", "n", "k", "intercept", "coefficients",
                    "r_squared", "adj_r_squared", "f_stat", "f_pvalue", "significant"):
            assert key in result, f"Missing key: {key}"

    def test_multiple_regression(self, lr_df, monkeypatch):
        """Multiple regression with 2 predictors should run without error."""
        inputs = iter(["Y", "X, X2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = linear_regression(lr_df)
        assert result["k"] == 2
        assert len(result["coefficients"]) == 2
        assert "vif" in result and len(result["vif"]) == 2

    def test_not_enough_columns(self):
        """Single numeric column should return empty dict."""
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
        result = linear_regression(df)
        assert result == {}

    def test_insufficient_observations(self, monkeypatch):
        """Too few rows for regression raises ValueError."""
        df = pd.DataFrame({"X": [1.0, 2.0], "Y": [1.0, 2.0]})
        inputs = iter(["Y", "X"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        with pytest.raises(ValueError, match="complete observations"):
            linear_regression(df)

    def test_y_in_x_raises(self, lr_df, monkeypatch):
        """Using Y as both dependent and independent variable should raise ValueError."""
        inputs = iter(["Y", "Y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        with pytest.raises(ValueError, match="cannot also be"):
            linear_regression(lr_df)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: one_way_anova() — extended with Levene + Tukey
# ─────────────────────────────────────────────────────────────────────────────

class TestOneWayAnovaExtended:
    """Extended ANOVA tests including Levene and Tukey results."""

    @pytest.fixture
    def anova_df(self):
        return pd.DataFrame({
            "Score": [85, 90, 78, 92, 88, 60, 65, 70, 55, 72, 95, 98, 100, 88, 92],
            "Group": ["A"]*5 + ["B"]*5 + ["C"]*5,
        })

    def test_result_has_levene_keys(self, anova_df, monkeypatch):
        """Result must include Levene's test keys."""
        inputs = iter(["Score", "Group"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = one_way_anova(anova_df)
        assert "levene_stat" in result
        assert "levene_p" in result
        assert "equal_var" in result

    def test_result_has_welch_keys(self, anova_df, monkeypatch):
        """Result must include Welch ANOVA keys."""
        inputs = iter(["Score", "Group"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = one_way_anova(anova_df)
        assert "welch_f" in result
        assert "welch_p" in result

    def test_tukey_hsd_present_when_significant(self, anova_df, monkeypatch):
        """When ANOVA is significant, Tukey HSD comparisons should be populated."""
        inputs = iter(["Score", "Group"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = one_way_anova(anova_df)
        assert result["significant"] is True
        assert isinstance(result["tukey_hsd"], list)
        assert len(result["tukey_hsd"]) > 0

    def test_tukey_comparison_keys(self, anova_df, monkeypatch):
        """Each Tukey comparison dict must have the required keys."""
        inputs = iter(["Score", "Group"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = one_way_anova(anova_df)
        for comp in result["tukey_hsd"]:
            for key in ("group1", "group2", "mean_diff", "q_stat", "p_approx", "significant"):
                assert key in comp

    def test_group_stats_present(self, anova_df, monkeypatch):
        """Result must include per-group descriptive stats."""
        inputs = iter(["Score", "Group"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = one_way_anova(anova_df)
        assert "group_stats" in result
        for g in ["A", "B", "C"]:
            assert g in result["group_stats"]
            assert "mean" in result["group_stats"][g]
            assert "n" in result["group_stats"][g]


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: data_cleaning_menu() — new options
# ─────────────────────────────────────────────────────────────────────────────

class TestDataCleaningMenuExtended:
    """Tests for the new data cleaning options (6–11)."""

    @pytest.fixture
    def clean_df(self):
        return pd.DataFrame({
            "A": [1.0, 2.0, 3.0, 100.0, 4.0],   # 100.0 is an outlier
            "B": ["yes", "no", None, "yes", "no"],  # categorical with NaN
            "C": ["  alice ", " bob", "carol ", "dave", " eve "],  # whitespace
        })

    def test_fill_categorical_mode(self, clean_df, monkeypatch):
        """Option 6 fills categorical NaN with mode."""
        inputs = iter(["6", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = data_cleaning_menu(clean_df.copy())
        assert result["B"].isna().sum() == 0

    def test_remove_outliers_iqr_option(self, clean_df, monkeypatch):
        """Option 7 removes IQR outliers from numeric column."""
        inputs = iter(["7", "A", "1.5", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = data_cleaning_menu(clean_df.copy())
        assert 100.0 not in result["A"].values

    def test_remove_outliers_zscore_option(self, clean_df, monkeypatch):
        """Option 8 removes Z-score outliers from numeric column."""
        inputs = iter(["8", "A", "2.0", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = data_cleaning_menu(clean_df.copy())
        assert 100.0 not in result["A"].values

    def test_rename_column(self, clean_df, monkeypatch):
        """Option 9 renames a column."""
        inputs = iter(["9", "A", "NumericA", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = data_cleaning_menu(clean_df.copy())
        assert "NumericA" in result.columns
        assert "A" not in result.columns

    def test_strip_whitespace(self, clean_df, monkeypatch):
        """Option 10 strips whitespace from string columns."""
        inputs = iter(["10", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = data_cleaning_menu(clean_df.copy())
        # All values in C should be stripped
        non_null = result["C"].dropna()
        for val in non_null:
            assert val == val.strip()

    def test_cast_column(self, monkeypatch):
        """Option 11 casts a numeric column to float."""
        df = pd.DataFrame({"Num": ["1", "2", "3"], "Cat": ["a", "b", "c"]})
        inputs = iter(["11", "Num", "float", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = data_cleaning_menu(df.copy())
        assert pd.api.types.is_float_dtype(result["Num"])


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: _remove_outliers_iqr() and _remove_outliers_zscore()
# ─────────────────────────────────────────────────────────────────────────────

class TestOutlierRemoval:
    """Unit tests for outlier removal utility functions."""

    @pytest.fixture
    def outlier_df(self):
        return pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})

    def test_iqr_removes_extreme_outlier(self, outlier_df):
        result = _remove_outliers_iqr(outlier_df, "X")
        assert 1000.0 not in result["X"].values

    def test_iqr_keeps_normal_values(self, outlier_df):
        result = _remove_outliers_iqr(outlier_df, "X")
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            assert v in result["X"].values

    def test_zscore_removes_extreme_outlier(self, outlier_df):
        result = _remove_outliers_zscore(outlier_df, "X", threshold=2.0)
        assert 1000.0 not in result["X"].values

    def test_zscore_keeps_normal_values(self, outlier_df):
        result = _remove_outliers_zscore(outlier_df, "X", threshold=3.0)
        assert 2.0 in result["X"].values


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: interactive_filter()
# ─────────────────────────────────────────────────────────────────────────────

class TestInteractiveFilter:
    """Tests for the robust interactive filtering function."""

    @pytest.fixture
    def filter_df(self):
        return pd.DataFrame({
            "Name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "Age":    [25, 30, 22, 35, 28],
            "City":   ["NYC", "LA", "NYC", "Chicago", "LA"],
        })

    def test_exact_value_filter(self, filter_df, monkeypatch):
        """Option 1: exact value filter should return only matching rows."""
        inputs = iter(["1", "City", "NYC", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = interactive_filter(filter_df.copy(), filter_df.copy())
        assert all(result["City"] == "NYC")
        assert len(result) == 2

    def test_numeric_range_filter(self, filter_df, monkeypatch):
        """Option 2: numeric range filter should keep rows within [25, 35]."""
        inputs = iter(["2", "Age", "25", "35", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = interactive_filter(filter_df.copy(), filter_df.copy())
        assert result["Age"].min() >= 25
        assert result["Age"].max() <= 35

    def test_string_contains_filter(self, filter_df, monkeypatch):
        """Option 3: contains filter should keep rows where City contains 'LA'."""
        inputs = iter(["3", "City", "1", "LA", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = interactive_filter(filter_df.copy(), filter_df.copy())
        assert all(result["City"] == "LA")

    def test_reset_filter(self, filter_df, monkeypatch):
        """Option 5: reset should restore original dataset."""
        inputs = iter(["1", "City", "NYC", "5", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = interactive_filter(filter_df.copy(), filter_df.copy())
        assert len(result) == len(filter_df)

    def test_multi_condition_and(self, filter_df, monkeypatch):
        """Option 4: AND filter with Age>=28 AND City==LA should return [Bob, Eve]."""
        inputs = iter(["4", "Age>=28, City==LA", "0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = interactive_filter(filter_df.copy(), filter_df.copy())
        assert len(result) == 2
        assert set(result["Name"].values) == {"Bob", "Eve"}

    def test_returns_dataframe(self, filter_df, monkeypatch):
        """interactive_filter always returns a DataFrame."""
        inputs = iter(["0"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = interactive_filter(filter_df.copy(), filter_df.copy())
        assert isinstance(result, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: _dummy_encode() and _sigmoid()
# ─────────────────────────────────────────────────────────────────────────────

class TestHelperFunctions:
    """Tests for the new helper utilities."""

    def test_dummy_encode_basic(self):
        """Dummy encoding should create binary columns with drop-first."""
        df = pd.DataFrame({
            "Color": ["Red", "Blue", "Green", "Red", "Blue"],
            "Size": [1, 2, 3, 4, 5],
        })
        encoded, new_cols = _dummy_encode(df, ["Color"])
        assert "Color" not in encoded.columns
        assert len(new_cols) == 2  # 3 categories, drop-first → 2
        assert "Size" in encoded.columns

    def test_dummy_encode_preserves_rows(self):
        """Row count should not change after encoding."""
        df = pd.DataFrame({"Cat": ["A", "B", "C", "A"], "X": [1, 2, 3, 4]})
        encoded, _ = _dummy_encode(df, ["Cat"])
        assert len(encoded) == 4

    def test_sigmoid_basic(self):
        """Sigmoid of 0 should be 0.5."""
        result = _sigmoid(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-10

    def test_sigmoid_large_positive(self):
        """Sigmoid of large positive should be near 1."""
        result = _sigmoid(np.array([100.0]))
        assert result[0] > 0.99

    def test_sigmoid_large_negative(self):
        """Sigmoid of large negative should be near 0."""
        result = _sigmoid(np.array([-100.0]))
        assert result[0] < 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: multiple_regression_test()
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleRegressionTest:
    """Tests for the multiple_regression_test function."""

    @pytest.fixture
    def multreg_df(self):
        np.random.seed(42)
        n = 60
        x1 = np.random.normal(50, 10, n)
        x2 = np.random.normal(30, 5, n)
        y = 3.0 * x1 + 2.0 * x2 + 10 + np.random.normal(0, 5, n)
        return pd.DataFrame({
            "Y": y, "X1": x1, "X2": x2,
            "X3": np.random.normal(0, 1, n),  # irrelevant predictor
            "Category": np.random.choice(["A", "B", "C"], n),
        })

    def test_significant_model(self, multreg_df, monkeypatch):
        """Multiple regression with correlated data should be significant."""
        inputs = iter(["Y", "X1, X2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = multiple_regression_test(multreg_df)
        assert result["significant"] is True
        assert result["r_squared"] > 0.8

    def test_result_has_required_keys(self, multreg_df, monkeypatch):
        """Result dict must contain all documented diagnostic keys."""
        inputs = iter(["Y", "X1, X2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = multiple_regression_test(multreg_df)
        for key in ("y_var", "x_vars", "x_vars_encoded", "n", "k",
                    "intercept", "coefficients", "std_errors", "t_stats",
                    "p_values", "std_coefs", "ci_lower", "ci_upper",
                    "r_squared", "adj_r_squared", "f_stat", "f_pvalue",
                    "significant", "vif", "durbin_watson", "jb_stat",
                    "jb_pvalue", "resid_normal", "cond_number"):
            assert key in result, f"Missing key: {key}"

    def test_enforces_minimum_2_predictors(self, multreg_df, monkeypatch):
        """Specifying only 1 predictor should raise ValueError."""
        inputs = iter(["Y", "X1"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        with pytest.raises(ValueError, match="at least 2"):
            multiple_regression_test(multreg_df)

    def test_handles_categorical_predictors(self, multreg_df, monkeypatch):
        """Including a categorical predictor should dummy-encode it."""
        inputs = iter(["Y", "X1, Category"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = multiple_regression_test(multreg_df)
        # Should have more encoded variables than original x_vars
        assert result["k"] >= 2
        assert len(result["x_vars_encoded"]) >= 2

    def test_y_in_x_raises_error(self, multreg_df, monkeypatch):
        """Y variable cannot also be an X variable."""
        inputs = iter(["Y", "X1, Y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        with pytest.raises(ValueError, match="cannot also be"):
            multiple_regression_test(multreg_df)

    def test_vif_present_for_multiple(self, multreg_df, monkeypatch):
        """VIF should be computed for multiple predictors."""
        inputs = iter(["Y", "X1, X2, X3"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = multiple_regression_test(multreg_df)
        assert len(result["vif"]) == 3
        for v in result["vif"].values():
            assert v >= 1.0  # VIF is always ≥ 1

    def test_durbin_watson_range(self, multreg_df, monkeypatch):
        """Durbin-Watson should be between 0 and 4."""
        inputs = iter(["Y", "X1, X2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = multiple_regression_test(multreg_df)
        assert 0 <= result["durbin_watson"] <= 4

    def test_confidence_intervals(self, multreg_df, monkeypatch):
        """Lower CI should be less than upper CI for all coefficients."""
        inputs = iter(["Y", "X1, X2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = multiple_regression_test(multreg_df)
        for var in result["x_vars_encoded"]:
            assert result["ci_lower"][var] < result["ci_upper"][var]

    def test_not_enough_columns(self):
        """A DataFrame with only 1 numeric column should return empty dict."""
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
        result = multiple_regression_test(df)
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
#  Tests: logistic_regression()
# ─────────────────────────────────────────────────────────────────────────────

class TestLogisticRegression:
    """Tests for the logistic_regression function."""

    @pytest.fixture
    def logistic_df_numeric(self):
        """DataFrame with binary 0/1 outcome and separable data."""
        np.random.seed(42)
        n = 100
        x1 = np.concatenate([np.random.normal(2, 1, n // 2),
                             np.random.normal(-2, 1, n // 2)])
        x2 = np.concatenate([np.random.normal(1, 0.5, n // 2),
                             np.random.normal(-1, 0.5, n // 2)])
        y = np.array([1] * (n // 2) + [0] * (n // 2))
        return pd.DataFrame({"Y": y, "X1": x1, "X2": x2})

    @pytest.fixture
    def logistic_df_categorical(self):
        """DataFrame with binary categorical outcome."""
        np.random.seed(42)
        n = 80
        x1 = np.concatenate([np.random.normal(5, 1, n // 2),
                             np.random.normal(-5, 1, n // 2)])
        y = ["Pass"] * (n // 2) + ["Fail"] * (n // 2)
        return pd.DataFrame({"Result": y, "Score": x1,
                             "Group": np.random.choice(["A", "B"], n)})

    def test_binary_numeric_outcome(self, logistic_df_numeric, monkeypatch):
        """Logistic regression with 0/1 outcome should work."""
        inputs = iter(["Y", "X1, X2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(logistic_df_numeric)
        assert result["n"] == 100
        assert result["k"] == 2
        assert result["converged"] is True

    def test_binary_categorical_outcome(self, logistic_df_categorical, monkeypatch):
        """Logistic regression with Pass/Fail outcome should auto-encode."""
        inputs = iter(["Result", "Score"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(logistic_df_categorical)
        assert result["y_var"] == "Result"
        assert "0" in str(result["y_labels"]) or "Fail" in str(result["y_labels"])

    def test_auto_encodes_categorical_predictors(self, logistic_df_categorical, monkeypatch):
        """Categorical X variables should be dummy-encoded."""
        inputs = iter(["Result", "Score, Group"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(logistic_df_categorical)
        assert result["k"] >= 2  # Score + at least 1 dummy for Group

    def test_result_has_required_keys(self, logistic_df_numeric, monkeypatch):
        """Result dict must contain all documented keys."""
        inputs = iter(["Y", "X1"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(logistic_df_numeric)
        for key in ("y_var", "x_vars", "x_vars_encoded", "y_labels",
                    "n", "k", "converged", "intercept", "coefficients",
                    "std_errors", "z_stats", "p_values", "odds_ratios",
                    "ci_lower", "ci_upper", "log_likelihood", "pseudo_r2",
                    "aic", "bic", "lr_stat", "lr_pvalue", "significant",
                    "confusion_matrix", "accuracy", "precision",
                    "recall", "f1_score"):
            assert key in result, f"Missing key: {key}"

    def test_significant_model(self, logistic_df_numeric, monkeypatch):
        """Model with separable data should be significant."""
        inputs = iter(["Y", "X1, X2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(logistic_df_numeric)
        assert result["significant"] is True

    def test_non_significant_model(self, monkeypatch):
        """Model with random data should not be significant."""
        np.random.seed(99)
        df = pd.DataFrame({
            "Y": np.random.choice([0, 1], 50),
            "X": np.random.normal(0, 1, 50),
        })
        inputs = iter(["Y", "X"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(df)
        assert result["significant"] is False

    def test_non_binary_y_raises_error(self, monkeypatch):
        """A Y variable with more than 2 unique values should raise ValueError."""
        df = pd.DataFrame({
            "Y": ["A", "B", "C", "A", "B"],
            "X": [1, 2, 3, 4, 5],
        })
        inputs = iter(["Y", "X"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        with pytest.raises(ValueError, match="exactly 2"):
            logistic_regression(df)

    def test_single_predictor_works(self, logistic_df_numeric, monkeypatch):
        """Logistic regression with a single predictor should work."""
        inputs = iter(["Y", "X1"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(logistic_df_numeric)
        assert result["k"] == 1
        assert len(result["coefficients"]) == 1

    def test_classification_metrics_valid(self, logistic_df_numeric, monkeypatch):
        """Accuracy, precision, recall, F1 should all be in [0, 1]."""
        inputs = iter(["Y", "X1, X2"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(logistic_df_numeric)
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            assert 0.0 <= result[metric] <= 1.0, f"{metric} = {result[metric]}"

    def test_confusion_matrix_sums_to_n(self, logistic_df_numeric, monkeypatch):
        """Confusion matrix elements should sum to n."""
        inputs = iter(["Y", "X1"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = logistic_regression(logistic_df_numeric)
        cm = result["confusion_matrix"]
        assert cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"] == result["n"]

    def test_no_binary_columns_returns_empty(self):
        """A DataFrame with no binary columns should return empty dict."""
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": ["x", "y", "z", "w", "v"],
        })
        result = logistic_regression(df)
        assert result == {}

    def test_y_in_x_raises_error(self, logistic_df_numeric, monkeypatch):
        """Y variable cannot also be an X variable."""
        inputs = iter(["Y", "X1, Y"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        with pytest.raises(ValueError, match="cannot also be"):
            logistic_regression(logistic_df_numeric)

