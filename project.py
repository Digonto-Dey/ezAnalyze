import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend — no GUI popups

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Color Helpers  (single source of truth for all terminal coloring)
# ─────────────────────────────────────────────────────────────────────────────

def _c_header(text: str) -> str:
    """Cyan bold — section headers."""
    return f"{Fore.CYAN}{Style.BRIGHT}{text}{Style.RESET_ALL}"

def _c_title(text: str) -> str:
    """White bold — sub-titles / variable names."""
    return f"{Fore.WHITE}{Style.BRIGHT}{text}{Style.RESET_ALL}"

def _c_ok(text: str) -> str:
    """Green — success / normal / significant."""
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def _c_warn(text: str) -> str:
    """Yellow — warnings / skipped / info."""
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

def _c_err(text: str) -> str:
    """Red — errors / failures / not-significant."""
    return f"{Fore.RED}{text}{Style.RESET_ALL}"

def _c_info(text: str) -> str:
    """Magenta — non-normal / special info."""
    return f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"

def _c_accent(text: str) -> str:
    """Blue — numeric values / accents."""
    return f"{Fore.BLUE}{Style.BRIGHT}{text}{Style.RESET_ALL}"

def _c_dim(text: str) -> str:
    """Dim white — secondary info."""
    return f"{Style.DIM}{text}{Style.RESET_ALL}"

def _banner(title: str, width: int = 54) -> str:
    """Return a styled banner string for section headings."""
    bar = "━" * width
    return (
        f"\n{_c_header(bar)}\n"
        f"  {_c_header(title)}\n"
        f"{_c_header(bar)}"
    )

def _print_banner(title: str, width: int = 54) -> None:
    print(_banner(title, width))

def _print_result_row(label: str, value, width: int = 14) -> None:
    """Print a formatted key=value row."""
    label_str = f"{label:<{width}}"
    print(f"    {_c_dim(label_str)} = {value}")


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

ALPHA = 0.05  # Significance level for hypothesis tests
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = BASE_DIR / "plots"
EXPORTS_DIR = BASE_DIR / "exports"

# Thresholds for high-cardinality categorical handling
MAX_BAR_CATEGORIES    = 15   # Max categories shown in bar charts
MAX_FREQ_DISPLAY      = 20   # Max categories shown in frequency tables / reports
SKIP_CHART_THRESHOLD  = 500  # Skip bar chart entirely if unique values exceed this

# Shapiro-Wilk sample limit (scipy limitation)
SHAPIRO_MAX_N = 5000

# Create directories if they don't exist
for _d in [DATA_DIR, REPORTS_DIR, PLOTS_DIR, EXPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

MENU_TEXT = f"""
{Fore.CYAN}{Style.BRIGHT}         ezpzAnalyze — Research Data Analysis Toolkit{Style.RESET_ALL}

{Fore.WHITE}   1.  Dataset Summary
   2.  Missing Value Analysis
   3.  Descriptive Statistics
   4.  Frequency Distribution
   5.  Generate Visualizations
   6.  Correlation Analysis
   7.  Independent t-Test
   8.  Chi-Square Test
   9.  Normality Test (Shapiro-Wilk / Anderson-Darling)
  10.  Linear Regression  (Simple & Multiple)
  11.  One-Way ANOVA  (+ Levene + Tukey HSD)
  12.  Data Cleaning Tools
  13.  Interactive Filtering{Style.RESET_ALL}
{Fore.YELLOW}  14.  Full Automated Analysis
  15.  Generate Report
  16.  Export Results{Style.RESET_ALL}
{Fore.RED}   0.  Exit{Style.RESET_ALL}

{Fore.CYAN}===================================================={Style.RESET_ALL}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    for directory in (DATA_DIR, REPORTS_DIR, PLOTS_DIR, EXPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    """Sanitize a string for safe use as a filename component."""
    return re.sub(r'[^\w\-]', '_', name.lower()).strip('_')


def _auto_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to convert object columns that look like dates to datetime64.
    Only converts columns where ≥80 % of non-null values parse successfully.
    """
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().sum() / len(sample)
                if success_rate >= 0.80:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            pass
    return df


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return names of numeric columns."""
    return df.select_dtypes(include="number").columns.tolist()


def _cat_cols(df: pd.DataFrame) -> list[str]:
    """Return names of categorical (object / category) columns."""
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _pick_column(
    df: pd.DataFrame,
    prompt: str,
    kind: str = "any",
    exclude: list[str] | None = None,
) -> str:
    """
    Interactively prompt the user to choose a column.

    Args:
        df:      Source DataFrame.
        prompt:  Prompt label.
        kind:    'numeric', 'categorical', or 'any'.
        exclude: Column names to exclude from the list.

    Returns:
        The chosen column name (validated to exist in df).

    Raises:
        KeyError: If the entered column is not found.
    """
    exclude = exclude or []
    if kind == "numeric":
        cols = [c for c in _numeric_cols(df) if c not in exclude]
        label = "Numeric"
    elif kind == "categorical":
        cols = [c for c in _cat_cols(df) if c not in exclude]
        label = "Categorical"
    else:
        cols = [c for c in df.columns.tolist() if c not in exclude]
        label = "Available"

    print(f"\n  {_c_title(label + ' columns:')} {', '.join(cols)}")
    chosen = input(f"  {prompt}: ").strip()
    if chosen not in df.columns:
        raise KeyError(f"Column '{chosen}' not found in the dataset.")
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
#  Module 1: Dataset Import
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and return a pandas DataFrame.

    Args:
        filepath: Path to the CSV file (absolute or relative).

    Returns:
        pd.DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file is not a .csv file or cannot be parsed.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if path.suffix.lower() != ".csv":
        raise ValueError(
            f"Unsupported file format '{path.suffix}'. Only .csv files are accepted."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The CSV file is empty — no data to analyze.")

    # Auto-detect and parse date columns
    df = _auto_parse_dates(df)

    rows, cols = df.shape
    print(f"\n{_c_ok('✅  Dataset Loaded Successfully')}")
    print(f"    Rows:    {_c_accent(str(rows))}")
    print(f"    Columns: {_c_accent(str(cols))}")

    # Report any auto-detected date columns
    date_cols = df.select_dtypes(include="datetime64").columns.tolist()
    if date_cols:
        print(f"    {_c_info('📅  Auto-detected date columns: ' + ', '.join(date_cols))}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Module 2: Dataset Summary
# ─────────────────────────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> str:
    """
    Print and return a textual overview of the dataset.

    Displays:
        - Number of observations (rows)
        - Number of variables (columns)
        - Variable names
        - Data type classification (Numeric / Categorical / Datetime)
    """
    rows, cols = df.shape
    lines = [
        "",
        _c_header("━" * 54),
        f"  {_c_header('DATASET SUMMARY')}",
        _c_header("━" * 54),
        f"  Observations : {_c_accent(str(rows))}",
        f"  Variables    : {_c_accent(str(cols))}",
        "",
        f"  {_c_title('Variable List:')}",
    ]

    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            kind = _c_ok("Numeric")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            kind = _c_info("Datetime")
        else:
            kind = _c_warn("Categorical")
        lines.append(f"    • {col:<22s} → {kind} ({_c_dim(str(dtype))})")

    lines.append(_c_header("━" * 54))
    summary_text = "\n".join(lines)
    print(summary_text)
    return summary_text


# ─────────────────────────────────────────────────────────────────────────────
#  Module 3: Missing Value Analysis
# ─────────────────────────────────────────────────────────────────────────────

def missing_value_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze missing values in every column.

    Returns:
        dict mapping column name → (missing_count, missing_percentage).
        Only columns with at least one missing value are included.
    """
    total = len(df)
    result: dict[str, tuple[int, float]] = {}

    _print_banner("MISSING VALUE ANALYSIS")

    any_missing = False
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            pct = (missing_count / total) * 100
            result[col] = (missing_count, round(pct, 2))
            # Severity coloring: ≥50 % → red, ≥20 % → yellow, else white
            if pct >= 50:
                row_color = _c_err
            elif pct >= 20:
                row_color = _c_warn
            else:
                row_color = lambda t: t  # noqa: E731
            print(f"    {row_color(f'{col:<22s}  {missing_count:>5d}  ({pct:.1f}%)')}")
            any_missing = True

    if not any_missing:
        print(f"    {_c_ok('No missing values found. ✓')}")

    print(_c_header("━" * 54))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Module 4: Descriptive Statistics
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for every numeric column.

    Statistics: Count, Mean, Median, Mode, Std Dev (ddof=1),
    Variance (ddof=1), Min, Max, Range, Skewness, Kurtosis.

    Returns:
        pd.DataFrame with statistics as rows and variables as columns.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print(f"\n{_c_warn('⚠  No numeric columns found for descriptive statistics.')}")
        return pd.DataFrame()

    stats_dict: dict[str, dict] = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        mode_vals = series.mode()
        mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan

        stats_dict[col] = {
            "Count":    int(series.count()),
            "Mean":     round(series.mean(), 4),
            "Median":   round(series.median(), 4),
            "Mode":     round(mode_val, 4) if pd.notna(mode_val) else "N/A",
            "Std Dev":  round(series.std(ddof=1), 4),
            "Variance": round(series.var(ddof=1), 4),
            "Min":      round(series.min(), 4),
            "Max":      round(series.max(), 4),
            "Range":    round(series.max() - series.min(), 4),
            "Skewness": round(float(series.skew()), 4),
            "Kurtosis": round(float(series.kurtosis()), 4),
        }

    result_df = pd.DataFrame(stats_dict)

    _print_banner("DESCRIPTIVE STATISTICS")
    for col, col_stats in stats_dict.items():
        print(f"\n  {_c_title('Variable: ' + col)}")
        for stat_name, value in col_stats.items():
            _print_result_row(stat_name, value)
    print(_c_header("━" * 54))

    return result_df


# ─────────────────────────────────────────────────────────────────────────────
#  Module 5: Frequency Distribution
# ─────────────────────────────────────────────────────────────────────────────

def frequency_distribution(df: pd.DataFrame) -> dict:
    """
    Compute frequency tables for every categorical (non-numeric, non-datetime) column.

    Returns:
        dict mapping column name → pd.DataFrame with columns
        ['Value', 'Frequency', 'Percentage'].
    """
    cat_cols_list = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols_list:
        print(f"\n{_c_warn('⚠  No categorical columns found for frequency distribution.')}")
        return {}

    result: dict[str, pd.DataFrame] = {}

    _print_banner("FREQUENCY DISTRIBUTION")

    for col in cat_cols_list:
        counts = df[col].value_counts(dropna=False)
        total = counts.sum()
        pct = (counts / total * 100).round(2)

        freq_df = pd.DataFrame({
            "Value":      counts.index.astype(str),
            "Frequency":  counts.values,
            "Percentage": pct.values,
        })
        result[col] = freq_df

        n_unique = len(counts)
        print(f"\n  {_c_title('Variable: ' + col)} " + _c_dim(f"({n_unique} unique values)"))

        display_limit = min(MAX_FREQ_DISPLAY, n_unique)
        for _, row in freq_df.head(display_limit).iterrows():
            print(f"    {str(row['Value']):<22s}  {int(row['Frequency']):>5d}  ({row['Percentage']:.1f}%)")

        if n_unique > MAX_FREQ_DISPLAY:
            remaining = n_unique - MAX_FREQ_DISPLAY
            print("    " + _c_warn(f"... and {remaining} more unique values (not shown)"))

    print(_c_header("━" * 54))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Module 6: Data Visualization
# ─────────────────────────────────────────────────────────────────────────────

def generate_visualizations(df: pd.DataFrame) -> list[str]:
    """
    Generate and save visualizations for all columns.

    - Histogram + Boxplot for each numeric column
    - Bar chart for each categorical column (top 15 categories max)
    - Skips categorical columns with >500 unique values
    - Skips datetime columns

    Returns:
        List of file paths for saved plots.
    """
    ensure_directories()
    saved_files: list[str] = []

    numeric_cols_list = df.select_dtypes(include="number").columns.tolist()
    cat_cols_list = df.select_dtypes(include=["object", "category"]).columns.tolist()

    _print_banner("GENERATING VISUALIZATIONS")

    # --- Histograms ---
    for col in numeric_cols_list:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            data = df[col].dropna()
            ax.hist(data, bins="auto", color="#4A90D9", edgecolor="#2C3E50", alpha=0.85)
            ax.set_title(f"Histogram — {col}", fontsize=14, fontweight="bold")
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            filepath = PLOTS_DIR / f"histogram_{_safe_filename(col)}.png"
            fig.tight_layout()
            fig.savefig(filepath, dpi=150)
            plt.close(fig)
            saved_files.append(str(filepath))
            print(f"    {_c_ok('✓ Saved: ' + filepath.name)}")
        except Exception as e:
            print("    " + _c_warn(f"⊘ Skipped histogram for {col}: {e}"))

    # --- Boxplots ---
    for col in numeric_cols_list:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            data = df[col].dropna()
            bp = ax.boxplot(data, patch_artist=True, vert=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#5DADE2")
                patch.set_alpha(0.7)
            ax.set_title(f"Boxplot — {col}", fontsize=14, fontweight="bold")
            ax.set_ylabel(col, fontsize=11)
            ax.set_xticklabels([col])
            ax.grid(axis="y", alpha=0.3)
            filepath = PLOTS_DIR / f"boxplot_{_safe_filename(col)}.png"
            fig.tight_layout()
            fig.savefig(filepath, dpi=150)
            plt.close(fig)
            saved_files.append(str(filepath))
            print(f"    {_c_ok('✓ Saved: ' + filepath.name)}")
        except Exception as e:
            print("    " + _c_warn(f"⊘ Skipped boxplot for {col}: {e}"))

    # --- Bar Charts (with high-cardinality protection) ---
    for col in cat_cols_list:
        n_unique = df[col].nunique()

        if n_unique > SKIP_CHART_THRESHOLD:
            print("    " + _c_warn(f"⊘ Skipped: {col} ({n_unique} unique values — too many for a bar chart)"))
            continue

        try:
            counts = df[col].value_counts()

            if n_unique > MAX_BAR_CATEGORIES:
                counts = counts.head(MAX_BAR_CATEGORIES)
                subtitle = f"(Top {MAX_BAR_CATEGORIES} of {n_unique} categories shown)"
            else:
                subtitle = ""

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(counts)))
            ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="#2C3E50")
            title = f"Bar Chart — {col}"
            if subtitle:
                title += f"\n{subtitle}"
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=45, ha="right")
            filepath = PLOTS_DIR / f"barchart_{_safe_filename(col)}.png"
            fig.tight_layout()
            fig.savefig(filepath, dpi=150)
            plt.close(fig)
            saved_files.append(str(filepath))
            print(f"    {_c_ok('✓ Saved: ' + filepath.name)}")
        except Exception as e:
            print("    " + _c_warn(f"⊘ Skipped bar chart for {col}: {e}"))

    print(f"\n    Total plots saved: {_c_accent(str(len(saved_files)))}")
    print(_c_header("━" * 54))
    return saved_files


# ─────────────────────────────────────────────────────────────────────────────
#  Module 7: Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for numeric columns and
    generate a correlation heatmap saved to plots/.

    Also prints per-pair r-values and p-values.

    Returns:
        pd.DataFrame — the Pearson correlation matrix.
    """
    numeric_df = df.select_dtypes(include="number").dropna()

    if numeric_df.shape[1] < 2:
        print(f"\n{_c_warn('⚠  Need at least 2 numeric columns for correlation analysis.')}")
        return pd.DataFrame()

    corr_matrix = numeric_df.corr(method="pearson")
    cols = numeric_df.columns.tolist()

    _print_banner("CORRELATION ANALYSIS (Pearson)")

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, p = stats.pearsonr(numeric_df[cols[i]], numeric_df[cols[j]])
            if p < ALPHA:
                sig_text = _c_ok("Significant ✓")
            else:
                sig_text = _c_dim("Not Significant")
            print(f"    {cols[i]} vs {cols[j]}")
            print(f"      r = {_c_accent(f'{r:.4f}')},  p = {_c_accent(f'{p:.4f}')}  →  {sig_text}")

    # --- Heatmap ---
    ensure_directories()
    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) * 0.8)))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    for (row, col_idx), val in np.ndenumerate(corr_matrix.values):
        ax.text(col_idx, row, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(val) > 0.6 else "black")

    ax.set_title("Pearson Correlation Heatmap", pad=20, fontsize=14, fontweight="bold")
    heatmap_path = PLOTS_DIR / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"\n    {_c_ok('✓ Heatmap saved: ' + heatmap_path.name)}")
    print(_c_header("━" * 54))

    return corr_matrix


# ─────────────────────────────────────────────────────────────────────────────
#  Module 8a: Hypothesis Testing — Independent t-Test
# ─────────────────────────────────────────────────────────────────────────────

def independent_ttest(df: pd.DataFrame) -> dict:
    """
    Perform an Independent Samples t-Test.

    Prompts the user to select:
        - A numeric (dependent) variable
        - A categorical (grouping) variable with exactly 2 groups

    Returns:
        dict with keys: 'variable', 'grouping', 'group1', 'group2',
        't_stat', 'p_value', 'significant'.
    """
    numeric_cols_list = _numeric_cols(df)
    cat_cols_list = _cat_cols(df)

    if not numeric_cols_list:
        print(f"\n{_c_warn('⚠  No numeric columns available for a t-test.')}")
        return {}
    if not cat_cols_list:
        print(f"\n{_c_warn('⚠  No categorical columns available as a grouping variable.')}")
        return {}

    numeric_var = _pick_column(df, "Enter the numeric (dependent) variable", kind="numeric")
    group_var   = _pick_column(df, "Enter the grouping variable (2 groups)", kind="categorical")

    subset = df[[numeric_var, group_var]].dropna()
    groups = subset[group_var].unique()

    if len(groups) != 2:
        raise ValueError(
            f"Grouping variable '{group_var}' has {len(groups)} groups — "
            f"exactly 2 are required for an independent t-test."
        )

    group1_data = subset[subset[group_var] == groups[0]][numeric_var]
    group2_data = subset[subset[group_var] == groups[1]][numeric_var]

    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    significant = p_value < ALPHA

    result = {
        "variable":    numeric_var,
        "grouping":    group_var,
        "group1":      str(groups[0]),
        "group2":      str(groups[1]),
        "group1_mean": round(float(group1_data.mean()), 4),
        "group2_mean": round(float(group2_data.mean()), 4),
        "t_stat":      round(float(t_stat), 4),
        "p_value":     round(float(p_value), 4),
        "significant": significant,
    }

    _print_banner("INDEPENDENT SAMPLES t-TEST")
    print(f"    {groups[0]} vs {groups[1]}  on  {numeric_var}")
    _print_result_row(f"{groups[0]} Mean", result["group1_mean"])
    _print_result_row(f"{groups[1]} Mean", result["group2_mean"])
    _print_result_row("t", result["t_stat"])
    _print_result_row("p", result["p_value"])
    if significant:
        print(f"    Result: {_c_ok('✅ Significant Difference')}")
    else:
        print(f"    Result: {_c_err('✗ No Significant Difference')}")
    print(_c_header("━" * 54))

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Module 8b: Hypothesis Testing — Chi-Square Test
# ─────────────────────────────────────────────────────────────────────────────

def chi_square_test(df: pd.DataFrame) -> dict:
    """
    Perform a Chi-Square Test of Independence.

    Prompts the user to select two categorical variables.

    Returns:
        dict with keys: 'var1', 'var2', 'chi2', 'p_value',
        'dof', 'significant'.
    """
    cat_cols_list = _cat_cols(df)

    if len(cat_cols_list) < 2:
        print(f"\n{_c_warn('⚠  Need at least 2 categorical columns for a Chi-Square test.')}")
        return {}

    var1 = _pick_column(df, "Enter the first categorical variable",  kind="categorical")
    var2 = _pick_column(df, "Enter the second categorical variable", kind="categorical")

    subset = df[[var1, var2]].dropna()
    contingency = pd.crosstab(subset[var1], subset[var2])

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    significant = p_value < ALPHA

    result = {
        "var1":        var1,
        "var2":        var2,
        "chi2":        round(float(chi2), 4),
        "p_value":     round(float(p_value), 4),
        "dof":         int(dof),
        "significant": significant,
    }

    _print_banner("CHI-SQUARE TEST OF INDEPENDENCE")
    print(f"    {var1} vs {var2}")
    _print_result_row("χ²", result["chi2"])
    _print_result_row("p", result["p_value"])
    _print_result_row("df", result["dof"])
    if significant:
        print(f"    Result: {_c_ok('✅ Significant Association')}")
    else:
        print(f"    Result: {_c_err('✗ No Significant Association')}")
    print(_c_header("━" * 54))

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Module 9: Normality Testing (Shapiro-Wilk + Anderson-Darling)
# ─────────────────────────────────────────────────────────────────────────────

def normality_test(df: pd.DataFrame) -> dict:
    """
    Perform normality testing on every numeric column.

    Strategy (robust for any dataset size):
        • n < 3             → skip (not enough data)
        • 3 ≤ n ≤ 5000      → Shapiro-Wilk  (exact test, W-statistic)
        • n > 5000          → Anderson-Darling at α=0.05 significance level
          (Shapiro-Wilk is unreliable / not supported above 5000 by scipy)

    Additionally computes:
        • Skewness  (|skew| < 1 is a rough normality indicator)
        • Kurtosis  (excess kurtosis ≈ 0 for normal data)

    Generates a Q-Q plot for every tested column and saves to plots/.

    Returns:
        dict mapping column name → {
            'test': str,        # 'shapiro' or 'anderson'
            'stat': float,      # W (Shapiro) or A² statistic (Anderson)
            'p_value': float,   # p-value (Shapiro) or interpolated p (Anderson)
            'normal': bool,
            'skewness': float,
            'kurtosis': float,
        }
    """
    numeric_cols_list = _numeric_cols(df)

    if not numeric_cols_list:
        print(f"\n{_c_warn('⚠  No numeric columns found for normality testing.')}")
        return {}

    results: dict[str, dict] = {}

    _print_banner("NORMALITY TEST (Shapiro-Wilk / Anderson-Darling)")

    for col in numeric_cols_list:
        data = df[col].dropna()

        if len(data) < 3:
            print(f"    {_c_warn(col + ': Skipped (need ≥ 3 observations)')}")
            continue

        try:
            skewness = round(float(data.skew()), 4)
            kurt     = round(float(data.kurtosis()), 4)

            if len(data) <= SHAPIRO_MAX_N:
                # ── Shapiro-Wilk ──────────────────────────────────────────
                w_stat, p_value = stats.shapiro(data)
                is_normal = p_value > ALPHA
                test_used = "shapiro"
                stat_val  = round(float(w_stat), 6)
                p_val_out = round(float(p_value), 6)
            else:
                # ── Anderson-Darling (large samples) ──────────────────────
                ad_result = stats.anderson(data, dist="norm")
                # significance_level index: [15, 10, 5, 2.5, 1] %
                # We use the 5 % level (index 2)
                sig_idx   = 2  # 5 %
                crit_val  = ad_result.critical_values[sig_idx]
                a2_stat   = float(ad_result.statistic)
                is_normal = a2_stat < crit_val

                # Interpolate a p-value from tabulated critical values
                # (approximate — Anderson does not return a p directly)
                sig_levels = np.array([0.15, 0.10, 0.05, 0.025, 0.01])
                crits      = ad_result.critical_values
                try:
                    p_approx = float(np.interp(a2_stat, crits, sig_levels))
                except Exception:
                    p_approx = float("nan")

                test_used = "anderson"
                stat_val  = round(a2_stat, 6)
                p_val_out = round(p_approx, 6)

            results[col] = {
                "test":     test_used,
                "stat":     stat_val,
                "p_value":  p_val_out,
                "normal":   is_normal,
                "skewness": skewness,
                "kurtosis": kurt,
            }

            # ── Print result row ───────────────────────────────────────────
            print(f"\n  {_c_title(col)}")
            stat_label = "W" if test_used == "shapiro" else "A²"
            test_label = "Shapiro-Wilk" if test_used == "shapiro" else "Anderson-Darling"
            print(f"    Test:     {test_label}  (n = {len(data)})")
            _print_result_row(stat_label, stat_val)
            _print_result_row("p", p_val_out)
            _print_result_row("Skewness", skewness)
            _print_result_row("Kurtosis", kurt)

            if is_normal:
                verdict = _c_ok("✅ Normal distribution (fail to reject H₀)")
            else:
                verdict = _c_info("✗ Not normal (reject H₀)")
            print(f"    {verdict}")

            # ── Q-Q Plot ───────────────────────────────────────────────────
            try:
                ensure_directories()
                sample = data if len(data) <= 5000 else data.sample(5000, random_state=42)
                fig, ax = plt.subplots(figsize=(6, 5))
                (osm, osr), (slope, intercept, _r) = stats.probplot(sample, dist="norm")
                ax.scatter(osm, osr, alpha=0.4, s=12, color="#4A90D9", label="Data")
                ax.plot(osm, slope * np.array(osm) + intercept,
                        color="#E74C3C", linewidth=1.5, label="Normal line")
                ax.set_title(f"Q-Q Plot — {col}", fontsize=12, fontweight="bold")
                ax.set_xlabel("Theoretical Quantiles", fontsize=10)
                ax.set_ylabel("Sample Quantiles", fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                qq_path = PLOTS_DIR / f"qqplot_{_safe_filename(col)}.png"
                fig.tight_layout()
                fig.savefig(qq_path, dpi=150)
                plt.close(fig)
                print(f"    {_c_ok('✓ Q-Q plot saved: ' + qq_path.name)}")
            except Exception as plot_err:
                print("    " + _c_warn(f"⊘ Q-Q plot failed: {plot_err}"))

        except Exception as col_err:
            print("    " + _c_err(f"❌  Error testing {col}: {col_err}"))
            continue

    print(f"\n{_c_header('━' * 54)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Module 10: Linear Regression  (Simple & Multiple)
# ─────────────────────────────────────────────────────────────────────────────

def _vif(X: np.ndarray) -> np.ndarray:
    """
    Compute Variance Inflation Factors for columns of design matrix X.
    X must include an intercept column (all ones) as the first column.
    Returns VIF for each predictor (intercept excluded).
    """
    n_cols = X.shape[1]
    vif_vals = np.full(n_cols - 1, np.nan)  # exclude intercept
    for i in range(1, n_cols):
        y_i   = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        # R² of regressing column i on all others
        _, res, rank, _ = np.linalg.lstsq(X_rest, y_i, rcond=None)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2 = 1 - (res[0] / ss_tot) if (len(res) > 0 and ss_tot > 0) else 0.0
        r2 = min(max(r2, 0.0), 1.0 - 1e-12)
        vif_vals[i - 1] = 1.0 / (1.0 - r2)
    return vif_vals


def linear_regression(df: pd.DataFrame) -> dict:
    """
    Perform linear regression between numeric variables.

    The user chooses:
        - One Y (dependent) variable
        - One OR MORE X (independent) variables

    If one X: simple linear regression (scipy.stats.linregress + OLS check).
    If multiple X: multiple linear regression via numpy OLS (X'X)⁻¹X'y.

    Additional outputs:
        - R², Adjusted R², F-statistic, F p-value
        - Per-coefficient t-tests and p-values
        - VIF (multi-collinearity check, multiple only)
        - Residual plot saved to plots/

    Returns:
        dict with regression results.
    """
    numeric_cols_list = _numeric_cols(df)

    if len(numeric_cols_list) < 2:
        print(f"\n{_c_warn('⚠  Need at least 2 numeric columns for linear regression.')}")
        return {}

    y_var = _pick_column(df, "Enter the dependent variable (Y)", kind="numeric")

    print(f"\n  {_c_title('Available X variables:')} {', '.join(c for c in numeric_cols_list if c != y_var)}")
    raw = input("  Enter one or more independent variable(s) (X), comma-separated: ").strip()
    x_vars = [v.strip() for v in raw.split(",") if v.strip()]

    if not x_vars:
        raise ValueError("At least one independent variable must be specified.")

    for v in x_vars:
        if v not in df.columns:
            raise KeyError(f"Column '{v}' not found in the dataset.")
        if not pd.api.types.is_numeric_dtype(df[v]):
            raise ValueError(f"Column '{v}' is not numeric.")
    if y_var in x_vars:
        raise ValueError("The dependent variable Y cannot also be an independent variable.")

    # Remove rows with any NaN in Y or any X
    cols_used = [y_var] + x_vars
    clean = df[cols_used].dropna()

    if len(clean) < len(x_vars) + 2:
        raise ValueError(
            f"Need at least {len(x_vars) + 2} complete observations for regression "
            f"with {len(x_vars)} predictor(s)."
        )

    y  = clean[y_var].values.astype(float)
    Xm = clean[x_vars].values.astype(float)

    # Design matrix with intercept
    ones  = np.ones((len(y), 1))
    X_des = np.hstack([ones, Xm])  # shape (n, k+1)

    n = len(y)
    k = len(x_vars)  # number of predictors

    # OLS: β = (X'X)⁻¹ X'y
    try:
        XtX_inv = np.linalg.inv(X_des.T @ X_des)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X_des.T @ X_des)

    beta    = XtX_inv @ X_des.T @ y
    y_hat   = X_des @ beta
    resid   = y - y_hat

    ss_res  = float(np.sum(resid ** 2))
    ss_tot  = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2    = 1.0 - (1.0 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    mse       = ss_res / (n - k - 1) if n > k + 1 else float("nan")

    # Coefficient standard errors and t-statistics
    var_beta = mse * XtX_inv if not np.isnan(mse) else np.full_like(XtX_inv, np.nan)
    se_beta  = np.sqrt(np.diag(var_beta))
    t_stats  = beta / se_beta
    df_resid = n - k - 1
    p_vals   = [2 * (1 - stats.t.cdf(abs(t), df_resid)) for t in t_stats]

    # F-statistic
    ss_model = ss_tot - ss_res
    if ss_model > 0 and mse > 0:
        f_stat  = (ss_model / k) / mse
        f_pval  = 1.0 - stats.f.cdf(f_stat, k, n - k - 1)
    else:
        f_stat, f_pval = float("nan"), float("nan")

    # Simple regression also via scipy for cross-check
    r_value = float(np.sqrt(r_squared)) * (1 if beta[1] >= 0 else -1) if k == 1 else float("nan")

    # VIF (only meaningful with ≥2 predictors)
    vif_vals = _vif(X_des) if k >= 2 else np.array([1.0])

    result = {
        "y_var":        y_var,
        "x_vars":       x_vars,
        "n":            n,
        "k":            k,
        "intercept":    round(float(beta[0]), 6),
        "coefficients": {x_vars[i]: round(float(beta[i + 1]), 6) for i in range(k)},
        "r_value":      round(r_value, 6) if not np.isnan(r_value) else None,
        "r_squared":    round(r_squared, 6),
        "adj_r_squared": round(adj_r2, 6) if not np.isnan(adj_r2) else None,
        "f_stat":       round(f_stat, 4) if not np.isnan(f_stat) else None,
        "f_pvalue":     round(f_pval, 6) if not np.isnan(f_pval) else None,
        "mse":          round(mse, 6) if not np.isnan(mse) else None,
        "significant":  (f_pval < ALPHA) if not np.isnan(f_pval) else False,
        "vif":          {x_vars[i]: round(float(vif_vals[i]), 4) for i in range(k)} if k >= 2 else {},
    }

    _print_banner("LINEAR REGRESSION")
    eq_parts = " + ".join(f"{round(beta[i+1], 4)}·{x_vars[i]}" for i in range(k))
    print(f"    {_c_title('Model:')}  Y = {round(beta[0], 4)} + {eq_parts}")
    print(f"    n = {n},  predictors = {k}")
    _print_result_row("R²", result["r_squared"])
    if result["adj_r_squared"] is not None:
        _print_result_row("Adj R²", result["adj_r_squared"])
    if result["f_stat"] is not None:
        _print_result_row("F-stat", result["f_stat"])
    if result["f_pvalue"] is not None:
        _print_result_row("F p-value", result["f_pvalue"])

    print(f"\n  {_c_title('Coefficients:')}")
    coef_names = ["(Intercept)"] + x_vars
    for i, name in enumerate(coef_names):
        sig = _c_ok("*") if p_vals[i] < ALPHA else " "
        print(f"    {name:<22s}  coef = {beta[i]:>10.4f}  "
              f"t = {t_stats[i]:>8.4f}  p = {p_vals[i]:.4f}  {sig}")

    if k >= 2:
        print(f"\n  {_c_title('VIF (multicollinearity):')}")
        for xv, v in result["vif"].items():
            flag = _c_warn(" ⚠ HIGH") if v > 10 else ""
            print(f"    {xv:<22s}  VIF = {v:.4f}{flag}")

    if result["significant"]:
        print(f"\n    {_c_ok('✅ Model is statistically significant (F-test)')}")
    else:
        print(f"\n    {_c_err('✗ Model is NOT statistically significant (F-test)')}")

    # ── Residual Plot ──────────────────────────────────────────────────────────
    try:
        ensure_directories()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Fitted vs Residuals
        axes[0].scatter(y_hat, resid, alpha=0.5, color="#4A90D9", edgecolors="#2C3E50", s=25)
        axes[0].axhline(0, color="#E74C3C", linewidth=1.5, linestyle="--")
        axes[0].set_title("Residuals vs Fitted", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Fitted values", fontsize=10)
        axes[0].set_ylabel("Residuals", fontsize=10)
        axes[0].grid(alpha=0.3)

        # Right: Scatter (only for simple regression)
        if k == 1:
            x_arr = clean[x_vars[0]].values
            axes[1].scatter(x_arr, y, alpha=0.5, color="#4A90D9", edgecolors="#2C3E50", s=25)
            x_line = np.linspace(x_arr.min(), x_arr.max(), 200)
            y_line = beta[0] + beta[1] * x_line
            axes[1].plot(x_line, y_line, color="#E74C3C", linewidth=2,
                         label=f"y = {beta[0]:.4f} + {beta[1]:.4f}·x  (R² = {r_squared:.4f})")
            axes[1].set_title(f"Regression — {y_var} vs {x_vars[0]}", fontsize=12, fontweight="bold")
            axes[1].set_xlabel(x_vars[0], fontsize=10)
            axes[1].set_ylabel(y_var, fontsize=10)
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)
        else:
            # Multiple: actual vs predicted
            axes[1].scatter(y_hat, y, alpha=0.5, color="#4A90D9", edgecolors="#2C3E50", s=25)
            lo, hi = min(y.min(), y_hat.min()), max(y.max(), y_hat.max())
            axes[1].plot([lo, hi], [lo, hi], color="#E74C3C", linewidth=1.5, linestyle="--", label="Perfect fit")
            axes[1].set_title(f"Actual vs Predicted — {y_var}", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Predicted", fontsize=10)
            axes[1].set_ylabel("Actual", fontsize=10)
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.3)

        safe_xvars = "_".join(_safe_filename(v) for v in x_vars)
        plot_path = PLOTS_DIR / f"regression_{_safe_filename(y_var)}_vs_{safe_xvars}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"    {_c_ok('✓ Regression plot saved: ' + plot_path.name)}")
    except Exception as plot_err:
        print("    " + _c_warn(f"⊘ Plot failed: {plot_err}"))

    print(_c_header("━" * 54))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Module 11: One-Way ANOVA  (+ Levene's test + Tukey HSD post-hoc)
# ─────────────────────────────────────────────────────────────────────────────

def _tukey_hsd(group_data: dict[str, np.ndarray], mse: float, n_harmonic: float) -> list[dict]:
    """
    Perform Tukey's Honestly Significant Difference (HSD) post-hoc test.

    Args:
        group_data:  {group_name: array of values}
        mse:         Mean square error from ANOVA table
        n_harmonic:  Harmonic mean of group sizes (for unbalanced designs)

    Returns:
        List of dicts: {group1, group2, mean_diff, q_stat, p_approx, significant}
    """
    from itertools import combinations
    groups = list(group_data.items())
    k = len(groups)
    df_within = sum(len(v) for v in group_data.values()) - k
    se = np.sqrt(mse / n_harmonic)

    comparisons = []
    for (name_a, data_a), (name_b, data_b) in combinations(groups, 2):
        mean_diff = abs(data_a.mean() - data_b.mean())
        q = mean_diff / se if se > 0 else 0.0
        # Approximate p-value from studentized range distribution
        try:
            from scipy.stats import studentized_range
            p_approx = float(1.0 - studentized_range.cdf(q, k, df_within))
        except (ImportError, Exception):
            # Fallback: approximate via t-distribution
            t_equiv = q / np.sqrt(2)
            p_approx = float(2 * (1 - stats.t.cdf(abs(t_equiv), df_within)))

        comparisons.append({
            "group1":      str(name_a),
            "group2":      str(name_b),
            "mean_diff":   round(float(data_a.mean() - data_b.mean()), 4),
            "q_stat":      round(float(q), 4),
            "p_approx":    round(p_approx, 4),
            "significant": p_approx < ALPHA,
        })
    return comparisons


def one_way_anova(df: pd.DataFrame) -> dict:
    """
    Perform a One-Way ANOVA test.

    Prompts the user to select a numeric variable and a categorical
    grouping variable (with ≥ 2 groups).

    Robustness additions:
        1. Levene's test for homogeneity of variances.
           If Levene is significant → prints a warning but still reports ANOVA.
        2. Welch's ANOVA alternative (reported alongside classic ANOVA).
        3. Tukey HSD post-hoc comparisons when ANOVA is significant.
        4. Group boxplot visualization saved to plots/.

    Returns:
        dict with full ANOVA + post-hoc results.
    """
    numeric_cols_list = _numeric_cols(df)
    cat_cols_list     = _cat_cols(df)

    if not numeric_cols_list:
        print(f"\n{_c_warn('⚠  No numeric columns available for ANOVA.')}")
        return {}
    if not cat_cols_list:
        print(f"\n{_c_warn('⚠  No categorical columns available as a grouping variable.')}")
        return {}

    numeric_var = _pick_column(df, "Enter the numeric (dependent) variable", kind="numeric")
    if not pd.api.types.is_numeric_dtype(df[numeric_var]):
        raise ValueError(f"Column '{numeric_var}' is not numeric.")

    group_var = _pick_column(df, "Enter the grouping variable (≥ 2 groups)", kind="categorical")

    subset = df[[numeric_var, group_var]].dropna()
    groups = subset[group_var].unique()

    if len(groups) < 2:
        raise ValueError(
            f"Grouping variable '{group_var}' has {len(groups)} group(s) — "
            f"at least 2 are required for ANOVA."
        )

    # Filter out groups with fewer than 2 observations (can't compute variance)
    valid_groups = [g for g in groups if subset[subset[group_var] == g].shape[0] >= 2]
    if len(valid_groups) < 2:
        raise ValueError("Need at least 2 groups with ≥ 2 observations each.")

    group_data: dict[str, np.ndarray] = {
        str(g): subset[subset[group_var] == g][numeric_var].values.astype(float)
        for g in valid_groups
    }

    group_arrays = list(group_data.values())
    f_stat, p_value = stats.f_oneway(*group_arrays)
    significant = p_value < ALPHA

    # Group means & sizes
    group_stats = {
        g: {
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "n":    int(len(arr)),
        }
        for g, arr in group_data.items()
    }

    # ── Levene's test ──────────────────────────────────────────────────────
    levene_stat, levene_p = stats.levene(*group_arrays)
    equal_var = levene_p >= ALPHA

    # ── Welch's ANOVA (Brown-Forsythe) ────────────────────────────────────
    # Welch's F using weighted means
    n_i   = np.array([len(a) for a in group_arrays], dtype=float)
    mean_i = np.array([a.mean() for a in group_arrays])
    var_i  = np.array([a.var(ddof=1) if len(a) > 1 else 0.0 for a in group_arrays])
    w_i   = n_i / np.where(var_i > 0, var_i, 1e-12)
    W_tot = w_i.sum()
    grand_mean_w = (w_i * mean_i).sum() / W_tot
    sos_between = (w_i * (mean_i - grand_mean_w) ** 2).sum()
    k = len(group_arrays)
    lambda_val = (3 / (k ** 2 - 1)) * np.sum(((1 - w_i / W_tot) ** 2) / (n_i - 1))
    welch_f = sos_between / (k - 1) / (1 + 2 * (k - 2) / 3 * lambda_val)
    df_denom_welch = 1 / lambda_val if lambda_val > 0 else float("inf")
    welch_p = float(1 - stats.f.cdf(welch_f, k - 1, df_denom_welch))

    # ── ANOVA table values ──────────────────────────────────────────────────
    grand_mean = np.concatenate(group_arrays).mean()
    ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in group_arrays)
    ss_within  = sum(np.sum((a - a.mean()) ** 2) for a in group_arrays)
    n_total    = sum(len(a) for a in group_arrays)
    df_between = k - 1
    df_within  = n_total - k
    ms_between = ss_between / df_between if df_between > 0 else float("nan")
    ms_within  = ss_within / df_within if df_within > 0 else float("nan")
    mse_val    = float(ms_within) if not np.isnan(ms_within) else 1.0

    # ── Tukey HSD ───────────────────────────────────────────────────────────
    tukey_results = []
    if significant and k <= 20:  # HSD with many groups is expensive
        # Harmonic mean of group sizes
        n_sizes = np.array([float(len(a)) for a in group_arrays])
        n_harm  = float(k / np.sum(1.0 / np.where(n_sizes > 0, n_sizes, 1.0)))
        tukey_results = _tukey_hsd(group_data, mse_val, n_harm)

    result = {
        "variable":     numeric_var,
        "grouping":     group_var,
        "groups":       list(group_data.keys()),
        "group_stats":  group_stats,
        "f_stat":       round(float(f_stat), 4),
        "p_value":      round(float(p_value), 4),
        "significant":  significant,
        "levene_stat":  round(float(levene_stat), 4),
        "levene_p":     round(float(levene_p), 4),
        "equal_var":    equal_var,
        "welch_f":      round(float(welch_f), 4),
        "welch_p":      round(float(welch_p), 4),
        "tukey_hsd":    tukey_results,
        "group_means":  {g: s["mean"] for g, s in group_stats.items()},
    }

    _print_banner("ONE-WAY ANOVA")
    print(f"    Variable: {_c_title(numeric_var)}  grouped by: {_c_title(group_var)}")
    print(f"    Number of groups: {_c_accent(str(k))}  |  Total n: {_c_accent(str(n_total))}")

    print(f"\n  {_c_title('Group Descriptives:')}")
    for g, s in group_stats.items():
        print(f"    {g:<22s}  n = {s['n']:<5d}  mean = {s['mean']:.4f}  SD = {s['std']:.4f}")

    print(f"\n  {_c_title('Classic ANOVA:')}")
    _print_result_row("F", result["f_stat"])
    _print_result_row("p", result["p_value"])
    if significant:
        print(f"    {_c_ok('✅ Significant difference between groups (p < α)')}")
    else:
        print(f"    {_c_err('✗ No significant difference between groups')}")

    print(f"\n  {_c_title('Levene Test (equal variances):')}")
    _print_result_row("Levene F", result["levene_stat"])
    _print_result_row("Levene p", result["levene_p"])
    if equal_var:
        print(f"    {_c_ok('✓ Equal variances assumed')}")
    else:
        print(f"    {_c_warn('⚠ Unequal variances — consider Welch ANOVA below')}")

    print(f"\n  {_c_title('Welch ANOVA (robust to unequal variance):')}")
    _print_result_row("Welch F", result["welch_f"])
    _print_result_row("Welch p", result["welch_p"])

    if tukey_results:
        print(f"\n  {_c_title('Tukey HSD Post-Hoc Comparisons:')}")
        for comp in tukey_results:
            sig_str = _c_ok("* Significant") if comp["significant"] else _c_dim("  Not significant")
            print(
                f"    {comp['group1']:<14s} vs {comp['group2']:<14s}"
                f"  Δmean = {comp['mean_diff']:>8.4f}"
                f"  q = {comp['q_stat']:>7.4f}"
                f"  p ≈ {comp['p_approx']:.4f}  {sig_str}"
            )

    # ── Group Boxplot ──────────────────────────────────────────────────────
    try:
        ensure_directories()
        fig, ax = plt.subplots(figsize=(max(8, k + 2), 6))
        box_data = [group_data[g] for g in group_data]
        bp = ax.boxplot(box_data, patch_artist=True, labels=list(group_data.keys()))
        colors = plt.cm.Set2(np.linspace(0, 1, k))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(f"ANOVA — {numeric_var} by {group_var}", fontsize=13, fontweight="bold")
        ax.set_xlabel(group_var, fontsize=11)
        ax.set_ylabel(numeric_var, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        anova_plot_path = PLOTS_DIR / f"anova_{_safe_filename(numeric_var)}_by_{_safe_filename(group_var)}.png"
        fig.tight_layout()
        fig.savefig(anova_plot_path, dpi=150)
        plt.close(fig)
        print(f"\n    {_c_ok('✓ ANOVA boxplot saved: ' + anova_plot_path.name)}")
    except Exception as plot_err:
        print("\n    " + _c_warn(f"⊘ Plot failed: {plot_err}"))

    print(_c_header("━" * 54))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Module 12: Data Cleaning Tools
# ─────────────────────────────────────────────────────────────────────────────

def _remove_outliers_iqr(df: pd.DataFrame, col: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Remove rows where col is outside [Q1 - m*IQR, Q3 + m*IQR]."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return df[(df[col] >= lower) & (df[col] <= upper)].copy()


def _remove_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> pd.DataFrame:
    """Remove rows where |z-score| > threshold for the given column."""
    z = np.abs(stats.zscore(df[col].dropna()))
    valid_idx = df[col].dropna().index[z <= threshold]
    return df.loc[df.index.isin(valid_idx) | df[col].isna()].copy()


def data_cleaning_menu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactive sub-menu for data cleaning operations.

    Operations:
        1.  Drop rows with any missing values
        2.  Fill missing numeric values with column mean
        3.  Fill missing numeric values with column median
        4.  Remove duplicate rows
        5.  Drop a specific column
        6.  Fill missing categorical values with mode
        7.  Remove outliers from a numeric column (IQR method)
        8.  Remove outliers from a numeric column (Z-score method)
        9.  Rename a column
        10. Strip leading/trailing whitespace from all string columns
        11. Cast a column to a different data type

    Returns:
        pd.DataFrame — the cleaned DataFrame.
    """
    def _menu_text() -> str:
        shape_text = f"Shape: {df.shape[0]} rows × {df.shape[1]} columns"
        return f"""
{_c_header('━' * 54)}
  {_c_header('DATA CLEANING TOOLS')}
{_c_header('━' * 54)}
  {_c_dim(shape_text)}

  {_c_title('1.')}  Drop rows with any missing values
  {_c_title('2.')}  Fill missing numeric values with mean
  {_c_title('3.')}  Fill missing numeric values with median
  {_c_title('4.')}  Remove duplicate rows
  {_c_title('5.')}  Drop a specific column
  {_c_title('6.')}  Fill missing categorical values with mode
  {_c_title('7.')}  Remove outliers — IQR method
  {_c_title('8.')}  Remove outliers — Z-score method
  {_c_title('9.')}  Rename a column
  {_c_title('10.')} Strip whitespace from all string columns
  {_c_title('11.')} Cast column to different data type
  {_c_err('0.')}  Back to main menu
{_c_header('━' * 54)}
"""

    while True:
        print(_menu_text())
        choice = input("  Select a cleaning option [0-11]: ").strip()
        before_shape = df.shape

        if choice == "1":
            df = df.dropna()
            dropped = before_shape[0] - df.shape[0]
            print("\n  " + _c_ok(f"✅  Dropped {dropped} row(s) with missing values."))
            print(f"      Before: {before_shape[0]} rows → After: {df.shape[0]} rows")

        elif choice == "2":
            numeric_cols_list = _numeric_cols(df)
            filled_count = 0
            for col in numeric_cols_list:
                n_missing = int(df[col].isna().sum())
                if n_missing > 0:
                    df[col] = df[col].fillna(df[col].mean())
                    filled_count += n_missing
            print("\n  " + _c_ok(f"✅  Filled {filled_count} missing numeric value(s) with column means."))

        elif choice == "3":
            numeric_cols_list = _numeric_cols(df)
            filled_count = 0
            for col in numeric_cols_list:
                n_missing = int(df[col].isna().sum())
                if n_missing > 0:
                    df[col] = df[col].fillna(df[col].median())
                    filled_count += n_missing
            print("\n  " + _c_ok(f"✅  Filled {filled_count} missing numeric value(s) with column medians."))

        elif choice == "4":
            df = df.drop_duplicates()
            dropped = before_shape[0] - df.shape[0]
            print("\n  " + _c_ok(f"✅  Removed {dropped} duplicate row(s)."))
            print(f"      Before: {before_shape[0]} rows → After: {df.shape[0]} rows")

        elif choice == "5":
            print(f"\n  {_c_title('Available columns:')} {', '.join(df.columns.tolist())}")
            col_name = input("  Enter column name to drop: ").strip()
            if col_name not in df.columns:
                print("  " + _c_err(f"❌  Column '{col_name}' not found."))
            else:
                df = df.drop(columns=[col_name])
                print("\n  " + _c_ok(f"✅  Dropped column '{col_name}'."))
                print(f"      Remaining columns: {df.shape[1]}")

        elif choice == "6":
            cat_cols_list = _cat_cols(df)
            filled_count = 0
            for col in cat_cols_list:
                n_missing = int(df[col].isna().sum())
                if n_missing > 0:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                        filled_count += n_missing
            print("\n  " + _c_ok(f"✅  Filled {filled_count} missing categorical value(s) with mode."))

        elif choice == "7":
            numeric_cols_list = _numeric_cols(df)
            if not numeric_cols_list:
                print(f"  {_c_warn('⚠  No numeric columns.')}")
            else:
                print(f"\n  {_c_title('Numeric columns:')} {', '.join(numeric_cols_list)}")
                col_name = input("  Enter column name: ").strip()
                if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
                    print("  " + _c_err("❌  Invalid numeric column."))
                else:
                    raw_mult = input("  IQR multiplier (default 1.5): ").strip()
                    try:
                        mult = float(raw_mult) if raw_mult else 1.5
                    except ValueError:
                        mult = 1.5
                    df = _remove_outliers_iqr(df, col_name, mult)
                    dropped = before_shape[0] - df.shape[0]
                    print("\n  " + _c_ok(f"✅  Removed {dropped} outlier(s) from '{col_name}' (IQR × {mult})."))
                    print(f"      Before: {before_shape[0]} rows → After: {df.shape[0]} rows")

        elif choice == "8":
            numeric_cols_list = _numeric_cols(df)
            if not numeric_cols_list:
                print(f"  {_c_warn('⚠  No numeric columns.')}")
            else:
                print(f"\n  {_c_title('Numeric columns:')} {', '.join(numeric_cols_list)}")
                col_name = input("  Enter column name: ").strip()
                if col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[col_name]):
                    print("  " + _c_err("❌  Invalid numeric column."))
                else:
                    raw_thresh = input("  Z-score threshold (default 3.0): ").strip()
                    try:
                        thresh = float(raw_thresh) if raw_thresh else 3.0
                    except ValueError:
                        thresh = 3.0
                    df = _remove_outliers_zscore(df, col_name, thresh)
                    dropped = before_shape[0] - df.shape[0]
                    print("\n  " + _c_ok(f"✅  Removed {dropped} outlier(s) from '{col_name}' (|z| > {thresh})."))
                    print(f"      Before: {before_shape[0]} rows → After: {df.shape[0]} rows")

        elif choice == "9":
            print(f"\n  {_c_title('Available columns:')} {', '.join(df.columns.tolist())}")
            old_name = input("  Enter current column name: ").strip()
            if old_name not in df.columns:
                print("  " + _c_err(f"❌  Column '{old_name}' not found."))
            else:
                new_name = input(f"  Enter new name for '{old_name}': ").strip()
                if not new_name:
                    print(f"  {_c_warn('⚠  New name cannot be empty.')}")
                else:
                    df = df.rename(columns={old_name: new_name})
                    print("\n  " + _c_ok(f"✅  Renamed '{old_name}' → '{new_name}'."))

        elif choice == "10":
            str_cols = df.select_dtypes(include="object").columns.tolist()
            for col in str_cols:
                df[col] = df[col].str.strip()
            print("\n  " + _c_ok(f"✅  Stripped whitespace from {len(str_cols)} string column(s)."))

        elif choice == "11":
            print(f"\n  {_c_title('Available columns:')} {', '.join(df.columns.tolist())}")
            col_name = input("  Enter column name: ").strip()
            if col_name not in df.columns:
                print("  " + _c_err(f"❌  Column '{col_name}' not found."))
            else:
                print(f"  Current dtype: {df[col_name].dtype}")
                print(f"  Target types: int, float, str, bool, datetime")
                target = input("  Enter target type: ").strip().lower()
                try:
                    if target in ("int", "int64"):
                        df[col_name] = pd.to_numeric(df[col_name], errors="coerce").astype("Int64")
                    elif target in ("float", "float64"):
                        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                    elif target in ("str", "string", "object"):
                        df[col_name] = df[col_name].astype(str)
                    elif target == "bool":
                        df[col_name] = df[col_name].astype(bool)
                    elif target in ("datetime", "date"):
                        df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
                    else:
                        print("  " + _c_warn(f"⚠  Unknown type '{target}'."))
                        continue
                    print("\n  " + _c_ok(f"✅  Cast '{col_name}' to {target}. New dtype: {df[col_name].dtype}"))
                except Exception as e:
                    print("  " + _c_err(f"❌  Cast failed: {e}"))

        elif choice == "0":
            break

        else:
            print(f"\n  {_c_warn('⚠  Invalid option. Please enter 0-11.')}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Module 13: Interactive Filtering
# ─────────────────────────────────────────────────────────────────────────────

def interactive_filter(df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interactive sub-menu for filtering the dataset with multi-condition support.

    Operations:
        1.  Filter by exact column value (categorical / numeric)
        2.  Filter by numeric range (min–max)
        3.  Filter by string contains / startswith / endswith / regex
        4.  Apply multiple AND conditions at once
        5.  Reset all filters (restore original)
        6.  Show active filter summary

    Args:
        df:          The current (possibly already filtered) DataFrame.
        original_df: The original unfiltered DataFrame for reset.

    Returns:
        pd.DataFrame — the filtered (or reset) DataFrame.
    """
    # Stack of applied filter descriptions for display
    active_filters: list[str] = []

    def _menu_text() -> str:
        filter_info = (
            _c_ok(str(len(active_filters)) + " filter(s) active")
            if active_filters
            else _c_dim("no filters")
        )
        current_info = f"Current: {len(df)} rows × {df.shape[1]} cols"
        return f"""
{_c_header('━' * 54)}
  {_c_header('INTERACTIVE FILTERING')}
{_c_header('━' * 54)}
  {_c_dim(current_info)}  [{filter_info}]

  {_c_title('1.')}  Filter by exact value
  {_c_title('2.')}  Filter by numeric range
  {_c_title('3.')}  Filter by string pattern (contains / startswith / endswith / regex)
  {_c_title('4.')}  Multi-condition AND filter
  {_c_title('5.')}  Reset all filters
  {_c_title('6.')}  Show active filters
  {_c_err('0.')}  Back to main menu
{_c_header('━' * 54)}
"""

    while True:
        print(_menu_text())
        choice = input("  Select a filter option [0-6]: ").strip()

        if choice == "1":
            # ── Exact value match ──────────────────────────────────────────
            print(f"\n  {_c_title('Columns:')} {', '.join(df.columns.tolist())}")
            col = input("  Enter column to filter by: ").strip()
            if col not in df.columns:
                print("  " + _c_err("❌  Column not found."))
                continue

            unique_vals = df[col].dropna().unique()
            display_n   = min(30, len(unique_vals))
            title_text = f"Sample values ({display_n} of {len(unique_vals)}):"
            print("  " + _c_title(title_text) + " " +
                  f"{', '.join(str(v) for v in unique_vals[:display_n])}")
            value = input("  Enter the value to keep: ").strip()

            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    value = float(value)
                except ValueError:
                    pass

            before = len(df)
            df = df[df[col] == value].copy()
            after = len(df)
            desc  = f"{col} == {value!r}"
            print("\n  " + _c_ok(f"✅  {desc}: {before} → {after} rows"))
            if after == 0:
                print(f"  {_c_warn('⚠  No rows match. Resetting to previous state.')}")
                df = original_df.copy() if not active_filters else df
            else:
                active_filters.append(desc)

        elif choice == "2":
            # ── Numeric range ──────────────────────────────────────────────
            numeric_cols_list = _numeric_cols(df)
            if not numeric_cols_list:
                print(f"  {_c_warn('⚠  No numeric columns available.')}")
                continue

            print(f"\n  {_c_title('Numeric columns:')} {', '.join(numeric_cols_list)}")
            col = input("  Enter numeric column: ").strip()
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                print("  " + _c_err("❌  Not a valid numeric column."))
                continue

            col_min, col_max = df[col].min(), df[col].max()
            print(f"  Current range: {_c_accent(str(col_min))} — {_c_accent(str(col_max))}")
            try:
                min_val = float(input(f"  Min value (Enter for {col_min}): ").strip() or col_min)
                max_val = float(input(f"  Max value (Enter for {col_max}): ").strip() or col_max)
            except ValueError:
                print(f"  {_c_err('❌  Invalid numeric input.')}")
                continue

            before = len(df)
            df = df[(df[col] >= min_val) & (df[col] <= max_val)].copy()
            after = len(df)
            desc  = f"{min_val} ≤ {col} ≤ {max_val}"
            print("\n  " + _c_ok(f"✅  {desc}: {before} → {after} rows"))
            if after == 0:
                print(f"  {_c_warn('⚠  No rows match. Resetting to previous state.')}")
                df = original_df.copy() if not active_filters else df
            else:
                active_filters.append(desc)

        elif choice == "3":
            # ── String pattern filter ──────────────────────────────────────
            str_cols = df.select_dtypes(include="object").columns.tolist()
            if not str_cols:
                print(f"  {_c_warn('⚠  No string columns available.')}")
                continue

            print(f"\n  {_c_title('String columns:')} {', '.join(str_cols)}")
            col = input("  Enter string column: ").strip()
            if col not in df.columns:
                print("  " + _c_err("❌  Column not found."))
                continue

            print(f"  Modes: {_c_dim('[1] contains  [2] startswith  [3] endswith  [4] regex')}")
            mode_choice = input("  Select mode [1-4]: ").strip()
            pattern = input("  Enter pattern: ").strip()
            before  = len(df)

            try:
                if mode_choice == "1":
                    mask = df[col].astype(str).str.contains(pattern, case=False, na=False, regex=False)
                    desc = f"{col} contains '{pattern}'"
                elif mode_choice == "2":
                    mask = df[col].astype(str).str.startswith(pattern, na=False)
                    desc = f"{col} startswith '{pattern}'"
                elif mode_choice == "3":
                    mask = df[col].astype(str).str.endswith(pattern, na=False)
                    desc = f"{col} endswith '{pattern}'"
                elif mode_choice == "4":
                    mask = df[col].astype(str).str.contains(pattern, case=False, na=False, regex=True)
                    desc = f"{col} matches regex '{pattern}'"
                else:
                    print(f"  {_c_warn('⚠  Invalid mode.')}")
                    continue

                df = df[mask].copy()
                after = len(df)
                print("\n  " + _c_ok(f"✅  {desc}: {before} → {after} rows"))
                if after == 0:
                    print(f"  {_c_warn('⚠  No rows match. Resetting.')}")
                    df = original_df.copy()
                else:
                    active_filters.append(desc)
            except Exception as e:
                print("  " + _c_err(f"❌  Filter error: {e}"))

        elif choice == "4":
            # ── Multi-condition AND filter ─────────────────────────────────
            print(f"\n  {_c_title('Enter multiple conditions (AND logic).')}")
            print(f"  {_c_dim('Format: column==value  or  column>=num  or  column<=num')}")
            print(f"  {_c_dim('Examples: Age>=25, Gender==Male, Income<=50000')}")
            raw_conds = input("  Enter conditions (comma-separated): ").strip()
            conditions = [c.strip() for c in raw_conds.split(",") if c.strip()]

            if not conditions:
                print(f"  {_c_warn('⚠  No conditions entered.')}")
                continue

            combined_mask = pd.Series([True] * len(df), index=df.index)
            valid_conds   = []

            for cond in conditions:
                # Parse: column OP value
                m = re.match(r"^(.+?)\s*(==|!=|>=|<=|>|<)\s*(.+)$", cond)
                if not m:
                    print("  " + _c_warn(f"⚠  Cannot parse condition: {cond!r} — skipping."))
                    continue
                col_c, op, val_c = m.group(1).strip(), m.group(2), m.group(3).strip()

                if col_c not in df.columns:
                    print("  " + _c_warn(f"⚠  Column '{col_c}' not found — skipping."))
                    continue

                try:
                    if pd.api.types.is_numeric_dtype(df[col_c]):
                        num_val = float(val_c)
                        ops = {"==": df[col_c] == num_val,
                               "!=": df[col_c] != num_val,
                               ">=": df[col_c] >= num_val,
                               "<=": df[col_c] <= num_val,
                               ">":  df[col_c] > num_val,
                               "<":  df[col_c] < num_val}
                        combined_mask &= ops[op]
                    else:
                        if op == "==":
                            combined_mask &= (df[col_c].astype(str) == val_c)
                        elif op == "!=":
                            combined_mask &= (df[col_c].astype(str) != val_c)
                        else:
                            print("  " + _c_warn(f"⚠  Operator {op!r} not valid for string column — skipping."))
                            continue
                    valid_conds.append(cond)
                except Exception as cond_err:
                    print("  " + _c_warn(f"⚠  Error in condition {cond!r}: {cond_err} — skipping."))

            if not valid_conds:
                print(f"  {_c_warn('⚠  No valid conditions.')}")
                continue

            before = len(df)
            df = df[combined_mask].copy()
            after = len(df)
            desc  = "AND(" + ", ".join(valid_conds) + ")"
            print("\n  " + _c_ok(f"✅  {before} → {after} rows with: {desc}"))
            if after == 0:
                print(f"  {_c_warn('⚠  No rows match. Resetting to original.')}")
                df = original_df.copy()
                active_filters.clear()
            else:
                active_filters.append(desc)

        elif choice == "5":
            df = original_df.copy()
            active_filters.clear()
            print("\n  " + _c_ok(f"✅  Filters reset. Dataset restored to {len(df)} rows."))

        elif choice == "6":
            if active_filters:
                print(f"\n  {_c_title('Active Filters:')}")
                for i, f_desc in enumerate(active_filters, 1):
                    print(f"    {_c_accent(str(i))}. {f_desc}")
                print(f"  Total: {_c_accent(str(len(active_filters)))} filter(s) applied")
                print(f"  Current rows: {_c_accent(str(len(df)))} / {_c_accent(str(len(original_df)))}")
            else:
                print(f"\n  {_c_dim('No active filters — showing all data.')}")

        elif choice == "0":
            break

        else:
            print(f"\n  {_c_warn('⚠  Invalid option. Please enter 0-6.')}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Module 14: Automatic Report Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive text report and save to reports/analysis_report.txt.

    The report includes:
        1. Dataset Information
        2. Missing Value Summary
        3. Descriptive Statistics
        4. Frequency Tables (capped at top 20 per column)
        5. Correlation Analysis
        6. Normality Tests

    Returns:
        str — the full report content.
    """
    ensure_directories()

    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 60)
        lines.append(f"  {title}")
        lines.append("=" * 60)

    # ---- 1. Dataset Information ----
    rows, cols = df.shape
    section("1. DATASET INFORMATION")
    lines.append(f"  Observations : {rows}")
    lines.append(f"  Variables    : {cols}")
    lines.append("")
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            kind = "Numeric"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            kind = "Datetime"
        else:
            kind = "Categorical"
        lines.append(f"    {col:<22s}  {kind} ({dtype})")

    # ---- 2. Missing Values ----
    section("2. MISSING VALUE SUMMARY")
    total = len(df)
    any_missing = False
    for col in df.columns:
        mc = int(df[col].isna().sum())
        if mc > 0:
            pct = mc / total * 100
            lines.append(f"    {col:<22s}  {mc:>5d}  ({pct:.1f}%)")
            any_missing = True
    if not any_missing:
        lines.append("    No missing values found.")

    # ---- 3. Descriptive Statistics ----
    section("3. DESCRIPTIVE STATISTICS")
    numeric_df = df.select_dtypes(include="number")
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        mode_vals = s.mode()
        mode_val = mode_vals.iloc[0] if not mode_vals.empty else "N/A"
        lines.append(f"\n  Variable: {col}")
        lines.append(f"    Count        = {int(s.count())}")
        lines.append(f"    Mean         = {s.mean():.4f}")
        lines.append(f"    Median       = {s.median():.4f}")
        lines.append(f"    Mode         = {mode_val}")
        lines.append(f"    Std Dev      = {s.std(ddof=1):.4f}")
        lines.append(f"    Variance     = {s.var(ddof=1):.4f}")
        lines.append(f"    Min          = {s.min():.4f}")
        lines.append(f"    Max          = {s.max():.4f}")
        lines.append(f"    Range        = {s.max() - s.min():.4f}")
        lines.append(f"    Skewness     = {s.skew():.4f}")
        lines.append(f"    Kurtosis     = {s.kurtosis():.4f}")

    # ---- 4. Frequency Tables (capped) ----
    section("4. FREQUENCY DISTRIBUTION")
    cat_cols_list = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols_list:
        for col in cat_cols_list:
            counts = df[col].value_counts(dropna=False)
            total_cat = counts.sum()
            n_unique = len(counts)
            lines.append(f"\n  Variable: {col} ({n_unique} unique values)")
            for idx, (val, cnt) in enumerate(counts.items()):
                if idx >= MAX_FREQ_DISPLAY:
                    remaining = n_unique - MAX_FREQ_DISPLAY
                    lines.append(f"    ... and {remaining} more unique values (not shown)")
                    break
                pct = cnt / total_cat * 100
                lines.append(f"    {str(val):<22s}  {cnt:>5d}  ({pct:.1f}%)")
    else:
        lines.append("    No categorical columns found.")

    # ---- 5. Correlation Analysis ----
    section("5. CORRELATION ANALYSIS (Pearson)")
    numeric_clean = numeric_df.dropna()
    num_cols = numeric_clean.columns.tolist()
    if len(num_cols) >= 2:
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r, p = stats.pearsonr(numeric_clean[num_cols[i]], numeric_clean[num_cols[j]])
                sig = "Significant" if p < ALPHA else "Not Significant"
                lines.append(f"    {num_cols[i]} vs {num_cols[j]}")
                lines.append(f"      r = {r:.4f},  p = {p:.4f}  →  {sig}")
    else:
        lines.append("    Insufficient numeric columns for correlation.")

    # ---- 6. Normality Tests (Shapiro-Wilk / Anderson-Darling) ----
    section("6. NORMALITY TESTS (Shapiro-Wilk / Anderson-Darling)")
    for col in numeric_df.columns:
        data = numeric_df[col].dropna()
        if len(data) < 3:
            lines.append(f"    {col}: Skipped (need ≥ 3 observations)")
            continue
        try:
            if len(data) <= SHAPIRO_MAX_N:
                w_stat, p_value = stats.shapiro(data)
                verdict = "Normal" if p_value > ALPHA else "Not Normal"
                lines.append(f"    {col:<22s}  W = {w_stat:.6f},  p = {p_value:.6f}  →  {verdict}  [Shapiro-Wilk]")
            else:
                sample = data.sample(SHAPIRO_MAX_N, random_state=42)
                ad_res = stats.anderson(data, dist="norm")
                crit   = ad_res.critical_values[2]
                verdict = "Normal" if ad_res.statistic < crit else "Not Normal"
                lines.append(
                    f"    {col:<22s}  A² = {ad_res.statistic:.6f},  crit(5%) = {crit:.6f}  →  {verdict}"
                    f"  [Anderson-Darling, n={len(data)}]"
                )
        except Exception as e:
            lines.append(f"    {col:<22s}  Error: {e}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("  END OF REPORT")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    report_path = REPORTS_DIR / "analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n{_c_ok('✅  Report saved to: ' + str(report_path))}")
    return report_text


# ─────────────────────────────────────────────────────────────────────────────
#  Module 15: Export Results
# ─────────────────────────────────────────────────────────────────────────────

def export_results(df: pd.DataFrame) -> list[str]:
    """
    Export descriptive statistics and frequency tables to CSV files
    in the exports/ directory.

    Returns:
        List of saved file paths.
    """
    ensure_directories()
    saved: list[str] = []

    # --- Descriptive Statistics CSV ---
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        stats_records = {}
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            mode_vals = s.mode()
            mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan
            stats_records[col] = {
                "Count":    int(s.count()),
                "Mean":     round(s.mean(), 4),
                "Median":   round(s.median(), 4),
                "Mode":     round(mode_val, 4) if pd.notna(mode_val) else "N/A",
                "Std Dev":  round(s.std(ddof=1), 4),
                "Variance": round(s.var(ddof=1), 4),
                "Min":      round(s.min(), 4),
                "Max":      round(s.max(), 4),
                "Range":    round(s.max() - s.min(), 4),
            }
        desc_df   = pd.DataFrame(stats_records)
        desc_path = EXPORTS_DIR / "descriptive_statistics.csv"
        desc_df.to_csv(desc_path)
        saved.append(str(desc_path))
        print(f"  {_c_ok('✓ Exported: ' + desc_path.name)}")

    # --- Frequency Tables CSV ---
    cat_cols_list = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols_list:
        freq_frames = []
        for col in cat_cols_list:
            counts = df[col].value_counts(dropna=False)
            total  = counts.sum()
            pct    = (counts / total * 100).round(2)
            temp = pd.DataFrame({
                "Variable":   col,
                "Value":      counts.index.astype(str),
                "Frequency":  counts.values,
                "Percentage": pct.values,
            })
            freq_frames.append(temp)
        freq_df   = pd.concat(freq_frames, ignore_index=True)
        freq_path = EXPORTS_DIR / "frequency_tables.csv"
        freq_df.to_csv(freq_path, index=False)
        saved.append(str(freq_path))
        print(f"  {_c_ok('✓ Exported: ' + freq_path.name)}")

    if saved:
        print("\n" + _c_ok(f"✅  {len(saved)} file(s) exported to: {EXPORTS_DIR}"))
    else:
        print(f"\n{_c_warn('⚠  No data available to export.')}")

    return saved


# ─────────────────────────────────────────────────────────────────────────────
#  Full Automated Analysis
# ─────────────────────────────────────────────────────────────────────────────

def full_automated_analysis(df: pd.DataFrame) -> None:
    """Run all analysis modules sequentially and generate the report."""
    print(f"\n{_c_header('🚀  Running Full Automated Analysis …')}\n")
    dataset_summary(df)
    missing_value_analysis(df)
    descriptive_statistics(df)
    frequency_distribution(df)
    generate_visualizations(df)
    correlation_analysis(df)
    normality_test(df)
    generate_report(df)
    export_results(df)
    print(f"\n{_c_ok('✅  Full Automated Analysis Complete!')}")


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: calculate_mean  (testable utility)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_mean(values: list) -> float:
    """
    Calculate the arithmetic mean of a list of numbers.

    Args:
        values: Non-empty list of numeric values.

    Returns:
        The arithmetic mean as a float.

    Raises:
        ValueError: If the list is empty.
    """
    if not values:
        raise ValueError("Cannot calculate mean of an empty list.")
    return sum(values) / len(values)


# ─────────────────────────────────────────────────────────────────────────────
#  Main Menu
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point — interactive CLI menu loop."""
    ensure_directories()

    print()
    print(f"{Fore.CYAN}{Style.BRIGHT}╔══════════════════════════════════════════════════════╗")
    print(f"║   ezpzAnalyze — Research Data Analysis Toolkit        ║")
    print(f"╚══════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print()

    # Load Dataset
    df = None
    while df is None:
        filepath = input(f"  {_c_title('Enter the path to your CSV file:')} ").strip()
        if not filepath:
            print(f"  {_c_warn('⚠  Please provide a file path.')}\n")
            continue
        try:
            df = load_dataset(filepath)
        except FileNotFoundError as e:
            print("  " + _c_err(f"❌  {e}"))
            print("      Please check the path and try again.\n")
        except ValueError as e:
            print("  " + _c_err(f"❌  {e}") + "\n")

    # Keep a copy of the original for filter reset
    original_df = df.copy()

    # Configure Output Folder
    output_dir = input(
        f"\n  {_c_title('Enter a folder path to store results (leave blank for default):')} "
    ).strip()
    if output_dir:
        global REPORTS_DIR, PLOTS_DIR, EXPORTS_DIR
        base_output = Path(output_dir).resolve()
        REPORTS_DIR = base_output / "reports"
        PLOTS_DIR   = base_output / "plots"
        EXPORTS_DIR = base_output / "exports"
        for d in [REPORTS_DIR, PLOTS_DIR, EXPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        print("  " + _c_ok(f"✅  Results will be stored in: {base_output}") + "\n")
    else:
        print("  " + _c_ok(f"✅  Results will be stored in: {BASE_DIR}") + "\n")

    # Menu Loop
    while True:
        print(MENU_TEXT)
        choice = input(f"  {_c_title('Select an option [0-16]:')} ").strip()

        try:
            if choice == "1":
                dataset_summary(df)
            elif choice == "2":
                missing_value_analysis(df)
            elif choice == "3":
                descriptive_statistics(df)
            elif choice == "4":
                frequency_distribution(df)
            elif choice == "5":
                generate_visualizations(df)
            elif choice == "6":
                correlation_analysis(df)
            elif choice == "7":
                independent_ttest(df)
            elif choice == "8":
                chi_square_test(df)
            elif choice == "9":
                normality_test(df)
            elif choice == "10":
                linear_regression(df)
            elif choice == "11":
                one_way_anova(df)
            elif choice == "12":
                df = data_cleaning_menu(df)
            elif choice == "13":
                df = interactive_filter(df, original_df)
            elif choice == "14":
                full_automated_analysis(df)
            elif choice == "15":
                generate_report(df)
            elif choice == "16":
                export_results(df)
            elif choice == "0":
                print(f"\n  {_c_info('👋  Thank you for using ezpzAnalyze. Goodbye!')}\n")
                sys.exit(0)
            else:
                print(f"\n  {_c_warn('⚠  Invalid option. Please enter a number between 0 and 16.')}")
        except KeyError as e:
            print("\n  " + _c_err(f"❌  Column Error: {e}"))
        except ValueError as e:
            print("\n  " + _c_err(f"❌  Value Error: {e}"))
        except Exception as e:
            print("\n  " + _c_err(f"❌  Unexpected Error: {e}"))


if __name__ == "__main__":
    main()
