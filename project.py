import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend — no GUI popups

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats



#  Constants

ALPHA = 0.05  # Significance level for hypothesis tests
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
PLOTS_DIR = BASE_DIR / "plots"
EXPORTS_DIR = BASE_DIR / "exports"

# Create directories if they don't exist
for d in [DATA_DIR, REPORTS_DIR, PLOTS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MENU_TEXT = """
         ezpzAnalyze — Research Data Analysis Toolkit

  1.  Dataset Summary
  2.  Missing Value Analysis
  3.  Descriptive Statistics
  4.  Frequency Distribution
  5.  Generate Visualizations
  6.  Correlation Analysis
  7.  Independent t-Test
  8.  Chi-Square Test
  9.  Full Automated Analysis
 10.  Generate Report
 11.  Export Results
  0.  Exit

====================================================
"""



#  Setup
def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    for directory in (DATA_DIR, REPORTS_DIR, PLOTS_DIR, EXPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


#  Module 1: Dataset Import
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

    rows, cols = df.shape
    print(f"\n✅  Dataset Loaded Successfully")
    print(f"    Rows:    {rows}")
    print(f"    Columns: {cols}")
    return df



#  Module 2: Dataset Summary
def dataset_summary(df: pd.DataFrame) -> str:
    """
    Print and return a textual overview of the dataset.

    Displays:
        - Number of observations (rows)
        - Number of variables (columns)
        - Variable names
        - Data type classification (Numeric / Categorical)
    """
    rows, cols = df.shape
    lines = [
        "",
        "━" * 50,
        "  DATASET SUMMARY",
        "━" * 50,
        f"  Observations : {rows}",
        f"  Variables    : {cols}",
        "",
        "  Variable List:",
    ]

    for col in df.columns:
        dtype = df[col].dtype
        kind = "Numeric" if pd.api.types.is_numeric_dtype(dtype) else "Categorical"
        lines.append(f"    • {col:<20s} → {kind} ({dtype})")

    lines.append("━" * 50)
    summary_text = "\n".join(lines)
    print(summary_text)
    return summary_text


#  Module 3: Missing Value Analysis
def missing_value_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze missing values in every column.

    Returns:
        dict mapping column name → (missing_count, missing_percentage).
        Only columns with at least one missing value are included.
    """
    total = len(df)
    result: dict[str, tuple[int, float]] = {}

    lines = [
        "",
        "━" * 50,
        "  MISSING VALUE ANALYSIS",
        "━" * 50,
    ]

    any_missing = False
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            pct = (missing_count / total) * 100
            result[col] = (missing_count, round(pct, 2))
            lines.append(f"    {col:<20s}  {missing_count:>5d}  ({pct:.1f}%)")
            any_missing = True

    if not any_missing:
        lines.append("    No missing values found. ✓")

    lines.append("━" * 50)
    print("\n".join(lines))
    return result



#  Module 4: Descriptive Statistics
def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for every numeric column.

    Statistics: Count, Mean, Median, Mode, Std Dev (ddof=1),
    Variance (ddof=1), Min, Max, Range.

    Returns:
        pd.DataFrame with statistics as rows and variables as columns.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("\n⚠  No numeric columns found for descriptive statistics.")
        return pd.DataFrame()

    stats_dict: dict[str, dict] = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        mode_vals = series.mode()
        mode_val = mode_vals.iloc[0] if not mode_vals.empty else np.nan

        stats_dict[col] = {
            "Count": int(series.count()),
            "Mean": round(series.mean(), 4),
            "Median": round(series.median(), 4),
            "Mode": round(mode_val, 4) if pd.notna(mode_val) else "N/A",
            "Std Dev": round(series.std(ddof=1), 4),
            "Variance": round(series.var(ddof=1), 4),
            "Min": round(series.min(), 4),
            "Max": round(series.max(), 4),
            "Range": round(series.max() - series.min(), 4),
        }

    result_df = pd.DataFrame(stats_dict)

    print()
    print("━" * 50)
    print("  DESCRIPTIVE STATISTICS")
    print("━" * 50)
    for col, col_stats in stats_dict.items():
        print(f"\n  Variable: {col}")
        for stat_name, value in col_stats.items():
            print(f"    {stat_name:<12s} = {value}")
    print("━" * 50)

    return result_df



#  Module 5: Frequency Distribution
def frequency_distribution(df: pd.DataFrame) -> dict:
    """
    Compute frequency tables for every categorical (non-numeric) column.

    Returns:
        dict mapping column name → pd.DataFrame with columns
        ['Value', 'Frequency', 'Percentage'].
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        print("\n⚠  No categorical columns found for frequency distribution.")
        return {}

    result: dict[str, pd.DataFrame] = {}

    print()
    print("━" * 50)
    print("  FREQUENCY DISTRIBUTION")
    print("━" * 50)

    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        total = counts.sum()
        pct = (counts / total * 100).round(2)

        freq_df = pd.DataFrame({
            "Value": counts.index.astype(str),
            "Frequency": counts.values,
            "Percentage": pct.values,
        })
        result[col] = freq_df

        print(f"\n  Variable: {col}")
        for _, row in freq_df.iterrows():
            print(f"    {str(row['Value']):<20s}  {int(row['Frequency']):>5d}  ({row['Percentage']:.1f}%)")

    print("━" * 50)
    return result



#  Module 6: Data Visualization
def generate_visualizations(df: pd.DataFrame) -> list[str]:
    """
    Generate and save visualizations for all columns.

    - Histogram + Boxplot for each numeric column
    - Bar chart for each categorical column

    Returns:
        List of file paths for saved plots.
    """
    ensure_directories()
    saved_files: list[str] = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    print()
    print("━" * 50)
    print("  GENERATING VISUALIZATIONS")
    print("━" * 50)

    # --- Histograms ---
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = df[col].dropna()
        ax.hist(data, bins="auto", color="#4A90D9", edgecolor="#2C3E50", alpha=0.85)
        ax.set_title(f"Histogram — {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        filepath = PLOTS_DIR / f"histogram_{col.lower().replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        saved_files.append(str(filepath))
        print(f"    ✓ Saved: {filepath.name}")

    # --- Boxplots ---
    for col in numeric_cols:
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
        filepath = PLOTS_DIR / f"boxplot_{col.lower().replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        saved_files.append(str(filepath))
        print(f"    ✓ Saved: {filepath.name}")

    # --- Bar Charts ---
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        counts = df[col].value_counts()
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(counts)))
        ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="#2C3E50")
        ax.set_title(f"Bar Chart — {col}", fontsize=14, fontweight="bold")
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        filepath = PLOTS_DIR / f"barchart_{col.lower().replace(' ', '_')}.png"
        fig.tight_layout()
        fig.savefig(filepath, dpi=150)
        plt.close(fig)
        saved_files.append(str(filepath))
        print(f"    ✓ Saved: {filepath.name}")

    print(f"\n    Total plots saved: {len(saved_files)}")
    print("━" * 50)
    return saved_files



#  Module 7: Correlation Analysis
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
        print("\n⚠  Need at least 2 numeric columns for correlation analysis.")
        return pd.DataFrame()

    corr_matrix = numeric_df.corr(method="pearson")
    cols = numeric_df.columns.tolist()

    print()
    print("━" * 50)
    print("  CORRELATION ANALYSIS (Pearson)")
    print("━" * 50)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, p = stats.pearsonr(numeric_df[cols[i]], numeric_df[cols[j]])
            sig = "Significant" if p < ALPHA else "Not Significant"
            print(f"    {cols[i]} vs {cols[j]}")
            print(f"      r = {r:.4f},  p = {p:.4f}  →  {sig}")

    # --- Heatmap ---
    ensure_directories()
    fig, ax = plt.subplots(figsize=(max(8, len(cols)), max(6, len(cols) * 0.8)))
    cax = ax.matshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    # Annotate cells
    for (row, col_idx), val in np.ndenumerate(corr_matrix.values):
        ax.text(col_idx, row, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(val) > 0.6 else "black")

    ax.set_title("Pearson Correlation Heatmap", pad=20, fontsize=14, fontweight="bold")
    heatmap_path = PLOTS_DIR / "correlation_heatmap.png"
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"\n    ✓ Heatmap saved: {heatmap_path.name}")
    print("━" * 50)

    return corr_matrix



#  Module 8: Hypothesis Testing — Independent t-Test
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
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not numeric_cols:
        print("\n⚠  No numeric columns available for a t-test.")
        return {}
    if not cat_cols:
        print("\n⚠  No categorical columns available as a grouping variable.")
        return {}

    print("\n  Numeric columns:", ", ".join(numeric_cols))
    numeric_var = input("  Enter the numeric (dependent) variable: ").strip()
    if numeric_var not in df.columns:
        raise KeyError(f"Column '{numeric_var}' not found in the dataset.")
    if not pd.api.types.is_numeric_dtype(df[numeric_var]):
        raise ValueError(f"Column '{numeric_var}' is not numeric.")

    print("  Categorical columns:", ", ".join(cat_cols))
    group_var = input("  Enter the grouping variable (2 groups): ").strip()
    if group_var not in df.columns:
        raise KeyError(f"Column '{group_var}' not found in the dataset.")

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
        "variable": numeric_var,
        "grouping": group_var,
        "group1": str(groups[0]),
        "group2": str(groups[1]),
        "group1_mean": round(float(group1_data.mean()), 4),
        "group2_mean": round(float(group2_data.mean()), 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "significant": significant,
    }

    print()
    print("━" * 50)
    print("  INDEPENDENT SAMPLES t-TEST")
    print("━" * 50)
    print(f"    {groups[0]} vs {groups[1]}  on  {numeric_var}")
    print(f"    {groups[0]} Mean = {result['group1_mean']}")
    print(f"    {groups[1]} Mean = {result['group2_mean']}")
    print(f"    t = {result['t_stat']},  p = {result['p_value']}")
    print(f"    Result: {'✅ Significant Difference' if significant else '✗ No Significant Difference'}")
    print("━" * 50)

    return result



#  Module 8: Hypothesis Testing — Chi-Square Test
def chi_square_test(df: pd.DataFrame) -> dict:
    """
    Perform a Chi-Square Test of Independence.

    Prompts the user to select two categorical variables.

    Returns:
        dict with keys: 'var1', 'var2', 'chi2', 'p_value',
        'dof', 'significant'.
    """
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(cat_cols) < 2:
        print("\n⚠  Need at least 2 categorical columns for a Chi-Square test.")
        return {}

    print("\n  Categorical columns:", ", ".join(cat_cols))
    var1 = input("  Enter the first categorical variable:  ").strip()
    var2 = input("  Enter the second categorical variable: ").strip()

    for v in (var1, var2):
        if v not in df.columns:
            raise KeyError(f"Column '{v}' not found in the dataset.")

    subset = df[[var1, var2]].dropna()
    contingency = pd.crosstab(subset[var1], subset[var2])

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    significant = p_value < ALPHA

    result = {
        "var1": var1,
        "var2": var2,
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 4),
        "dof": int(dof),
        "significant": significant,
    }

    print()
    print("━" * 50)
    print("  CHI-SQUARE TEST OF INDEPENDENCE")
    print("━" * 50)
    print(f"    {var1} vs {var2}")
    print(f"    χ² = {result['chi2']},  p = {result['p_value']},  df = {result['dof']}")
    print(f"    Result: {'✅ Significant Association' if significant else '✗ No Significant Association'}")
    print("━" * 50)

    return result



#  Module 9: Automatic Report Generation
def generate_report(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive text report and save to reports/analysis_report.txt.

    The report includes:
        1. Dataset Information
        2. Missing Value Summary
        3. Descriptive Statistics
        4. Frequency Tables
        5. Correlation Analysis

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
        kind = "Numeric" if pd.api.types.is_numeric_dtype(dtype) else "Categorical"
        lines.append(f"    {col:<20s}  {kind} ({dtype})")

    # ---- 2. Missing Values ----
    section("2. MISSING VALUE SUMMARY")
    total = len(df)
    any_missing = False
    for col in df.columns:
        mc = int(df[col].isna().sum())
        if mc > 0:
            pct = mc / total * 100
            lines.append(f"    {col:<20s}  {mc:>5d}  ({pct:.1f}%)")
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

    # ---- 4. Frequency Tables ----
    section("4. FREQUENCY DISTRIBUTION")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False)
            total_cat = counts.sum()
            lines.append(f"\n  Variable: {col}")
            for val, cnt in counts.items():
                pct = cnt / total_cat * 100
                lines.append(f"    {str(val):<20s}  {cnt:>5d}  ({pct:.1f}%)")
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

    lines.append("")
    lines.append("=" * 60)
    lines.append("  END OF REPORT")
    lines.append("=" * 60)

    report_text = "\n".join(lines)

    report_path = REPORTS_DIR / "analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n✅  Report saved to: {report_path}")
    return report_text



#  Module 10: Export Results
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
                "Count": int(s.count()),
                "Mean": round(s.mean(), 4),
                "Median": round(s.median(), 4),
                "Mode": round(mode_val, 4) if pd.notna(mode_val) else "N/A",
                "Std Dev": round(s.std(ddof=1), 4),
                "Variance": round(s.var(ddof=1), 4),
                "Min": round(s.min(), 4),
                "Max": round(s.max(), 4),
                "Range": round(s.max() - s.min(), 4),
            }
        desc_df = pd.DataFrame(stats_records)
        desc_path = EXPORTS_DIR / "descriptive_statistics.csv"
        desc_df.to_csv(desc_path)
        saved.append(str(desc_path))
        print(f"  ✓ Exported: {desc_path.name}")

    # --- Frequency Tables CSV ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        freq_frames = []
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False)
            total = counts.sum()
            pct = (counts / total * 100).round(2)
            temp = pd.DataFrame({
                "Variable": col,
                "Value": counts.index.astype(str),
                "Frequency": counts.values,
                "Percentage": pct.values,
            })
            freq_frames.append(temp)
        freq_df = pd.concat(freq_frames, ignore_index=True)
        freq_path = EXPORTS_DIR / "frequency_tables.csv"
        freq_df.to_csv(freq_path, index=False)
        saved.append(str(freq_path))
        print(f"  ✓ Exported: {freq_path.name}")

    if saved:
        print(f"\n✅  {len(saved)} file(s) exported to: {EXPORTS_DIR}")
    else:
        print("\n⚠  No data available to export.")

    return saved

    
#  Module 9 (Full): Full Automated Analysis
def full_automated_analysis(df: pd.DataFrame) -> None:
    """Run all analysis modules sequentially and generate the report."""
    print("\n🚀  Running Full Automated Analysis …\n")
    dataset_summary(df)
    missing_value_analysis(df)
    descriptive_statistics(df)
    frequency_distribution(df)
    generate_visualizations(df)
    correlation_analysis(df)
    generate_report(df)
    export_results(df)
    print("\n✅  Full Automated Analysis Complete!")



# Helper: calculate_mean (testable utility)
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

# Main Menu
def main() -> None:
    """Main entry point — interactive CLI menu loop."""
    ensure_directories()

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   ezpzAnalyze — Research Data Analysis Toolkit    ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # Load Dataset
    df = None
    while df is None:
        filepath = input("  Enter the path to your CSV file: ").strip()
        if not filepath:
            print("  ⚠  Please provide a file path.\n")
            continue
        try:
            df = load_dataset(filepath)
        except FileNotFoundError as e:
            print(f"  ❌  {e}")
            print("      Please check the path and try again.\n")
        except ValueError as e:
            print(f"  ❌  {e}\n")

    # Configure Output Folder 
    output_dir = input("\n  Enter a folder path to store analysis results (leave blank for default): ").strip()
    if output_dir:
        global REPORTS_DIR, PLOTS_DIR, EXPORTS_DIR
        base_output = Path(output_dir).resolve()
        REPORTS_DIR = base_output / "reports"
        PLOTS_DIR = base_output / "plots"
        EXPORTS_DIR = base_output / "exports"
        for d in [REPORTS_DIR, PLOTS_DIR, EXPORTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"  ✅  Results will be stored in: {base_output}\n")
    else:
        print(f"  ✅  Results will be stored in: {BASE_DIR}\n")

    # Menu Loop 
    while True:
        print(MENU_TEXT)
        choice = input("  Select an option [0-11]: ").strip()

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
                full_automated_analysis(df)
            elif choice == "10":
                generate_report(df)
            elif choice == "11":
                export_results(df)
            elif choice == "0":
                print("\n  👋  Thank you for using ezpzAnalyze. Goodbye!\n")
                sys.exit(0)
            else:
                print("\n  ⚠  Invalid option. Please enter a number between 0 and 11.")
        except KeyError as e:
            print(f"\n  ❌  Column Error: {e}")
        except ValueError as e:
            print(f"\n  ❌  Value Error: {e}")
        except Exception as e:
            print(f"\n  ❌  Unexpected Error: {e}")


if __name__ == "__main__":
    main()
