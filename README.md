# ezpzAnalyze: Research Data Analysis Toolkit

#### Video Demo: *\<https://youtu.be/imI2qQg2ZEE\>*

## Description

**ezpzAnalyze** is a terminal-based Python application that performs comprehensive statistical analysis on any user-provided CSV dataset. It is designed as a simplified alternative to tools like SPSS, Stata, or R — entirely from the command line.

A user simply runs `python project.py`, points to any CSV file, and can instantly generate descriptive statistics, frequency tables, visualizations, correlation analyses, hypothesis tests, and a complete automated report.

---

## Features

| # | Module | Description |
|---|--------|-------------|
| 1 | **Dataset Import** | Load any CSV file with error handling for missing files and invalid formats |
| 2 | **Dataset Summary** | View observation count, variable names, and data type classifications |
| 3 | **Missing Value Analysis** | Identify missing counts and percentages for every column |
| 4 | **Descriptive Statistics** | Mean, median, mode, SD, variance, min, max, range for all numeric columns |
| 5 | **Frequency Distribution** | Frequency and percentage tables for all categorical columns |
| 6 | **Data Visualization** | Auto-generated histograms, boxplots, and bar charts saved as PNG |
| 7 | **Correlation Analysis** | Pearson correlation matrix with r-values, p-values, and a heatmap |
| 8 | **Hypothesis Testing** | Independent t-Test, Chi-Square Test, Normality Test, ANOVA |
| 9 | **Regression Analysis** | Linear Regression, Multiple Regression Test, Logistic Regression |
| 10 | **Data Processing** | Data Cleaning Tools and Interactive Filtering |
| 11 | **Full Automated Analysis** | Runs all descriptive modules at once |
| 12 | **Report & Export** | Text report to `reports/` and CSV exports to `exports/` |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup

```bash
# Clone or download the project
cd ezpzAnalyze

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running the Application

```bash
python project.py
```

You will be prompted to enter the path to your CSV file:

```
Enter the path to your CSV file: data/sample.csv
```

> **Note:** The program works with **any** CSV file you provide. A `data/sample.csv` is included for demonstration and testing purposes only.

### Menu Options

After loading a dataset, the interactive menu appears:

```
====================================================
         ezpzAnalyze — Research Data Analysis Toolkit
====================================================

  1.  Dataset Summary
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
 13.  Interactive Filtering
 14.  Multiple Regression Test  (formal hypothesis testing)
 15.  Logistic Regression  (binary outcome)
 16.  Full Automated Analysis
 17.  Generate Report
 18.  Export Results
  0.  Exit

====================================================
```

**Recommended first step:** Select option `9` (Full Automated Analysis) to run every module and generate the report, visualizations, and exports in one go.

### Hypothesis Testing

- **Independent t-Test:** You will be asked to select a numeric variable and a grouping variable with exactly 2 groups.
- **Chi-Square Test:** You will be asked to select two categorical variables.
- **ANOVA:** Compares means across 3+ groups and performs Tukey HSD post-hoc testing.
- **Multiple Regression Test:** Provides detailed coefficients, p-values, VIF, and a 4-panel diagnostic plot.
- **Logistic Regression:** Classifies a binary outcome (e.g., Yes/No, 0/1) generating an ROC curve and confusion matrix.

---

## Project Structure

```
ezpzAnalyze/
├── project.py               # Main application — all 10 analysis modules
├── test_project.py           # Pytest test suite (28 tests)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/                     # Place your CSV datasets here
│   └── sample.csv            # Bundled sample dataset (50 rows)
│
├── reports/                  # Auto-generated text reports
│   └── analysis_report.txt
│
├── plots/                    # Auto-generated visualizations
│   ├── histogram_*.png
│   ├── boxplot_*.png
│   ├── barchart_*.png
│   └── correlation_heatmap.png
│
└── exports/                  # Exported CSV results
    ├── descriptive_statistics.csv
    └── frequency_tables.csv
```

> **Note:** The `data/`, `reports/`, `plots/`, and `exports/` directories are created automatically if they don't exist.

---


### Core Functions

| Function | Purpose |
|----------|---------|
| `main()` | Entry point — handles the interactive menu loop |
| `ensure_directories()` | Creates required output directories |
| `load_dataset(filepath)` | Loads and validates CSV files |
| `dataset_summary(df)` | Generates a textual overview of the dataset |
| `missing_value_analysis(df)` | Analyzes and reports missing values |
| `descriptive_statistics(df)` | Computes stats for numeric columns (ddof=1) |
| `frequency_distribution(df)` | Computes frequencies for categorical columns |
| `generate_visualizations(df)` | Creates and saves all chart types |
| `correlation_analysis(df)` | Pearson correlations with heatmap |
| `independent_ttest(df)` | Interactive independent samples t-test |
| `chi_square_test(df)` | Interactive chi-square test of independence |
| `normality_test(df)` | Shapiro-Wilk and Anderson-Darling tests |
| `linear_regression(df)` | Basic Simple/Multiple OLS Regression |
| `one_way_anova(df)` | ANOVA with Levene's test and Tukey HSD |
| `data_cleaning_menu(df)` | Tools for imputation and outlier removal |
| `interactive_filter(df)` | Filters data based on user-defined criteria |
| `multiple_regression_test(df)`| Advanced multiple regression with diagnostics |
| `logistic_regression(df)`| Binary outcome prediction with IRLS and ROC |
| `full_automated_analysis(df)` | Runs all descriptive modules sequentially |
| `generate_report(df)` | Writes a comprehensive text report |
| `export_results(df)` | Exports statistics to CSV files |
| `calculate_mean(values)` | Standalone utility for arithmetic mean |

### Statistical Rigor

- All standard deviations and variances use **`ddof=1`** (sample statistics, not population).
- NaN values are **dropped before** running hypothesis tests to prevent runtime errors.
- Significance level is set at **α = 0.05** for all hypothesis tests.

### Error Handling

The application uses `try-except` blocks to handle:
- `FileNotFoundError` — when the CSV file path is invalid
- `ValueError` — when the file format is unsupported or data is incompatible
- `KeyError` — when a user selects a column that doesn't exist

---




---

## Technology Stack

| Library | Purpose |
|---------|---------|
| **pandas** | Data loading, manipulation, and analysis |
| **numpy** | Numerical calculations |
| **scipy** | Statistical hypothesis tests (t-test, chi-square, Pearson) |
| **matplotlib** | Chart and heatmap generation |
| **os / pathlib** | File system and directory management |
| **pytest** | Automated testing framework |

---

