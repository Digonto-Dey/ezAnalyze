# ezAnalysis: Research Data Analysis Toolkit

## Project Overview

The Research Data Analysis Toolkit is a terminal-based Python application that automatically performs statistical analysis on CSV datasets. The goal is to create a simplified version of software such as SPSS, Stata, or R that can be used directly from the command line.

The application should allow a user to load a dataset and generate a complete statistical report including descriptive statistics, missing value analysis, frequency distributions, visualizations, correlation analysis with correlation heatmap, and hypothesis testing.

# Project Objectives

The toolkit should:

- Import CSV datasets
- Validate and inspect data
- Analyze missing values
- Produce descriptive statistics
- Generate frequency distributions
- Create visualizations
- Perform correlation analysis
- Perform statistical hypothesis tests
- Generate a complete report
- Save outputs automatically

# Technology Stack

## Required Libraries

pip install pandas  
pip install numpy  
pip install scipy  
pip install matplotlib

### Library Purpose

| Library | Purpose |
| --- | --- |
| pandas | Data loading and manipulation |
| numpy | Numerical calculations |
| scipy | Statistical tests |
| matplotlib | Plot generation |
| os  | File management |
| pathlib | Directory management |

# Recommended Project Structure

research_toolkit/  
<br/>│  
├── project.py  
├── test_project.py  
├── README.md  
├── requirements.txt  
│  
├── data/  
│ └── sample.csv  
│  
├── reports/  
│ └── analysis_report.txt  
│  
├── plots/  
│ ├── histogram_age.png  
│ ├── boxplot_income.png  
│ └── correlation_heatmap.png  
│  
└── exports/  
└── descriptive_statistics.csv

# Functional Requirements

## Module 1: Dataset Import

### Goal

Load CSV datasets safely.

### Features

- Read CSV files
- Handle invalid file paths
- Handle unsupported formats
- Display dataset dimensions

### Example Output

Dataset Loaded Successfully  
<br/>Rows: 500  
Columns: 12

### Function

load_dataset(filepath)

Returns:

pandas.DataFrame

## Module 2: Dataset Summary

### Goal

Provide a quick overview.

### Information Displayed

- Number of observations
- Number of variables
- Variable names
- Variable data types

### Example

Variables:  
<br/>Age  
Gender  
Income  
GPA  
<br/>Data Types:  
<br/>Age -> Numeric  
Gender -> Categorical

### Function

dataset_summary(df)

## Module 3: Missing Value Analysis

### Goal

Identify incomplete data.

### Statistics

For every variable:

- Missing count
- Missing percentage

### Example

Missing Values  
<br/>Age 5 (1.0%)  
Income 12 (2.4%)

### Function

missing_value_analysis(df)

## Module 4: Descriptive Statistics

### Goal

Analyze numerical variables.

### Statistics

For every numeric column:

- Count
- Mean
- Median
- Mode
- Standard Deviation
- Variance
- Minimum
- Maximum
- Range

### Example

Variable: Age  
<br/>Mean = 23.4  
Median = 22  
Mode = 21  
SD = 4.1  
Variance = 16.81  
Min = 18  
Max = 42

### Function

descriptive_statistics(df)

## Module 5: Frequency Distribution

### Goal

Analyze categorical variables.

### Statistics

- Frequency
- Percentage

### Example

Gender  
<br/>Male 120 (60%)  
Female 80 (40%)

### Function

frequency_distribution(df)

## Module 6: Data Visualization

### Goal

Automatically generate charts.

### Charts

#### Histogram

For numeric variables.

histogram(variable)

#### Boxplot

For numeric variables.

boxplot(variable)

#### Bar Chart

For categorical variables.

bar_chart(variable)

### Output Folder

plots/

### Example

Histogram Saved:  
plots/histogram_age.png

## Module 7: Correlation Analysis

### Goal

Determine relationships among numerical variables.

### Methods

#### Pearson Correlation

For continuous variables.

#### Spearman Correlation

Optional bonus feature.

### Example

Correlation Analysis  
<br/>Age vs Income  
<br/>r = 0.68  
p = 0.002

### Function

correlation_analysis(df)

## Module 8: Hypothesis Testing

### Goal

Perform common statistical tests.

### Independent Samples t-Test

Example:

Male GPA vs Female GPA  
<br/>t = 2.15  
p = 0.032  
<br/>Result:  
Significant Difference

Function:

independent_ttest(df)

### Chi-Square Test

Example:

Gender vs Smoking  
<br/>Chi-square = 8.32  
p = 0.003  
<br/>Result:  
Significant Association

Function:

chi_square_test(df)

## Module 9: Automatic Report Generation

### Goal

Create a text report containing all analyses.

### Output

reports/analysis_report.txt

### Contents

1.  Dataset Information
2.  Missing Value Summary
3.  Descriptive Statistics
4.  Frequency Tables
5.  Correlation Analysis
6.  Hypothesis Testing Results

Function:

generate_report(df)

## Module 10: Export Results

### Goal

Allow users to save analysis outputs.

### Files

exports/descriptive_statistics.csv  
exports/frequency_tables.csv

Function:

export_results()

# User Workflow

## Step 1

Run:

python project.py

## Step 2

Enter dataset path:

Enter CSV file path:

Example:

data/students.csv

## Step 3

Program loads dataset.

Dataset Loaded Successfully

## Step 4

Display Menu

\====================================  
RESEARCH DATA ANALYSIS TOOLKIT  
\====================================  
<br/>1\. Dataset Summary  
2\. Missing Value Analysis  
3\. Descriptive Statistics  
4\. Frequency Distribution  
5\. Generate Visualizations  
6\. Correlation Analysis  
7\. Independent t-Test  
8\. Chi-Square Test  
9\. Full Automated Analysis  
10\. Generate Report  
11\. Exit

## Step 5

If user selects:

9

Program performs everything automatically.

# CS50P Requirements Compliance

The project should demonstrate:

### Functions

Multiple reusable functions.

### Loops

Menu navigation.

### Conditionals

Error handling and menu choices.

### File I/O

CSV import and report export.

### Exceptions

Try-except blocks.

### Libraries

Pandas, NumPy, SciPy, Matplotlib.

### Unit Tests

Create:

test_project.py

Test at least:

load_dataset()  
<br/>calculate_mean()  
<br/>missing_value_analysis()  
<br/>correlation_analysis()

using pytest.

# Final Deliverables

1.  project.py
2.  test_project.py
3.  README.md
4.  requirements.txt
5.  Sample dataset
6.  Generated report
7.  Generated plots
8.  CS50P project demonstration video

Project Goal:

A user should be able to run:

python project.py

load a CSV dataset, select “Full Automated Analysis”, and receive a complete statistical report with visualizations and hypothesis testing results directly from the terminal.