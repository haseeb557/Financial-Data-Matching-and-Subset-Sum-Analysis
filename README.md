# Fintech Transaction Reconciliation Toolkit

## Overview

This project implements a comprehensive toolkit for financial transaction reconciliation between two Excel data sources (e.g., bank statements vs. customer ledgers). It combines classical algorithms, machine learning, genetic optimization, and fuzzy matching to identify and explain matches between transactions.

Designed for fintech, auditing, and accounting use cases, the toolkit enables:
- Automated reconciliation of mismatched records.
- Advanced matching beyond simple 1:1 lookups.
- Benchmarking and visualization of algorithm performance.
- Extendable ML-based predictions for future reconciliation tasks.
  
## Features
### 1. **Data Preparation & Exploration**
- Handles Excel imports with varying formats.
- Cleans & standardizes amounts (removes symbols, commas, handles negatives).
- Generates unique transaction IDs for traceability.

### 2. **Classical Brute Force Matching**
- Direct 1:1 Matching — exact matches or within tolerance.
- Subset-Sum Brute Force — tries all combinations of transactions up to k size.
- Performance Analysis — time complexity measurement for brute force.

### 3. **Optimized & ML Approaches**
- Dynamic Programming (DP) — efficient subset-sum solver.
- Machine Learning Regression Pipeline:
- Feature engineering from subsets (sum, diff, fuzzy score, stats).
- Regression models (RandomForest, LinearRegression) to predict best matches.
- Ranking of likely matches with predicted error.

### 4. **Advanced Techniques**
- Genetic Algorithm (GA) for approximate subset search.
- Fuzzy Matching using RapidFuzz or difflib (string similarity on references/descriptions).

### 5. **Performance Comparison & Visualization**
- Benchmarking across: Brute Force, DP, GA, ML.
- Time measurements across increasing dataset sizes.
- Visualizations with log-scaled runtime plots.
- Excel report generation with:
  - Matches
  - Fuzzy Suggestions
  - Benchmarks
  - ML Info
  - Candidate Rankings
 
## Algorithms Implemented

| Method            | Type              | Strengths                     | Limitations                                  |
|-------------------|-------------------|--------------------------------|----------------------------------------------|
| **Direct Matching** | Exact Search       | Fast, simple                  | Only works on exact (or near tolerance) amounts |
| **Brute Force**     | Exhaustive Search  | Finds all small subsets       | Exponential runtime, only feasible for k < 8 |
| **Dynamic Programming** | Optimized Search   | Efficient for larger sets     | Limited by memory, assumes mostly non-negative amounts |
| **Genetic Algorithm**  | Heuristic Search   | Handles larger noisy datasets | Approximate, may miss exact matches          |
| **ML Regression**     | Predictive Model   | Learns matching patterns      | Requires labeled/synthetic training data     |
| **Fuzzy Matching**    | Similarity Search  | Matches descriptions, references | Not amount-aware, string similarity only   |

## Usage

### 1. Install the dependencies
```bash
pip install numpy pandas matplotlib xlsxwriter scikit-learn rapidfuzz openpyxl
``` 

### 2. Run the Python file
```bash
python3 fintech_reconcile.py \
  --transactions sample/Customer_Ledger_Entries_FULL.xlsx \
  --targets sample/KH_Bank.XLSX \
  --max-subset-size 8 \
  --tolerance 100 \
  --fuzzy \
  --ga \
  --ml --ml-model rf --ml-max-k 3 --ml-negatives 50 \
  --benchmark \
  --plots-dir sample/plots \
  --report sample/out.xlsx
```
  
### 3. Command-line options

Flag	Description:
- --transactions :	Path to Excel file with ledger transactions
- --targets :	Path to Excel file with bank statement targets
- --max-subset-size :	Max size of transaction subsets (brute force)
- --tolerance :	Allowed difference in cents (e.g., 100 = ±1.00)
- --fuzzy	: Enable fuzzy description/reference matching
- --ga : Enable Genetic Algorithm fallback
- --ml : Enable ML pipeline
- --ml-model : ML model (rf = Random Forest, logreg = Logistic Regression)
- --ml-max-k : Max subset size considered in ML candidates
--ml-negatives : Random negative samples per target for ML training
- --benchmark : Run benchmarking suite
- --plots-dir : Directory to save runtime plots
- --report : Path to save final Excel report
  
## Expected Output
- Excel Report (out.xlsx)
- Matches (direct, brute force, DP, GA, ML).
- Fuzzy Suggestions.
- Benchmark results.
- ML Info (model metrics, features).
- ML Candidate rankings.
- Benchmark Plots (sample/plots/)
- Runtime comparison of Brute Force, DP, GA, ML.
