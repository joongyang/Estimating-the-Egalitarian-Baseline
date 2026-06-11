# Estimating the Egalitarian Baseline: The UD/RUD Framework

This repository contains the official Python implementation for the Unequally Distributed (UD) and Relative UD (RUD) income framework. The codebase provides computational tools to estimate the absolute minimum threshold income (the egalitarian baseline) from complex survey microdata and to calculate norm-based inequality indices.

Traditional dispersion-based metrics, such as the Gini coefficient, often suffer from the "information mixture problem" and require arbitrary data truncation when confronted with negative or zero incomes. This package resolves these structural limitations by utilizing Extreme Value Theory (EVT) to mathematically extract the shared egalitarian baseline prior to inequality evaluation.

## Repository Structure

* `udrud_framework.py`: The core computational module. Contains the EVT algorithms (`estimate_gamma`) and the mathematical indices calculator (`calculate_ud_rud_metrics`).
* `application.ipynb`: A Jupyter Notebook demonstrating a full empirical application of the framework. It includes the exact pipeline used to analyze the equivalised household disposable income from the Korea Household Finance and Welfare Survey (KHFS) for the years 2021–2025.

## Data Availability & Access

**Please Note: The raw microdata is not included in this repository.**

The empirical application in this study utilizes the Korea Household Finance and Welfare Survey (KHFS). These are third-party data provided by the Ministry of Data and Statistics (MODS) via the Microdata Integrated Service (MDIS). 

Under the strict terms of the data use agreement, we are prohibited from redistributing the raw datasets. To replicate this study or apply the code to the KHFS data, researchers must independently obtain the data:

1. Register and apply for data access directly through the MDIS portal at: [https://mdis.mods.go.kr/](https://mdis.mods.go.kr/)
2. Download the KHFS household datasets for the target years.
3. Format your compiled dataset into a CSV file (e.g., `data.csv`) with the following required column headers for the pipeline to function correctly:
   * `year`: The survey year.
   * `weight`: The raw household survey weight.
   * `size`: The number of household members (used for equivalisation).
   * `income`: The raw household disposable income.

## Dependencies

The code is written in Python 3 and requires the following standard scientific libraries:
* `pandas`
* `numpy`

## Quick Start Guide

Once you have obtained and formatted your microdata, you can run the framework. For accurate macroeconomic evaluation, income should be equivalised prior to estimating the baseline.

### 1. Data Preparation and Equivalisation

```python
import pandas as pd
import numpy as np
from udrud_framework import estimate_gamma, calculate_ud_rud_metrics

# Load your locally saved, formatted dataset
df = pd.read_csv("data.csv")

# Equivalise survey weights and income (e.g., using the square root of household size)
df["weight"] = df["weight"] * df["size"]
df["income"] = df["income"] / np.sqrt(df["size"])

y_raw = df['income'].values
weights = df['weight'].values
```

### 2. Estimating the Egalitarian Baseline ($\gamma$)

The estimate_gamma function executes the weight calibration and Cluster-Jackknife algorithms to find the optimal truncation parameter ($k^*$) and the true egalitarian baseline.

```python
# Estimate the minimum threshold using a 5% tail mass target
gamma_hat, optimal_k, plateau_series = estimate_gamma(y_raw, weights, p=0.05)

print(f"Optimal k^*: {optimal_k}")
print(f"Estimated Egalitarian Baseline (Gamma): {gamma_hat}")
```

### 3. Calculating Inequality Indices
   
Once the baseline is established, the calculate_ud_rud_metrics function evaluates the true structural disparity, returning both conventional metrics and the UD/RUD norm indices.

```python
# Calculate inequality metrics
indices = calculate_ud_rud_metrics(y_raw, weights, gamma_hat)

for metric, value in indices.items():
    print(f"{metric}: {value:.4f}")
```

## Citation and Reference

If you utilize this code or the UD/RUD framework in your research, please cite the foundational manuscript:

[Authors pending double-blind review]. "Estimating the Egalitarian Baseline: An Empirical Re-evaluation of Income Inequality via the UD/RUD Framework." Review of Income and Wealth (Under Review).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
