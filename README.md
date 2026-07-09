# Estimating the Egalitarian Baseline: The UD/RUD Framework

This repository contains the official Python implementation for the Unequally Distributed (UD) and Relative UD (RUD) income framework. The codebase provides computational tools to estimate the absolute minimum threshold income (the egalitarian baseline) from complex survey microdata and to calculate norm-based inequality indices.

Traditional dispersion-based metrics, such as the Gini coefficient, often suffer from the "information mixture problem" and require arbitrary data truncation when confronted with negative or zero incomes. This package resolves these structural limitations by utilizing Extreme Value Theory (EVT) to mathematically extract the shared egalitarian baseline prior to inequality evaluation.

## Repository Structure

* `udrud_framework.py`: The core computational module. Contains the mathematical primitives for the Cooke-base and Cluster-Jackknife estimators, the plateau detection optimizer (`detect_plateau_and_estimate`), the maximum threshold calculator (`calculate_k_max`), and the mathematical indices calculator (`calculate_ud_rud_metrics`).
* `application.ipynb`: A Jupyter Notebook demonstrating a full empirical application of the framework. It includes the exact pipeline used to analyze the equivalised household disposable income from the Korea Household Finance and Welfare Survey (KHFS) for the years 2021–2025.
* `gen_weibull.py`: A utility script containing the `generate_3p_weibull` function, which generates data from a 3-parameter Weibull distribution for methodological testing.
* `performance_comparison.ipynb`: A Jupyter Notebook that executes Monte Carlo simulations to evaluate the Mean Squared Error (MSE) and Bias of the Cooke-Base versus Cluster-Jackknife estimators across various Weibull shape parameters.
* `sensitivity_kmin.ipynb`: A notebook utilizing parallel processing to test the sensitivity of the threshold estimator to the lower truncation bound parameter.
* `sensitivity_tail_depth.ipynb`: A notebook evaluating the sensitivity of the threshold estimator to changes in the target tail depth proportion.
* `sensitivity_window_size.ipynb`: A notebook evaluating the robustness of the plateau detection against different smoothing window sizes.

## Data Availability & Access

**Please Note: The raw microdata is not included in this repository.**

The empirical application in this study utilizes the Korea Household Finance and Welfare Survey (KHFS). These are third-party data provided by the Ministry of Data and Statistics (MODS) via the Microdata Integrated Service (MDIS). 

Under the strict terms of the data use agreement, we are prohibited from redistributing the raw datasets. To replicate this study or apply the code to the KHFS data, researchers must independently obtain the data.

1. Register and apply for data access directly through the MDIS portal at: https://mdis.mods.go.kr/
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
* `scipy`
* `matplotlib`
* `joblib`

## Quick Start Guide

Once you have obtained and formatted your microdata, you can run the framework. For accurate macroeconomic evaluation, income should be equivalised prior to estimating the baseline.

### 1. Data Preparation and Equivalisation

```python
import pandas as pd
import numpy as np
from udrud_framework import calculate_k_max, detect_plateau_and_estimate, calculate_ud_rud_metrics

# Load your locally saved, formatted dataset
df = pd.read_csv("data.csv")

# Equivalise survey weights and income (e.g., using the square root of household size)
df["weight"] = df["weight"] * df["size"]
df["income"] = df["income"] / np.sqrt(df["size"])

y_raw = df['income'].values
weights = df['weight'].values
```

### 2. Estimating the Egalitarian Baseline

Utilize the calculate_k_max function to determine the target tail depth bound, then execute detect_plateau_and_estimate to find the optimal truncation parameter and the true egalitarian baseline.

```python
# 1. Determine the maximum truncation parameter (k_max) for a 5% tail depth
k_max = calculate_k_max(y_raw, weights, p=0.05)

# 2. Define the search range (e.g., starting safely from the 20th unique order statistic)
k_range = range(20, max(21, k_max))

# 3. Estimate the minimum threshold using a 5-period smoothing window
gamma_hat, optimal_k = detect_plateau_and_estimate(y_raw, weights, k_range, window_size=5)

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

"A Distribution-Free Cluster-Jackknife Estimator for the Minimum Threshold of the Unequally Distributed Income Framework." Journal of Business & Economic Statistics (Under Review).

## License

This project is licensed under the MIT License.
