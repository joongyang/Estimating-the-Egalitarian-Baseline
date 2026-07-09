# Estimating the Egalitarian Baseline: The UD/RUD Framework

This repository contains the official Python implementation for the Unequally Distributed (UD) and Relative UD (RUD) income framework. The codebase provides computational tools to estimate the absolute minimum threshold income (the egalitarian baseline) from complex survey microdata and to calculate norm-based inequality indices[cite: 9].

Traditional dispersion-based metrics, such as the Gini coefficient, often suffer from the "information mixture problem" and require arbitrary data truncation when confronted with negative or zero incomes[cite: 9]. This package resolves these structural limitations by utilizing Extreme Value Theory (EVT) to mathematically extract the shared egalitarian baseline prior to inequality evaluation[cite: 9].

## Repository Structure

* `udrud_framework.py`: The core computational module[cite: 9]. Contains the mathematical primitives for the Cooke-base and Cluster-Jackknife estimators, the plateau detection optimizer (`detect_plateau_and_estimate`), and the mathematical indices calculator (`calculate_ud_rud_metrics`).
* `application.ipynb`: A Jupyter Notebook demonstrating a full empirical application of the framework[cite: 9]. It includes the exact pipeline used to analyze the equivalised household disposable income from the Korea Household Finance and Welfare Survey (KHFS) for the years 2021–2025[cite: 6, 9].
* `gen_weibull.py`: A utility script containing the `generate_3p_weibull` function, which generates data from a 3-parameter Weibull distribution for methodological testing[cite: 7].
* `performance_comparison.ipynb`: A Jupyter Notebook that executes Monte Carlo simulations to evaluate the Mean Squared Error (MSE) and Bias of the Cooke-Base versus Cluster-Jackknife estimators across various Weibull shape parameters[cite: 8].
* `sensitivity_kmin.ipynb`: A notebook utilizing parallel processing to test the sensitivity of the threshold estimator to the lower truncation bound parameter[cite: 10].
* `sensitivity_tail_depth.ipynb`: A notebook evaluating the sensitivity of the threshold estimator to changes in the target tail depth proportion[cite: 11].
* `sensitivity_window_size.ipynb`: A notebook evaluating the robustness of the plateau detection against different smoothing window sizes[cite: 12].

## Data Availability & Access

**Please Note: The raw microdata is not included in this repository.**[cite: 9]

The empirical application in this study utilizes the Korea Household Finance and Welfare Survey (KHFS)[cite: 9]. These are third-party data provided by the Ministry of Data and Statistics (MODS) via the Microdata Integrated Service (MDIS)[cite: 9]. 

Under the strict terms of the data use agreement, we are prohibited from redistributing the raw datasets[cite: 9]. To replicate this study or apply the code to the KHFS data, researchers must independently obtain the data:[cite: 9]

1. Register and apply for data access directly through the MDIS portal at: https://mdis.mods.go.kr/[cite: 9]
2. Download the KHFS household datasets for the target years[cite: 9].
3. Format your compiled dataset into a CSV file (e.g., `data.csv`) with the following required column headers for the pipeline to function correctly:[cite: 9]
   * `year`: The survey year[cite: 9].
   * `weight`: The raw household survey weight[cite: 9].
   * `size`: The number of household members (used for equivalisation)[cite: 9].
   * `income`: The raw household disposable income[cite: 9].

## Dependencies

The code is written in Python 3 and requires the following standard scientific libraries:[cite: 9]
* `pandas`[cite: 9]
* `numpy`[cite: 9]
* `scipy`[cite: 8]
* `matplotlib`[cite: 8, 10]
* `joblib`[cite: 10, 11, 12]

## Quick Start Guide

Once you have obtained and formatted your microdata, you can run the framework[cite: 9]. For accurate macroeconomic evaluation, income should be equivalised prior to estimating the baseline[cite: 9].

### 1. Data Preparation and Equivalisation

```python
import pandas as pd
import numpy as np
from udrud_framework import detect_plateau_and_estimate, calculate_ud_rud_metrics

# Load your locally saved, formatted dataset
df = pd.read_csv("data.csv")

# Equivalise survey weights and income (e.g., using the square root of household size)
df["weight"] = df["weight"] * df["size"]
df["income"] = df["income"] / np.sqrt(df["size"])

y_raw = df['income'].values
weights = df['weight'].values
```

### 2. Estimating the Egalitarian Baseline

The detect_plateau_and_estimate function executes the weight calibration and Cluster-Jackknife algorithms to find the optimal truncation parameter and the true egalitarian baseline.

```python
# Aggregate into unique support points and calculate cumulative weights
p = 0.05
N = np.sum(weights)
target_weight = p * N
support_df = pd.DataFrame({'y': y_raw, 'w': weights}).groupby('y').sum().sort_index()
cum_w = support_df['w'].cumsum().values

# Define valid k_range
k_max = np.searchsorted(cum_w, target_weight, side='right')
k_range_obj = range(20, max(21, k_max))

# Estimate the minimum threshold using a 5-period smoothing window
gamma_hat, optimal_k = detect_plateau_and_estimate(y_raw, weights, k_range_obj, window_size=5)

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
