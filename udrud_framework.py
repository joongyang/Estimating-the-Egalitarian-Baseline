import numpy as np
import pandas as pd

# ==========================================
# MODULE 1: THRESHOLD ESTIMATION (EVT)
# ==========================================

def get_unique_support(y, w):
    """Aggregates raw data into unique support points and cumulative weights."""
    df = pd.DataFrame({'y': y, 'w': w})
    # Group by unique income values and sum the weights
    support = df.groupby('y', as_index=False)['w'].sum()
    support = support.sort_values('y').reset_index(drop=True)
    return support['y'].values, support['w'].values

def cookes_base_estimate(y_k):
    """Calculates the base Cooke's spacing estimator for k unique points."""
    k = len(y_k)
    if k < 2:
        return y_k[0]
    
    y_1 = y_k[0]
    estimate = y_1
    
    for j in range(1, k):
        # Cooke's optimal minimax weights for alpha=2
        xi_j = (2 * (k - j)) / (k * (k - 1))
        gap = y_k[j] - y_k[j-1]
        estimate -= xi_j * gap
        
    return estimate

def cluster_jackknife(y_k):
    """Calculates the Jackknife bias-corrected endpoint estimate."""
    k = len(y_k)
    base_est = cookes_base_estimate(y_k)
    
    leave_one_out_estimates = []
    for m in range(k):
        # Omit the m-th unique order statistic
        y_k_minus_m = np.delete(y_k, m)
        loo_est = cookes_base_estimate(y_k_minus_m)
        leave_one_out_estimates.append(loo_est)
        
    # Jackknife bias correction formula
    jackknife_est = k * base_est - ((k - 1) / k) * np.sum(leave_one_out_estimates)
    return jackknife_est

def estimate_gamma(y, w, p=0.05, window_size=5):
    """
    Executes Algorithms 1, 2, and 3 to find the optimal minimum threshold.
    p: fractional depth of the tail.
    window_size: rolling window for plateau detection.
    """
    y_unique, w_unique = get_unique_support(y, w)
    
    total_N = np.sum(w_unique)
    cumulative_w = np.cumsum(w_unique)
    
    # Algorithm 1: Weight Calibration
    valid_k_indices = np.where(cumulative_w <= p * total_N)[0]
    if len(valid_k_indices) < window_size:
        raise ValueError("Tail fraction p is too small to form a plateau window.")
    
    K_max = valid_k_indices[-1] + 1 
    
    # Algorithm 2: Cluster-Jackknife over k continuum
    k_values = range(window_size, K_max + 1)
    gamma_estimates = []
    
    for k in k_values:
        y_k = y_unique[:k]
        gamma_est = cluster_jackknife(y_k)
        gamma_estimates.append(gamma_est)
        
    estimates_series = pd.Series(gamma_estimates, index=k_values)
    
    # Algorithm 3: Empirical Plateau Detection
    rolling_var = estimates_series.rolling(window=window_size, center=True).var()
    
    # Find the k that minimizes the rolling variance
    optimal_k = rolling_var.idxmin()
    optimal_gamma = estimates_series.loc[optimal_k]
    
    return optimal_gamma, optimal_k, estimates_series

# ==========================================
# MODULE 2: UD/RUD METRICS ENGINE
# ==========================================

def weighted_mean(val, w):
    return np.average(val, weights=w)

def weighted_variance(val, w):
    mean_val = weighted_mean(val, w)
    return np.average((val - mean_val)**2, weights=w)

def weighted_gini(val, w):
    """Calculates the weighted Gini coefficient."""
    # Sort values and weights
    sorted_indices = np.argsort(val)
    val_sorted = val[sorted_indices]
    w_sorted = w[sorted_indices]
    
    # Cumulative weight fractions
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    
    # Expected value (mean)
    mean_val = weighted_mean(val_sorted, w_sorted)
    if mean_val == 0:
        return 0.0
        
    # Gini calculation via covariance method for weighted data
    w_prop = w_sorted / total_w
    cum_w_prop = np.cumsum(w_prop)
    
    # F(y) is the cumulative distribution function
    F = cum_w_prop - (w_prop / 2)
    
    cov = np.cov(val_sorted, F, aweights=w_prop)[0][1]
    return 2 * cov / mean_val

def calculate_ud_rud_metrics(y, w, gamma):
    """
    Calculates the conventional and UD/RUD inequality metrics.
    """
    y = np.array(y)
    w = np.array(w)
    
    # Calculate conventional parameters
    mu_y = weighted_mean(y, w)
    gini_y = weighted_gini(y, w)
    cv_y = np.sqrt(weighted_variance(y, w)) / mu_y if mu_y != 0 else 0
    
    # Map to UD and RUD space
    x = y - gamma          # UD Income
    z = x / mu_y           # RUD Income
    
    # RUD Parameters
    mu_z = weighted_mean(z, w)
    var_z = weighted_variance(z, w)
    cv_z = np.sqrt(var_z) / mu_z if mu_z != 0 else 0
    gini_z = weighted_gini(z, w)
    
    # L1 Norms
    L1_c = mu_z
    L1_q = mu_z
    L1_cq = 2 * mu_z
    
    # Squared L2 Norms
    L2_c_sq = mu_z * (1 - gini_z)
    L2_q_sq = (mu_z ** 2) + var_z
    L2_cq_sq = L2_c_sq + L2_q_sq
    
    return {
        'Baseline_Gamma': gamma,
        'Mean_Income_y': mu_y,
        'Conventional_Gini': gini_y,
        'Conventional_CV': cv_y,
        'RUD_Mean_mu_z': mu_z,
        'RUD_Variance_V_z': var_z,
        'RUD_Gini_G_z': gini_z,
        'RUD_CV_z': cv_z,
        'L1_c (CDF)': L1_c,
        'L1_q (QF)': L1_q,
        'L1_cq (CDQF)': L1_cq,
        'L2_c_sq (CDF)': L2_c_sq,
        'L2_q_sq (QF)': L2_q_sq,
        'L2_cq_sq (CDQF)': L2_cq_sq
    }