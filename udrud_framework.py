import numpy as np
import pandas as pd

# ==========================================================
# LEVEL 1: THE MATHEMATICAL PRIMITIVES
# ==========================================================
def cookes_base_estimate(y_k):
    # This must be the very first check. No exceptions.
    if len(y_k) == 0: 
        return np.nan
        
    k = len(y_k)
    if k < 2: 
        return y_k[0]
        
    # ==========================================================
    # OPTIMIZATION: Vectorized Cooke's Calculation (No for-loops)
    # ==========================================================
    y_1 = y_k[0]
    
    # Create an array of indices j from 1 to k-1
    j_vals = np.arange(1, k)
    
    # Vectorized calculation of the minimax weights
    xi = (2 * (k - j_vals)) / (k * (k - 1))
    
    # Vectorized calculation of all gaps (y_k[j] - y_k[j-1])
    gaps = np.diff(y_k)
    
    # Compute the final estimate using a highly optimized C-level sum
    estimate = y_1 - np.sum(xi * gaps)
    
    return estimate
def cluster_jackknife(y_k):
    """
    Bias-corrected endpoint estimate using Jackknife on support points.
    Includes robustness checks to prevent NaN propagation.
    """
    k = len(y_k)
    base_est = cookes_base_estimate(y_k)
    
    # Base estimate failure check
    if np.isnan(base_est):
        return np.nan
    
    # SAFETY: Jackknife requires at least 2 points to perform leave-one-out
    if k < 2:
        return base_est
    
    jackknife_sum = 0
    valid_iterations = 0
    
    for m in range(k):
        y_minus_m = np.delete(y_k, m)
        res = cookes_base_estimate(y_minus_m)
        
        # Robustness: Only include valid results in the sum
        if not np.isnan(res):
            jackknife_sum += res
            valid_iterations += 1
            
    # If the jackknife failed to compute sufficient iterations, return NaN
    # as the estimate is unreliable
    if valid_iterations < k:
        return np.nan
        
    return k * base_est - ((k - 1) / k) * jackknife_sum

# ==========================================================
# LEVEL 2: THE DRIVER
# ==========================================================

def get_unique_support(y, w):
    """Aggregates raw data into unique support points."""
    # Force use of .values to strip indices and ensure alignment
    df = pd.DataFrame({'y': np.array(y), 'w': np.array(w)})
    
    support = df.groupby('y', as_index=False)['w'].sum()
    support = support.sort_values('y').reset_index(drop=True)
    return support['y'].values

def calculate_k_max(y, w, p=0.05):
    """
    Calculates the maximum truncation parameter (k_max) based on the target tail depth (p).
    
    Parameters:
    y : array-like - raw income values
    w : array-like - survey weights
    p : float - target tail depth proportion (default: 0.05)
    
    Returns:
    int : The optimal maximum index (k_max) for the unique support points.
    """
    y = np.array(y)
    w = np.array(w)
    
    N = np.sum(w)
    target_weight = p * N
    
    # Aggregate into unique support points and calculate cumulative weights
    df = pd.DataFrame({'y': y, 'w': w})
    support_w = df.groupby('y')['w'].sum().sort_index()
    cum_w = support_w.cumsum().values
    
    # Find the maximum k where cumulative weight <= target_weight
    k_max = np.searchsorted(cum_w, target_weight, side='right')
    
    return k_max

def estimate_gamma_for_k(y, w, k):
    """Calculates gamma for a specific truncation parameter k."""
    # Now expecting only one return value
    y_unique = get_unique_support(y, w) 
    
    # Safety: check if empty (as discussed previously)
    if len(y_unique) == 0:
        return np.nan
        
    k_actual = min(k, len(y_unique))
    y_k = y_unique[:k_actual]
    
    return cluster_jackknife(y_k)

# ==========================================================
# LEVEL 3: THE OPTIMIZER (PLATEAU DETECTION)
# ==========================================================

def detect_plateau_and_estimate(y, w, k_range, window_size=5):
    """
    Level 3 Optimizer: Finds the most stable threshold estimate across a range of k.
    """
    estimates = []
    
    # 1. Sweep and collect weight-calibrated estimates
    for k in k_range:
        # Crucial: This calls the Level 2 Driver, ensuring weights are handled!
        estimates.append(estimate_gamma_for_k(y, w, k))
        
    # 2. Filter out NaN values explicitly
    valid_estimates = pd.Series(estimates, index=k_range).dropna()
    
    # 3. Safety check
    if valid_estimates.empty:
        return np.nan, np.nan
        
    # 4. Calculate rolling std on the valid data
    rolling_std = valid_estimates.rolling(window=window_size).std()
    
    # 5. Identify plateau index (minimize volatility)
    best_k = rolling_std.idxmin()
    
    if pd.isna(best_k):
        return valid_estimates.mean(), valid_estimates.index[0]
        
    return valid_estimates[best_k], best_k

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
        'Gamma^': gamma,
        'mu_y': mu_y,
        'Gini_y': gini_y,
        'CV_y': cv_y,
        'mu_z': mu_z,
        'V_z': var_z,
        'Gini_z': gini_z,
        'CV_z': cv_z,
        'L1_c': L1_c,
        'L1_q': L1_q,
        'L1_cq': L1_cq,
        'L2_c_sq': L2_c_sq,
        'L2_q_sq': L2_q_sq,
        'L2_cq_sq': L2_cq_sq
    }