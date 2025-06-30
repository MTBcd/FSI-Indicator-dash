# fsi_estimation.py
import numpy as np
import pandas as pd
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from scipy.stats import rankdata
from statsmodels.tsa.seasonal import STL
from numba import njit, prange


def rolling_window(arr, window):
    shape = (arr.shape[0] - window + 1, window, arr.shape[1])
    strides = (arr.strides[0], arr.strides[0], arr.strides[1])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

@njit
def mean_axis0(a):
    nrow, ncol = a.shape
    out = np.zeros(ncol)
    for j in range(ncol):
        s = 0.0
        for i in range(nrow):
            s += a[i, j]
        out[j] = s / nrow
    return out

@njit
def std_axis0(a):
    nrow, ncol = a.shape
    means = mean_axis0(a)
    out = np.zeros(ncol)
    for j in range(ncol):
        s = 0.0
        for i in range(nrow):
            diff = a[i, j] - means[j]
            s += diff * diff
        out[j] = np.sqrt(s / nrow)
    return out

@njit
def fast_als(X_t, n_iter):
    U, S, Vt = np.linalg.svd(X_t, full_matrices=False)
    omega = Vt[0]
    omega /= np.linalg.norm(omega)
    prev_omega = omega.copy()
    for _ in range(n_iter):
        f = X_t @ omega / (omega @ omega)
        omega = X_t.T @ f / (f @ f)
        norm = np.linalg.norm(omega)
        if norm != 0:
            omega /= norm
        else:
            omega = prev_omega
        prev_omega = omega.copy()
    return omega

@njit(parallel=True)
def parallel_fsi_windows(roll, arr, n_iter):
    num_windows, window, N = roll.shape
    omega_history = np.zeros((num_windows, N))
    fsi_series = np.zeros(num_windows)
    for t in prange(num_windows):   # parallel loop
        X_t = roll[t].copy()
        # Standardize window (manual for Numba)
        m = mean_axis0(X_t)
        s = std_axis0(X_t)
        for i in range(window):
            for j in range(N):
                if s[j] != 0:
                    X_t[i, j] = (X_t[i, j] - m[j]) / s[j]
                else:
                    X_t[i, j] = 0.0
        omega = fast_als(X_t, n_iter)
        omega_history[t] = omega
        f_t = arr[t + window - 1]
        fsi_series[t] = np.dot(f_t, omega)
    return omega_history, fsi_series

def estimate_fsi_recursive_rolling_with_stability(
    df, window_size=125, n_iter=150, stability_threshold=0.7
):
    df = df.dropna()
    arr = df.values
    dates = df.index[window_size-1:]
    columns = df.columns

    # Batched rolling windows
    roll = rolling_window(arr, window_size)  # (num_windows, window_size, N)
    omega_history, fsi_series = parallel_fsi_windows(roll, arr, n_iter)

    # Cosine similarity (serial, negligible cost)
    stability_series = np.zeros(len(omega_history))
    flagged_idx = []
    prev_omega = None
    for t in range(len(omega_history)):
        omega = omega_history[t]
        if prev_omega is not None:
            cos_sim = np.dot(prev_omega, omega) / (np.linalg.norm(prev_omega) * np.linalg.norm(omega))
            if cos_sim < 0:
                omega *= -1
                cos_sim *= -1
            stability_series[t] = cos_sim
            if cos_sim < stability_threshold:
                flagged_idx.append(t)
        else:
            stability_series[t] = 1.0
        prev_omega = omega.copy()
        omega_history[t] = omega  # correct direction if flipped

    # Convert back to pandas
    fsi_series_pd = pd.Series(fsi_series, index=dates)
    omega_df = pd.DataFrame(omega_history, index=dates, columns=columns)
    stability_series_pd = pd.Series(stability_series, index=dates)
    flagged_dates = [dates[i] for i in flagged_idx]

    return fsi_series_pd, omega_df, stability_series_pd, flagged_dates


def compute_variable_contributions(df, omega):
    """Compute contributions of each variable to the FSI."""
    try:
        df_std = (df - df.mean()) / df.std()
        omega = np.array(omega)
        contribs = df_std.multiply(omega / np.dot(omega, omega), axis=1)
        contribs['FSI'] = contribs.sum(axis=1)
        return contribs
    except Exception as e:
        logging.error(f"Error computing variable contributions: {e}", exc_info=True)
        return pd.DataFrame()

