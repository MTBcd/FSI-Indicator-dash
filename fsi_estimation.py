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
        f_t_std = X_t[-1, :]
        fsi_series[t] = np.dot(f_t_std, omega)
    return omega_history, fsi_series






# def estimate_fsi_recursive_rolling_with_stability(
#     df, window_size=125, n_iter=150, stability_threshold=0.7
# ):
#     df = df.dropna()
#     arr = np.ascontiguousarray(df.values)
#     dates = df.index[window_size-1:]
#     columns = df.columns

#     # Batched rolling windows
#     roll = rolling_window(arr, window_size)  # (num_windows, window_size, N)
#     roll = np.ascontiguousarray(roll)

#     omega_history, fsi_series = parallel_fsi_windows(roll, arr, n_iter)

#     # Cosine similarity (serial, negligible cost)
#     stability_series = np.zeros(len(omega_history))
#     flagged_idx = []
#     prev_omega = None
#     # for t in range(len(omega_history)):
#     #     omega = omega_history[t]
#     #     if prev_omega is not None:
#     #         cos_sim = np.dot(prev_omega, omega) / (np.linalg.norm(prev_omega) * np.linalg.norm(omega))
#     #         if cos_sim < 0:
#     #             omega *= -1
#     #             cos_sim *= -1
#     #         stability_series[t] = cos_sim
#     #         if cos_sim < stability_threshold:
#     #             flagged_idx.append(t)
#     #     else:
#     #         stability_series[t] = 1.0
#     #     prev_omega = omega.copy()
#     #     omega_history[t] = omega  # correct direction if flipped

#     # inside the stability loop in estimate_fsi_recursive_rolling_with_stability

#     prev_omega = None
#     stability_series = np.zeros(len(omega_history))
#     flagged_idx = []

#     for t in range(len(omega_history)):
#         omega = omega_history[t]
#         if prev_omega is not None:
#             cos_sim = np.dot(prev_omega, omega) / (np.linalg.norm(prev_omega) * np.linalg.norm(omega))
#             if cos_sim < 0:
#                 # flip BOTH omega and current FSI so the two remain consistent
#                 omega *= -1.0
#                 fsi_series[t] *= -1.0
#                 cos_sim *= -1.0
#             stability_series[t] = cos_sim
#             if cos_sim < stability_threshold:
#                 flagged_idx.append(t)
#         else:
#             stability_series[t] = 1.0

#         omega_history[t] = omega
#         prev_omega = omega.copy()

#     # Convert back to pandas
#     fsi_series_pd = pd.Series(fsi_series, index=dates)
#     omega_df = pd.DataFrame(omega_history, index=dates, columns=columns)
#     stability_series_pd = pd.Series(stability_series, index=dates)
#     flagged_dates = [dates[i] for i in flagged_idx]

#     return fsi_series_pd, omega_df, stability_series_pd, flagged_dates




def estimate_fsi_expanding_with_als(
    df,
    min_history=250,
    n_iter=150,
    stability_threshold=0.7
):
    """
    Expanding-window ALS-based FSI:
    - At each t >= min_history-1, use all rows 0..t.
    - Standardize each column using mean/std over 0..t (no look-ahead).
    - Use fast_als (Numba) to extract the first principal component on Z_t.
    - FSI_t = z_t · omega_t, where z_t is standardized row at time t.

    Returns:
        fsi_series_pd       : pd.Series (index = dates)
        omega_df            : pd.DataFrame (index = dates, columns = df.columns)
        stability_series_pd : pd.Series (cosine similarity between omegas)
        flagged_dates       : list of dates where stability < threshold
    """
    import numpy as np
    import pandas as pd
    import logging

    # Ensure sorted index and no all-NaN columns
    df = df.sort_index().copy()
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        logging.error("Not enough columns for FSI estimation.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), []

    # If you want to allow some NaNs, you should impute before this call.
    # Here we assume df has been imputed already (no NaNs), like in your original pipeline.
    if df.isna().any().any():
        logging.warning("[ALS_EXPANDING] df still has NaNs; filling with 0 for ALS.")
        df = df.fillna(0.0)

    X = np.ascontiguousarray(df.values, dtype=np.float64)
    dates = df.index
    T, N = X.shape

    if T < min_history:
        logging.error(f"Not enough time points ({T}) for min_history={min_history}.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float), []

    fsi_vals = []
    omega_list = []
    out_dates = []

    for t in range(min_history - 1, T):
        # Use all data up to and including t
        X_t = X[:t+1, :].copy()   # shape (t+1, N)

        # Expanding mean/std up to t
        mu_t = X_t.mean(axis=0)
        sd_t = X_t.std(axis=0, ddof=0)
        sd_t[sd_t == 0] = 1.0  # avoid division by zero; column becomes 0 after standardization

        Z_t = (X_t - mu_t) / sd_t    # shape (t+1, N)
        Z_t = np.ascontiguousarray(Z_t, dtype=np.float64)

        # Numba ALS on the standardized expanding window
        omega_t = fast_als(Z_t, n_iter)   # uses your existing Numba function
        # ^ fast_als does SVD init + ALS iterations, and normalizes omega inside

        # FSI_t is projection of current standardized observation onto omega_t
        z_current = Z_t[-1, :]       # standardized last row at time t
        fsi_t = float(np.dot(z_current, omega_t))

        fsi_vals.append(fsi_t)
        omega_list.append(omega_t.copy())
        out_dates.append(dates[t])

    # Convert to arrays and then pandas
    fsi_vals = np.array(fsi_vals, dtype=np.float64)
    omega_arr = np.vstack(omega_list)     # shape (num_times, N)
    out_dates = pd.to_datetime(out_dates)

    # --- Stability & sign-consistency (same pattern as your rolling version) ---
    stability_series = np.zeros(len(omega_arr))
    flagged_idx = []
    prev_omega = None

    for i in range(len(omega_arr)):
        omega = omega_arr[i, :]
        if prev_omega is not None:
            num = np.dot(prev_omega, omega)
            den = (np.linalg.norm(prev_omega) * np.linalg.norm(omega))
            cos_sim = num / den if den != 0 else 0.0

            if cos_sim < 0:
                # keep direction continuous: flip omega and FSI
                omega *= -1.0
                fsi_vals[i] *= -1.0
                cos_sim *= -1.0

            stability_series[i] = cos_sim
            if cos_sim < stability_threshold:
                flagged_idx.append(i)
        else:
            stability_series[i] = 1.0  # first one is "perfectly stable" by convention

        omega_arr[i, :] = omega
        prev_omega = omega.copy()

    # Convert to pandas objects
    fsi_series_pd = pd.Series(fsi_vals, index=out_dates, name="FSI")
    omega_df = pd.DataFrame(omega_arr, index=out_dates, columns=df.columns)
    stability_series_pd = pd.Series(stability_series, index=out_dates, name="cosine_stability")
    flagged_dates = [out_dates[i] for i in flagged_idx]

    return fsi_series_pd, omega_df, stability_series_pd, flagged_dates





def compute_variable_contributions(df, omega):
    """Compute contributions of each variable to the FSI."""
    try:
        df_std = (df - df.mean()) / df.std()
        omega = np.array(omega)
        contribs = df_std.multiply(omega, axis=1)
        contribs['FSI'] = contribs.sum(axis=1)
        return contribs
    except Exception as e:
        logging.error(f"Error computing variable contributions: {e}", exc_info=True)
        return pd.DataFrame()




def compute_timevarying_contributions(df, omega_history, min_history):
    """
    Leakage-free contributions, consistent with estimate_fsi_expanding_with_als:
    At time t use z_t ⊙ ω_t, where

        z_tj = (x_tj - μ_tj) / σ_tj

    and μ_tj, σ_tj are computed over ALL observations up to t
    (expanding window, no look-ahead).

    Inputs:
      df            : DataFrame of engineered features (index=Date, cols=features)
      omega_history : DataFrame of loadings (index ⊆ df.index, cols=features),
                      e.g. output of estimate_fsi_expanding_with_als
      min_history   : int, same as you used for min_history in the estimator

    Returns:
      DataFrame of variable contributions per date, with 'FSI' column as row-sum.
    """
    df = df.sort_index()

    # Expanding μ, σ up to each t (no look-ahead)
    mu = df.expanding(min_periods=min_history).mean()
    sd = df.expanding(min_periods=min_history).std().replace(0, np.nan)

    # Align to ω dates
    mu = mu.reindex(omega_history.index)
    sd = sd.reindex(omega_history.index)
    X  = df.reindex(omega_history.index)

    # Standardize with the same expanding stats used by the estimator
    Z = (X - mu) / sd

    # Common columns, same order
    common = omega_history.columns.intersection(Z.columns)
    Zc = Z[common]
    Om = omega_history[common]

    contribs = Zc * Om             # elementwise z_tj * ω_tj
    contribs['FSI'] = contribs.sum(axis=1)
    return contribs





# def compute_timevarying_contributions(df, omega_history, window_size):
#     """
#     Leakage-free contributions: at time t use z_t ⊙ ω_t,
#     where z_t = (x_t - mu_t) / sd_t with mu_t, sd_t computed over the LAST 'window_size' observations ending at t.
#     Inputs:
#       df            : DataFrame of engineered features (index=Date, cols=features)
#       omega_history : DataFrame of rolling loadings (index aligns to df[window-1:], cols=features)
#       window_size   : int, the same window size used in the ALS estimator
#     Returns:
#       DataFrame of variable contributions per date, with 'FSI' column as row-sum.
#     """
#     # Ensure chronological order & alignment
#     df = df.sort_index()
#     # Rolling μ, σ computed exactly like the estimator (window stats)
#     mu = df.rolling(window_size).mean().iloc[window_size-1:]
#     sd = df.rolling(window_size).std().replace(0, np.nan).iloc[window_size-1:]

#     # Align to ω dates
#     mu = mu.reindex(omega_history.index)
#     sd = sd.reindex(omega_history.index)
#     X = df.reindex(omega_history.index)

#     # Standardize per window (same as estimator)
#     Z = (X - mu) / sd

#     # Use only common columns, same order
#     common = omega_history.columns.intersection(Z.columns)
#     Zc = Z[common]
#     Om = omega_history[common]

#     contribs = Zc * Om  # elementwise: z_tj * ω_tj
#     contribs['FSI'] = contribs.sum(axis=1)
#     return contribs

