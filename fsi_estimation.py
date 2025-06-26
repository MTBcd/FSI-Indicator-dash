# fsi_estimation.py
import numpy as np
import pandas as pd
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from scipy.stats import rankdata
from pykalman import KalmanFilter
from statsmodels.tsa.seasonal import STL


def estimate_fsi_recursive_rolling_with_stability(df, window_size=125, n_iter=300, stability_threshold=0.7):
    """
    Estimates FSI using a fixed-length rolling window and tracks ω stability via cosine similarity.

    Parameters:
        df (pd.DataFrame): Z-score standardized input data.
        window_size (int): Size of the rolling window.
        n_iter (int): ALS iteration count.
        stability_threshold (float): Minimum acceptable cosine similarity for ω stability.

    Returns:
        fsi_series (pd.Series): FSI values.
        omega_df (pd.DataFrame): Loadings at each step.
        stability_series (pd.Series): Cosine similarity of ω_t vs ω_{t-1}
        flagged_dates (List[str]): Dates where ω_t stability < threshold
    """
    try:
        df = df.dropna()
        columns = df.columns
        fsi_series = []
        omega_history = []
        stability_series = []

        prev_omega = None

        for t in range(window_size, len(df)):
            X_window = df.iloc[t - window_size:t]
            X_t = X_window.values
            X_t = (X_t - X_t.mean(axis=0)) / X_t.std(axis=0)

            # === Init via PCA ===
            _, _, Vt = np.linalg.svd(X_t, full_matrices=False)
            omega = Vt[0]
            omega /= np.linalg.norm(omega)

            # === ALS Iteration ===
            for _ in range(n_iter):
                f = X_t @ omega / np.dot(omega, omega)
                omega = X_t.T @ f / np.dot(f, f)
                omega /= np.linalg.norm(omega)

            # === Cosine Similarity ===
            cos_sim = None
            if prev_omega is not None:
                dot = np.dot(prev_omega, omega)
                cos_sim = dot / (np.linalg.norm(prev_omega) * np.linalg.norm(omega))
                # Flip direction if negatively aligned
                if cos_sim < 0:
                    omega *= -1
                    cos_sim *= -1  # correct for flip
            else:
                cos_sim = 1.0  # Initial state

            stability_series.append((df.index[t], cos_sim))
            prev_omega = omega.copy()

            # === Compute FSI ===
            f_t = df.iloc[t].values
            fsi_t = np.dot(f_t, omega)
            fsi_series.append((df.index[t], fsi_t))
            omega_history.append(pd.Series(omega, index=columns, name=df.index[t]))

        fsi_series = pd.Series(dict(fsi_series))
        omega_df = pd.DataFrame(omega_history)
        stability_series = pd.Series(dict(stability_series))

        # Flag periods with low similarity
        flagged_dates = stability_series[stability_series < stability_threshold].index.tolist()

        return fsi_series, omega_df, stability_series, flagged_dates
    except Exception as e:
        logging.error(f"Error estimating FSI: {e}", exc_info=True)
        return pd.Series(), pd.DataFrame(), pd.Series(), []


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
