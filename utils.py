# utils.py
import numpy as np
import pandas as pd
import logging
from scipy.stats import rankdata
from pykalman import KalmanFilter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


def normalize_loadings(weights):
    """Normalize the loadings vector."""
    try:
        return weights / np.linalg.norm(weights)
    except Exception as e:
        logging.error(f"Error normalizing loadings: {e}", exc_info=True)
        return weights


def moving_average_deviation(series, window, invert=False):
    """Calculate the deviation from the moving average."""
    try:
        ma = series.rolling(window).mean()
        dev = (series - ma) / ma
        return -dev if invert else dev
    except Exception as e:
        logging.error(f"Error calculating moving average deviation: {e}", exc_info=True)
        return pd.Series()


def absolute_deviation_rotated(series, window):
    """Calculate the rotated absolute deviation from the moving average."""
    try:
        ma = series.rolling(window).mean()
        return ma - series
    except Exception as e:
        logging.error(f"Error calculating absolute deviation rotated: {e}", exc_info=True)
        return pd.Series()


def absolute_deviation(series, window, invert=False):
    """Calculate the absolute deviation from the moving average."""
    try:
        ma = series.rolling(window).mean()
        dev = series - ma
        return -dev if invert else dev
    except Exception as e:
        logging.error(f"Error calculating absolute deviation: {e}", exc_info=True)
        return pd.Series()


def aggregate_contributions_by_group(contribs, group_map):
    """Aggregate variable contributions by group."""
    try:
        grouped = pd.DataFrame(index=contribs.index)
        for group, patterns in group_map.items():
            cols = [col for col in contribs.columns if any(p in col for p in patterns)]
            grouped[group] = contribs[cols].sum(axis=1)
        grouped['FSI'] = contribs['FSI']
        return grouped
    except Exception as e:
        logging.error(f"Error aggregating contributions by group: {e}", exc_info=True)
        return pd.DataFrame()

def kalman_impute(series):
    """Impute missing values in a series using Kalman filtering."""
    try:
        if series.isnull().sum() == 0:
            return series

        # Replace inf with NaN, then fill small gaps so Kalman can run
        series = series.replace([np.inf, -np.inf], np.nan)
        
        # Fill leading/trailing NaNs temporarily for Kalman smoothing to work
        filled = series.copy()
        filled = filled.ffill().bfill()

        # Kalman filter will now work on filled series
        kf = KalmanFilter(initial_state_mean=filled.mean(), n_dim_obs=1)
        state_means, _ = kf.em(filled.values, n_iter=10).smooth(filled.values)

        # Replace original NaNs with smoothed values; keep original values otherwise
        smoothed = pd.Series(state_means.flatten(), index=series.index)
        return series.combine_first(smoothed)
    except Exception as e:
        logging.error(f"Error imputing data using Kalman filter: {e}", exc_info=True)
        return series

def smart_impute(series):
    """Impute missing values in a series using a hybrid approach."""
    try:
        if series.isna().sum() == 0:
            return series

        # Defensive cleanup
        series = series.replace([np.inf, -np.inf], np.nan)

        missing_ratio = series.isna().mean()

        if series.isna().sum() < 5:
            return series.ffill().bfill()

        elif missing_ratio < 0.1:
            return series.fillna(series.rolling(window=15, min_periods=1).mean())

        elif missing_ratio < 0.3:
            return kalman_impute(series)

        else:
            logging.warning(f"Dropping or neutralizing highly missing series: {series.name}")
            return pd.Series(index=series.index, data=np.nan)
    except Exception as e:
        logging.error(f"Error imputing data: {e}", exc_info=True)
        return pd.Series(index=series.index, data=np.nan)

def classify_risk_regime_hybrid(fsi_series, vol_window=20, vol_spike_quantile=0.9, simplify_to_3=False):
    """
    Hybrid regime classification combining quantile levels and volatility spikes.

    Parameters:
        fsi_series (pd.Series): FSI index series.
        vol_window (int): Rolling window for FSI volatility.
        vol_spike_quantile (float): Threshold for volatility change to qualify as a spike.
        simplify_to_3 (bool): If True, collapse Amber and Red into a single 'Red'.

    Returns:
        pd.Series: Regime labels (Green, Yellow, Amber, Red)
    """
    try:
        # Compute ECDF percentiles
        ranks = rankdata(fsi_series)
        percentiles = pd.Series(ranks / len(fsi_series), index=fsi_series.index)

        # Volatility change
        fsi_vol = fsi_series.rolling(vol_window).std()
        fsi_vol_delta = fsi_vol.diff()
        vol_spike_threshold = fsi_vol_delta.quantile(vol_spike_quantile)
        vol_spike_flags = (fsi_vol_delta > vol_spike_threshold).reindex(fsi_series.index).fillna(False)

        # Regime classification logic
        def hybrid_classify(fsi_val, pct, vol_spike):
            if vol_spike:
                if pct <= 0.35: return 'Yellow'
                elif pct <= 0.75: return 'Amber'
                elif pct <= 0.95: return 'Red'
                else: return 'Red'
            else:
                if pct <= 0.35: return 'Green'
                elif pct <= 0.75: return 'Yellow'
                elif pct <= 0.95: return 'Amber'
                else: return 'Red'

        regimes = pd.Series(index=fsi_series.index, dtype='object')
        for date in fsi_series.index:
            regimes[date] = hybrid_classify(fsi_series[date], percentiles[date], vol_spike_flags[date])

        if simplify_to_3:
            regimes = regimes.replace({'Amber': 'Red'})

        return regimes
    except Exception as e:
        logging.error(f"Error classifying risk regime: {e}", exc_info=True)
        return pd.Series()

def smooth_transition_regime(fsi_series, gamma=2.5, c=0.5):
    """Calculate smooth transition weights for regime classification."""
    try:
        transition_weight = 1 / (1 + np.exp(-gamma * (fsi_series - c)))
        return pd.Series(transition_weight, index=fsi_series.index)
    except Exception as e:
        logging.error(f"Error calculating smooth transition regime: {e}", exc_info=True)
        return pd.Series()


def regime_from_smooth_weight(weight_series, quantiles=(0.33, 0.66, 0.90)):
    """Map smooth transition weights to regimes using quantile-based thresholds."""
    try:
        q1, q2, q3 = weight_series.quantile(quantiles)

        def map_regime(w):
            if w < q1:
                return 'Green'
            elif w < q2:
                return 'Yellow'
            elif w < q3:
                return 'Amber'
            return 'Red'

        return weight_series.apply(map_regime)
    except Exception as e:
        logging.error(f"Error mapping regime from smooth weight: {e}", exc_info=True)
        return pd.Series()


def get_current_regime(df):
    """Return the most recent regime label from rule-based regime column."""
    if 'Regime' in df.columns:
        return df['Regime'].iloc[-1]
    else:
        raise ValueError("Regime column not found in dataframe.")

def run_hmm(df, n_states=4, columns=None):
    """
    Fit an HMM on the specified columns and return most recent state and all states.
    """
    if columns is None:
        # Use all columns except known labels
        columns = [c for c in df.columns if c not in ['Regime', 'HMM_State', 'Future_Red']]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[columns].dropna())
    
    hmm = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=1000, random_state=42)
    hmm.fit(df_scaled)
    hidden_states = hmm.predict(df_scaled)
    state_probs = hmm.predict_proba(df_scaled)

    hmm_states_full = np.full(len(df), np.nan)
    hmm_states_full[-len(hidden_states):] = hidden_states

    df_result = df.copy()
    df_result['HMM_State'] = hmm_states_full

    most_recent_state = int(hidden_states[-1])
    return most_recent_state, df_result, state_probs

def predict_regime_probability(df, model_type='xgboost', lookahead=20, columns=None):
    """
    Predict the probability of being in 'Red' regime in N days using XGBoost or Logistic Regression.
    Returns most recent probability, full predicted probability series, and variable importance.
    """
    # Prepare target
    if 'Regime' not in df.columns:
        raise ValueError("'Regime' column required for regime prediction.")

    df = df.copy()
    df['Future_Red'] = (df['Regime'].shift(-lookahead) == 'Red').astype(int)
    df_logit = df.dropna()

    # Feature columns
    exclude = ['Future_Red', 'Regime', 'HMM_State']
    if columns is None:
        columns = [c for c in df_logit.columns if c not in exclude]
    X = df_logit[columns]
    y = df_logit['Future_Red']

    # Split (no shuffle: time series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.3)

    if model_type == 'xgboost':
        model = XGBClassifier(n_estimators=200, eval_metric='logloss', use_label_encoder=False, random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        importance = model.feature_importances_
    else:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        importance = np.abs(model.coef_[0])

    # Full probability array (align to df)
    proba_full = np.full(len(df_logit), np.nan)
    proba_full[-len(y_proba):] = y_proba

    # Most recent predicted probability
    most_recent_proba = y_proba[-1]
    feature_importance = dict(zip(X_train.columns, importance))

    # Optional: print classification report (can be commented out)
    # print(classification_report(y_test, model.predict(X_test)))

    return most_recent_proba, proba_full, feature_importance

def compute_transition_matrix(series):
    """
    Compute the normalized historical transition matrix (from regime/state series).
    Returns a pandas DataFrame (rows: FROM, cols: TO, values: probability).
    """
    from_states = series[:-1]
    to_states = series[1:]
    matrix = pd.crosstab(from_states, to_states, normalize='index')
    return matrix
