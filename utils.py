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
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

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

def aggregate_contributions_by_group(df, group_map):
    out = pd.DataFrame(index=df.index)
    any_found = False
    for g, cols in group_map.items():
        present = [c for c in cols if c in df.columns]
        if not present:
            logging.warning(f"[GroupMap] No columns found for group '{g}'. Check feature engineering.")
            out[g] = 0.0
        else:
            any_found = True
            out[g] = df[present].sum(axis=1)
    out['FSI'] = out.sum(axis=1)
    if not any_found:
        raise ValueError("No group columns matched. Verify GROUP_MAP vs engineered features.")
    return out

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

def impute_data(df):
    imputer = IterativeImputer(random_state=42, max_iter=10, estimator=BayesianRidge())
    df_imputed = pd.DataFrame(imputer.fit_transform(df), index=df.index, columns=df.columns)
    return df_imputed



###################### LESS SENSITIVE framework ##############################

def adaptive_quantile_thresholds(series, window=500, quantiles=(0.45, 0.80, 0.96)):
    q_green = series.rolling(window, min_periods=window//2).quantile(quantiles[0])
    q_amber = series.rolling(window, min_periods=window//2).quantile(quantiles[1])
    q_red   = series.rolling(window, min_periods=window//2).quantile(quantiles[2])
    return pd.DataFrame({'green': q_green, 'amber': q_amber, 'red': q_red})

def ewma_volatility(series, lambda_=0.96):
    returns = series.pct_change().dropna()
    squared_returns = returns ** 2
    span = (2 / (1 - lambda_)) - 1
    ewma_vol = squared_returns.ewm(span=span, min_periods=30).mean().apply(np.sqrt)
    return ewma_vol.reindex(series.index).ffill()

def volatility_spike_flags(series, vol_window=300, spike_quantile=0.95, lambda_=0.97):
    ewma_vol = ewma_volatility(series, lambda_)
    vol_change = ewma_vol.diff()
    spike_threshold = vol_change.rolling(vol_window).quantile(spike_quantile)
    spike_flags = vol_change > spike_threshold
    return spike_flags.fillna(False)

def classify_risk_regime_hybrid(
    fsi_series,
    vol_window=20,
    vol_spike_quantile=0.92,   # was 0.90
    simplify_to_3=False
):
    try:
        ranks = rankdata(fsi_series)
        percentiles = pd.Series(ranks / len(fsi_series), index=fsi_series.index)

        fsi_vol = fsi_series.rolling(vol_window).std()
        fsi_vol_delta = fsi_vol.diff()
        vol_spike_threshold = fsi_vol_delta.quantile(vol_spike_quantile)
        vol_spike_flags = (fsi_vol_delta > vol_spike_threshold).reindex(fsi_series.index).fillna(False)

        # New breakpoints: 0.40 / 0.80 / 0.96
        p1, p2, p3 = 0.40, 0.80, 0.96

        # Spikes (softer): fewer yellows at low percentiles, push Amber/Red thresholds up
        spike_yellow = (vol_spike_flags) & (percentiles <= p1)                         # <= 0.40
        spike_amber  = (vol_spike_flags) & (percentiles > p1) & (percentiles <= p2)    # (0.40, 0.80]
        spike_red    = (vol_spike_flags) & (percentiles > p2)                          # > 0.80 → Red

        # No spike
        nospike_green  = (~vol_spike_flags) & (percentiles <= p1)
        nospike_yellow = (~vol_spike_flags) & (percentiles > p1) & (percentiles <= p2)
        nospike_amber  = (~vol_spike_flags) & (percentiles > p2) & (percentiles <= p3)
        nospike_red    = (~vol_spike_flags) & (percentiles > p3)

        regimes = pd.Series(index=fsi_series.index, dtype='object')
        regimes[spike_yellow]   = 'Yellow'
        regimes[spike_amber]    = 'Amber'
        regimes[spike_red]      = 'Red'
        regimes[nospike_green]  = 'Green'
        regimes[nospike_yellow] = 'Yellow'
        regimes[nospike_amber]  = 'Amber'
        regimes[nospike_red]    = 'Red'

        regimes = regimes.fillna('Yellow')
        if simplify_to_3:
            regimes = regimes.replace({'Amber': 'Red'})
        return regimes
    except Exception as e:
        logging.error(f"Error classifying risk regime: {e}", exc_info=True)
        return pd.Series()






##############################


# def adaptive_quantile_thresholds(series, window=500, quantiles=(0.40, 0.75, 0.95)):
#     q_green = series.rolling(window, min_periods=window//2).quantile(quantiles[0])
#     q_amber = series.rolling(window, min_periods=window//2).quantile(quantiles[1])
#     q_red   = series.rolling(window, min_periods=window//2).quantile(quantiles[2])
#     thresholds = pd.DataFrame({
#         'green': q_green,
#         'amber': q_amber,
#         'red': q_red
#     })
#     return thresholds

# def ewma_volatility(series, lambda_=0.94):
#     returns = series.pct_change().dropna()
#     squared_returns = returns ** 2
#     span = (2 / (1 - lambda_)) - 1
#     ewma_vol = squared_returns.ewm(span=span, min_periods=30).mean().apply(np.sqrt)
#     return ewma_vol.reindex(series.index).ffill()

# def volatility_spike_flags(series, vol_window=250, spike_quantile=0.9, lambda_=0.94):
#     ewma_vol = ewma_volatility(series, lambda_)
#     vol_change = ewma_vol.diff()
#     spike_threshold = vol_change.rolling(vol_window).quantile(spike_quantile)
#     spike_flags = vol_change > spike_threshold
#     return spike_flags.fillna(False)

def classify_adaptive_regime(fsi_series, quantile_window=500, vol_window=250, spike_quantile=0.9, lambda_=0.94):
    thresholds = adaptive_quantile_thresholds(fsi_series, window=quantile_window)
    spikes = volatility_spike_flags(fsi_series, vol_window=vol_window, spike_quantile=spike_quantile, lambda_=lambda_)
    regimes = pd.Series(index=fsi_series.index, dtype='object')
    for date in fsi_series.index:
        try:
            fsi_value = fsi_series.loc[date]
            green_thr = thresholds.at[date, 'green']
            amber_thr = thresholds.at[date, 'amber']
            red_thr = thresholds.at[date, 'red']
            if spikes.at[date]:
                if fsi_value <= green_thr:
                    regimes.at[date] = 'Yellow'
                elif fsi_value <= amber_thr:
                    regimes.at[date] = 'Amber'
                else:
                    regimes.at[date] = 'Red'
            else:
                if fsi_value <= green_thr:
                    regimes.at[date] = 'Green'
                elif fsi_value <= amber_thr:
                    regimes.at[date] = 'Yellow'
                elif fsi_value <= red_thr:
                    regimes.at[date] = 'Amber'
                else:
                    regimes.at[date] = 'Red'
        except Exception as e:
            regimes.at[date] = 'Green'
    return regimes.reindex(fsi_series.index).ffill().bfill()

##################################


def classify_adaptive_regime_hybrid_fallback(
    fsi_series, 
    quantile_window=500, 
    vol_window=250, 
    spike_quantile=0.9, 
    lambda_=0.94
):
    # 1. Compute rolling quantile thresholds as usual
    thresholds = adaptive_quantile_thresholds(fsi_series, window=quantile_window)
    spikes = volatility_spike_flags(fsi_series, vol_window=vol_window, spike_quantile=spike_quantile, lambda_=lambda_)
    regimes = pd.Series(index=fsi_series.index, dtype='object')

    # 2. Determine cutoff where quantile window is populated (first non-NaN green)
    valid_quantile_mask = ~thresholds['green'].isna()
    first_valid_idx = valid_quantile_mask.idxmax()  # first index where green is not nan
    if isinstance(first_valid_idx, bool):  # If all values are False, idxmax returns False
        first_valid_idx = None

    # 3. For initial region, use hybrid (static quantile) approach
    if first_valid_idx is not None:
        # Hybrid regime for first_valid_idx
        pre_regime = classify_risk_regime_hybrid(
            fsi_series.loc[:first_valid_idx],
            vol_window=20,  # Use your chosen value
            vol_spike_quantile=spike_quantile
        )
        regimes.loc[:first_valid_idx] = pre_regime

        # Rolling regime for rest
        for date in fsi_series.index[fsi_series.index.get_loc(first_valid_idx):]:
            try:
                fsi_value = fsi_series.loc[date]
                green_thr = thresholds.at[date, 'green']
                amber_thr = thresholds.at[date, 'amber']
                red_thr = thresholds.at[date, 'red']
                if pd.isna(green_thr) or pd.isna(amber_thr) or pd.isna(red_thr):
                    regimes.at[date] = 'Yellow'
                    continue
                if spikes.at[date]:
                    if fsi_value <= green_thr:
                        regimes.at[date] = 'Yellow'
                    elif fsi_value <= amber_thr:
                        regimes.at[date] = 'Amber'
                    else:
                        regimes.at[date] = 'Red'
                else:
                    if fsi_value <= green_thr:
                        regimes.at[date] = 'Green'
                    elif fsi_value <= amber_thr:
                        regimes.at[date] = 'Yellow'
                    elif fsi_value <= red_thr:
                        regimes.at[date] = 'Amber'
                    else:
                        regimes.at[date] = 'Red'
            except Exception as e:
                regimes.at[date] = 'Yellow'
    else:
        # If for some reason thresholds are always nan, fallback entirely to hybrid
        regimes = classify_risk_regime_hybrid(
            fsi_series,
            vol_window=20,
            vol_spike_quantile=spike_quantile
        )

    regimes = regimes.ffill().bfill()
    return regimes



##################################








# def classify_risk_regime_hybrid(fsi_series, vol_window=20, vol_spike_quantile=0.9, simplify_to_3=False):
#     """
#     Hybrid regime classification combining quantile levels and volatility spikes.

#     Parameters:
#         fsi_series (pd.Series): FSI index series.
#         vol_window (int): Rolling window for FSI volatility.
#         vol_spike_quantile (float): Threshold for volatility change to qualify as a spike.
#         simplify_to_3 (bool): If True, collapse Amber and Red into a single 'Red'.

#     Returns:
#         pd.Series: Regime labels (Green, Yellow, Amber, Red)
#     """
#     try:
#         # Compute ECDF percentiles
#         ranks = rankdata(fsi_series)
#         percentiles = pd.Series(ranks / len(fsi_series), index=fsi_series.index)

#         # Volatility change
#         fsi_vol = fsi_series.rolling(vol_window).std()
#         fsi_vol_delta = fsi_vol.diff()
#         vol_spike_threshold = fsi_vol_delta.quantile(vol_spike_quantile)
#         vol_spike_flags = (fsi_vol_delta > vol_spike_threshold).reindex(fsi_series.index).fillna(False)

#         # Vectorized regime classification logic
#         # For volatility spikes
#         spike_yellow = (vol_spike_flags) & (percentiles <= 0.35)
#         spike_amber  = (vol_spike_flags) & (percentiles > 0.35) & (percentiles <= 0.75)
#         spike_red    = (vol_spike_flags) & (percentiles > 0.75)  # All > 0.75 go to Red (including > 0.95 as in original)

#         # For no spike
#         nospike_green = (~vol_spike_flags) & (percentiles <= 0.35)
#         nospike_yellow = (~vol_spike_flags) & (percentiles > 0.35) & (percentiles <= 0.75)
#         nospike_amber  = (~vol_spike_flags) & (percentiles > 0.75) & (percentiles <= 0.95)
#         nospike_red    = (~vol_spike_flags) & (percentiles > 0.95)

#         regimes = pd.Series(index=fsi_series.index, dtype='object')
#         regimes[spike_yellow] = 'Yellow'
#         regimes[spike_amber] = 'Amber'
#         regimes[spike_red] = 'Red'
#         regimes[nospike_green] = 'Green'
#         regimes[nospike_yellow] = 'Yellow'
#         regimes[nospike_amber] = 'Amber'
#         regimes[nospike_red] = 'Red'

#         # Fill any missing values (e.g. due to NaN at start) with 'Green'
#         regimes = regimes.fillna('Yellow')

#         if simplify_to_3:
#             regimes = regimes.replace({'Amber': 'Red'})

#         return regimes
#     except Exception as e:
#         logging.error(f"Error classifying risk regime: {e}", exc_info=True)
#         return pd.Series()


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



def predict_regime_probability(
    df, 
    model_type='xgboost', 
    lookahead=20, 
    columns=None,
    xgb_grid=None,
    logit_grid=None,
    n_splits=5,
    scoring='roc_auc'
):
    """
    Predict the probability of being in 'Red' regime in N days using XGBoost or Logistic Regression,
    with TimeSeriesSplit and hyperparameter optimization.
    Returns most recent probability, full predicted probability series, variable importance,
    best estimator, and cross-validated metric.
    """
    if 'Regime' not in df.columns:
        raise ValueError("'Regime' column required for regime prediction.")

    df = df.copy()
    df['Future_Red'] = (df['Regime'].shift(-lookahead) == 'Red').astype(int)
    df_logit = df.dropna()

    # --- Build feature set as a DataFrame and remember names
    exclude = ['Future_Red', 'Regime', 'HMM_State']
    if columns is None:
        columns = [c for c in df_logit.columns if c not in exclude]
    X = df_logit[columns]                      # keep as DataFrame
    y = df_logit['Future_Red']
    feature_cols = X.columns.tolist()          # <-- remember names

    # --- Scale but preserve DataFrame structure
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_cols, index=df_logit.index)  # <-- wrap back to DF

    # StratifiedKFold for classification
    tscv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Hyperparameter grids
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        if xgb_grid is None:
            xgb_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.7, 1.0],
            }
        model = XGBClassifier(eval_metric='logloss', random_state=42)
        search = GridSearchCV(
            estimator=model,
            param_grid=xgb_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
    else:
        from sklearn.linear_model import LogisticRegression
        if logit_grid is None:
            logit_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [700, 1200]
            }
        model = LogisticRegression()
        search = GridSearchCV(
            estimator=model,
            param_grid=logit_grid,
            cv=tscv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )

    search.fit(X, y)
    best_model = search.best_estimator_
    best_score = search.best_score_

    # Predict proba for entire data (align with df_logit)
    y_proba = best_model.predict_proba(X)[:, 1]
    proba_full = np.full(len(df_logit), np.nan)
    proba_full[-len(y_proba):] = y_proba

    # Most recent probability
    most_recent_proba = y_proba[-1]

    # Variable importance using preserved names
    if model_type == 'xgboost':
        importance = best_model.feature_importances_
    else:
        importance = np.abs(best_model.coef_[0])
    feature_importance = dict(zip(feature_cols, importance))  # <-- use saved names

    return most_recent_proba, proba_full, feature_importance, best_model, best_score



def compute_transition_matrix(series):
    """
    Compute the normalized historical transition matrix (from regime/state series).
    Returns a pandas DataFrame (rows: FROM, cols: TO, values: probability).
    """
    series = pd.Series(series).astype(str).reset_index(drop=True)
    from_states = series[:-1].values
    to_states = series[1:].values
    matrix = pd.crosstab(from_states, to_states, normalize='index')
    print("Transition matrix after fix:\n", matrix)
    return matrix


def average_time_in_regime(regime_series):
    """
    Compute average consecutive time spent in each regime.
    Returns a pandas Series: index=regime, value=average streak length (in days).
    """
    import pandas as pd
    s = pd.Series(regime_series).reset_index(drop=True)
    result = {}
    i = 0
    n = len(s)
    while i < n:
        current = s[i]
        start = i
        while i + 1 < n and s[i+1] == current:
            i += 1
        end = i
        length = end - start + 1
        result.setdefault(current, []).append(length)
        i += 1
    # Average streak for each regime
    avg = {k: sum(v)/len(v) for k, v in result.items()}
    return pd.Series(avg).sort_index()
