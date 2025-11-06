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
from sklearn.model_selection import BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
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

def ewma_volatility(series, lambda_=0.97):
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
    vol_spike_quantile=0.93,   # was 0.90
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


##########################################################
##########################################################

class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Time-series split with optional embargo (in samples) to prevent leakage around split boundaries
    and optional lookahead purge (in samples) for label construction like y_t+N.
    Splits are contiguous and respect time order.
    """
    def __init__(self, n_splits=5, embargo=0, lookahead=0):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo = int(embargo)
        self.lookahead = int(lookahead)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        test_size = n_samples // self.n_splits
        test_starts = [i * test_size for i in range(self.n_splits)]
        test_stops  = test_starts[1:] + [n_samples]

        for start, stop in zip(test_starts, test_stops):
            test_indices = indices[start:stop]

            # Embargo around test to avoid leakage
            left_cut  = max(0, start - self.embargo)
            right_cut = min(n_samples, stop + self.embargo)

            # Purge lookahead on the TRAIN side (labels use t+lookahead)
            purge_right = min(n_samples, start + self.lookahead)

            train_left  = indices[:left_cut]
            train_mid   = indices[right_cut:purge_right]  # will be empty if right_cut >= purge_right
            train_right = indices[purge_right:]

            # Concatenate legal train regions that do not overlap with test or lookahead
            train_indices = np.concatenate([train_left, train_mid, train_right])
            # Remove any overlap with test indices
            train_indices = train_indices[~np.isin(train_indices, test_indices)]

            yield train_indices, test_indices


def predict_regime_probability(
    df, 
    model_type='xgboost', 
    lookahead=20, 
    columns=None,
    xgb_grid=None,
    logit_grid=None,
    n_splits=5,
    scoring='roc_auc',
    use_purged=True,
    embargo=0
):
    """
    Leakage-aware prediction of P(Red in N days).
    - Builds y = 1{Regime_{t+lookahead} == 'Red'}
    - Uses Pipeline(scaler -> model) so scaling is fit within each CV fold
    - Uses TimeSeriesSplit or PurgedTimeSeriesSplit (with lookahead & embargo)
    """
    if 'Regime' not in df.columns:
        raise ValueError("'Regime' column required for regime prediction.")

    df = df.copy()
    df['Future_Red'] = (df['Regime'].shift(-lookahead) == 'Red').astype(int)
    df_logit = df.dropna().copy()

    exclude = ['Future_Red', 'Regime', 'HMM_State']
    if columns is None:
        columns = [c for c in df_logit.columns if c not in exclude]
    X = df_logit[columns].values
    y = df_logit['Future_Red'].values

    # Build model pipeline
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        base = XGBClassifier(
            eval_metric='logloss',
            random_state=42
        )
        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('clf', base)
        ])
        if xgb_grid is None:
            xgb_grid = {
                'clf__n_estimators': [200, 400],
                'clf__max_depth': [3, 5],
                'clf__learning_rate': [0.03, 0.1],
                'clf__subsample': [0.7, 1.0],
                'clf__colsample_bytree': [0.8, 1.0],
            }
        param_grid = xgb_grid
    else:
        from sklearn.linear_model import LogisticRegression
        base = LogisticRegression(max_iter=1200, random_state=42)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', base)
        ])
        if logit_grid is None:
            logit_grid = {
                'clf__C': [0.01, 0.1, 1, 10],
                'clf__penalty': ['l2'],
                'clf__solver': ['lbfgs', 'liblinear']
            }
        param_grid = logit_grid

    # Splitter: purged TSS or vanilla TSS
    if use_purged:
        cv = PurgedTimeSeriesSplit(n_splits=n_splits, embargo=embargo, lookahead=lookahead)
    else:
        cv = TimeSeriesSplit(n_splits=n_splits)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    best_score = search.best_score_

    # In-sample probas (for monitoring, not for headline OOS metrics)
    y_proba_all = best_model.predict_proba(X)[:, 1]
    most_recent_proba = y_proba_all[-1]

    # Feature importance (mapped if possible)
    if model_type == 'xgboost':
        importance = best_model.named_steps['clf'].feature_importances_
    else:
        importance = np.abs(best_model.named_steps['clf'].coef_[0])
    feature_importance = dict(zip(columns, importance))

    return most_recent_proba, y_proba_all, feature_importance, best_model, best_score


###########################################################
##########################################################


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





###################################


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


#########=========================================================================================
#########=========================================================================================


# def predict_regime_probability(
#     df, 
#     model_type='xgboost', 
#     lookahead=20, 
#     columns=None,
#     xgb_grid=None,
#     logit_grid=None,
#     n_splits=5,
#     scoring='roc_auc'
# ):
#     """
#     Predict the probability of being in 'Red' regime in N days using XGBoost or Logistic Regression,
#     with TimeSeriesSplit and hyperparameter optimization.
#     Returns most recent probability, full predicted probability series, variable importance,
#     best estimator, and cross-validated metric.
#     """
#     if 'Regime' not in df.columns:
#         raise ValueError("'Regime' column required for regime prediction.")

#     df = df.copy()
#     df['Future_Red'] = (df['Regime'].shift(-lookahead) == 'Red').astype(int)
#     df_logit = df.dropna()

#     # --- Build feature set as a DataFrame and remember names
#     exclude = ['Future_Red', 'Regime', 'HMM_State']
#     if columns is None:
#         columns = [c for c in df_logit.columns if c not in exclude]
#     X = df_logit[columns]                      # keep as DataFrame
#     y = df_logit['Future_Red']
#     feature_cols = X.columns.tolist()          # <-- remember names

#     # --- Scale but preserve DataFrame structure
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X = pd.DataFrame(X_scaled, columns=feature_cols, index=df_logit.index)  # <-- wrap back to DF

#     # StratifiedKFold for classification
#     tscv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

#     # Hyperparameter grids
#     if model_type == 'xgboost':
#         from xgboost import XGBClassifier
#         if xgb_grid is None:
#             xgb_grid = {
#                 'n_estimators': [100, 200],
#                 'max_depth': [3, 5],
#                 'learning_rate': [0.01, 0.1],
#                 'subsample': [0.7, 1.0],
#             }
#         model = XGBClassifier(eval_metric='logloss', random_state=42)
#         search = GridSearchCV(
#             estimator=model,
#             param_grid=xgb_grid,
#             cv=tscv,
#             scoring=scoring,
#             n_jobs=-1,
#             verbose=0
#         )
#     else:
#         from sklearn.linear_model import LogisticRegression
#         if logit_grid is None:
#             logit_grid = {
#                 'C': [0.01, 0.1, 1, 10],
#                 'penalty': ['l2'],
#                 'solver': ['lbfgs', 'liblinear'],
#                 'max_iter': [700, 1200]
#             }
#         model = LogisticRegression()
#         search = GridSearchCV(
#             estimator=model,
#             param_grid=logit_grid,
#             cv=tscv,
#             scoring=scoring,
#             n_jobs=-1,
#             verbose=0
#         )

#     search.fit(X, y)
#     best_model = search.best_estimator_
#     best_score = search.best_score_

#     # Predict proba for entire data (align with df_logit)
#     y_proba = best_model.predict_proba(X)[:, 1]
#     proba_full = np.full(len(df_logit), np.nan)
#     proba_full[-len(y_proba):] = y_proba

#     # Most recent probability
#     most_recent_proba = y_proba[-1]

#     # Variable importance using preserved names
#     if model_type == 'xgboost':
#         importance = best_model.feature_importances_
#     else:
#         importance = np.abs(best_model.coef_[0])
#     feature_importance = dict(zip(feature_cols, importance))  # <-- use saved names

#     return most_recent_proba, proba_full, feature_importance, best_model, best_score

#########=========================================================================================
#########=========================================================================================

# --- HHI helpers ---

def compute_hhi_ranking(contribs: pd.DataFrame, window: int = 20):
    """
    Compute HHI of variable contributions over the last `window` rows and
    return (hhi, effective_n, ranking_shares).

    - Uses absolute contributions so positive/negative don't cancel.
    - Drops 'FSI' column if present.
    """
    if contribs is None or contribs.empty:
        return np.nan, np.nan, pd.Series(dtype=float)

    recent = contribs.tail(window).copy()
    vars_only = recent.drop(columns=['FSI'], errors='ignore')

    # average absolute contribution per variable
    avg_abs = vars_only.abs().mean()

    # normalize to shares
    total = avg_abs.sum()
    if total == 0 or np.isnan(total):
        return np.nan, np.nan, pd.Series(dtype=float)

    shares = (avg_abs / total).sort_values(ascending=False)

    # HHI and "effective number of contributors"
    hhi = float((shares ** 2).sum())
    effective_n = float(1.0 / hhi) if hhi > 0 else np.nan
    return hhi, effective_n, shares


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













########################################################
########################################################
########################################################




def build_dynamic_group_map(df, window_pref=("250","252","260","126","125","63")):
    """
    Build groups from PRESENT columns, preferring the largest available window suffix.
    Returns a dict group -> [cols].
    """
    # Decide one preferred suffix that actually exists
    suffixes = []
    for c in df.columns:
        parts = c.rsplit("_", 1)
        if len(parts)==2 and parts[1].isdigit():
            suffixes.append(parts[1])
    suffix = next((s for s in window_pref if s in set(suffixes)), None)
    suf = f"_{suffix}" if suffix else ""

    # candidate columns per group (try both with/without suffix; we’ll filter to present)
    candidates = {
        "Volatility": [f"VIX_dev{suf}", f"MOVE_dev{suf}", f"OVX_dev{suf}", f"VIX3M_dev{suf}", f"VIX_VIX3M_spread_dev{suf}",
                       "VIX_dev", "MOVE_dev", "OVX_dev", "VIX3M_dev", "VIX_VIX3M_spread_dev"],
        "Rates": [f"2Y_rate{suf}", f"10Y_rate{suf}", f"10Y_3M_slope_dev{suf}",
                  "2Y_rate", "10Y_rate", "10Y_3M_slope_dev"],
        "Funding": [f"3M_TBill_stress{suf}", f"EFFR_stress{suf}", "3M_TBill_stress", "EFFR_stress"],
        "Credit": [f"IG_OAS_dev{suf}", f"HY_OAS_dev{suf}", f"BBB_OAS_dev{suf}", f"HY_IG_spread{suf}",
                   "IG_OAS_dev", "HY_OAS_dev", "BBB_OAS_dev", "HY_IG_spread"],
        "FX/Safe_Haven": [f"Gold_dev{suf}", f"USDJPY_dev{suf}", f"USD_stress{suf}",
                          "Gold_dev", "USDJPY_dev", "USD_stress"],
    }
    group_map = {}
    present_cols = set(df.columns)
    for g, opts in candidates.items():
        cols = [c for c in opts if c in present_cols]
        if cols:
            # de-dup while preserving order
            seen=set(); clean=[]
            for c in cols:
                if c not in seen:
                    seen.add(c); clean.append(c)
            group_map[g] = clean
        else:
            # leave group empty -> downstream will set to 0 if desired
            group_map[g] = []
    return group_map






import logging

def _pick_anchor_columns(df, pref_windows=("250","252","260","126","125","63")):
    """
    Return best-available engineered anchors in the form actually present.
    Preference: longer windows first.
    """
    # base logical anchors, without window suffix
    base_opts = {
        "VIX_dev": ["VIX_dev"],
        "MOVE_dev": ["MOVE_dev"],
        "HY_OAS_dev": ["HY_OAS_dev","US HY OAS_dev"],
        "IG_OAS_dev": ["IG_OAS_dev","US IG OAS_dev"]
    }
    present = []
    # find suffixes present (e.g., "250")
    suffixes = []
    for c in df.columns:
        parts = c.rsplit("_", 1)
        if len(parts)==2 and parts[1].isdigit():
            suffixes.append(parts[1])
    # prefer longer windows
    ordered_suffixes = [s for s in pref_windows if s in set(suffixes)]
    for s in ordered_suffixes:
        for family, stems in base_opts.items():
            for stem in stems:
                col = f"{stem}_{s}"
                if col in df.columns:
                    present.append(col)
        if present:  # once we collected any with this suffix, use them
            break
    return present

def _make_stress_proxy(df):
    """
    Robust proxy using available credit/vol vars (already engineered, deviation-type).
    Falls back to anything strongly 'stressy' (~OAS, VIX, MOVE).
    Returns a pandas Series aligned to df index.
    """
    # candidates by importance
    priority_groups = [
        [c for c in df.columns if ("HY_OAS_dev" in c) or ("IG_OAS_dev" in c) or ("BBB_OAS_dev" in c)],
        [c for c in df.columns if ("VIX_dev" in c) or ("MOVE_dev" in c) or ("OVX_dev" in c)],
        [c for c in df.columns if ("HY_IG_spread" in c) or ("USD_stress" in c)]
    ]
    for group in priority_groups:
        if group:
            return df[group].mean(axis=1)
    # nothing found -> zero proxy to avoid flips
    return pd.Series(0.0, index=df.index)

def orient_fsi_and_omega(
    fsi_series: pd.Series,
    omega_history: pd.DataFrame,
    df_engineered: pd.DataFrame,
    stability_series: pd.Series = None,
    stability_threshold: float = 0.7,
    freeze_after_days: int = 60,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Enforce 'stress positive' orientation with anchor fallback & freezing.
    - Try anchors at each t; if unavailable/NaN, use correlation with robust proxy.
    - Once stable for 'freeze_after_days' (cosine >= threshold) and orientation agrees with proxy,
      freeze sign (no further flips) unless a *compelling flip* occurs:
      cosine < 0 (direction break) AND corr(FSI, proxy) changes sign contemporaneously.
    Returns: (fsi_oriented, omega_oriented, audit_log_df)
    """
    fsi = fsi_series.copy()
    omega = omega_history.copy()
    idx = fsi.index
    df_e = df_engineered.reindex(idx)

    # anchors (at the suffix we actually have)
    anchors = _pick_anchor_columns(omega)
    proxy = _make_stress_proxy(df_e).reindex(idx)

    frozen = False
    freeze_start = None
    flip_events = []

    # running orientation state (+1 or -1)
    sign = 1.0

    # helper to compute anchor sign at t
    def anchor_sign_t(t):
        if not anchors:
            return np.nan
        vals = omega.loc[t, [a for a in anchors if a in omega.columns]]
        m = np.nanmean(vals.values.astype(float)) if len(vals) else np.nan
        if np.isnan(m):
            return np.nan
        return np.sign(m) if m != 0 else 1.0

    # rolling logic
    for i, t in enumerate(idx):
        # 1) default: anchor sign
        a_sign = anchor_sign_t(t)

        # 2) fallback: correlation with proxy up to t
        if np.isnan(a_sign):
            # windowed correlation to avoid tiny sample issues
            w = min(252, i+1)
            if w < 30:
                p_sign = 1.0
            else:
                corr = pd.concat([fsi.iloc[:i+1], proxy.iloc[:i+1]], axis=1).dropna()
                if len(corr) < 30:
                    p_sign = 1.0
                else:
                    r = corr.corr().iloc[0,1]
                    p_sign = 1.0 if pd.isna(r) else (1.0 if r >= 0 else -1.0)
            desired = p_sign
            rationale = "proxy"
        else:
            desired = a_sign
            rationale = "anchors"

        # 3) freeze logic: after a stable stretch, stop flipping unless 'compelling'
        if not frozen and stability_series is not None:
            # detect stable consecutive run
            stable_mask = (stability_series >= stability_threshold).reindex(idx).fillna(False)
            if i >= freeze_after_days and stable_mask.iloc[max(0, i-freeze_after_days+1):i+1].all():
                # orientation also agrees with proxy? if so, freeze now
                w = min(252, i+1)
                if w >= 60:
                    corr = pd.concat([fsi.iloc[:i+1], proxy.iloc[:i+1]], axis=1).dropna()
                    if len(corr) >= 60:
                        r = corr.corr().iloc[0,1]
                        if not pd.isna(r) and (r >= 0) == (desired >= 0):
                            frozen = True
                            freeze_start = t
                            logging.info(f"[ORIENT] Orientation frozen at {t.date()} (stable {freeze_after_days}d, r={r:.3f}).")

        # 4) apply or block flip
        flip_now = (np.sign(desired) != np.sign(sign))
        if frozen and flip_now:
            # Only allow flip if 'compelling': cosine < 0 AND proxy correlation sign changed using a recent window
            compelling = False
            if stability_series is not None and stability_series.loc[t] < 0:
                w = min(252, i+1)
                pre_w = max(60, w//3)
                corr_recent = pd.concat([fsi.iloc[max(0,i-pre_w):i+1], proxy.iloc[max(0,i-pre_w):i+1]], axis=1).dropna()
                if len(corr_recent) >= 30:
                    r = corr_recent.corr().iloc[0,1]
                    # flip if correlation indicates mis-orientation
                    compelling = (r < 0)
            if not compelling:
                # block the flip
                desired = sign
                rationale += "|frozen"
            else:
                rationale += "|compelling_flip"

        # record event
        if np.sign(desired) != np.sign(sign):
            flip_events.append({"date": t, "reason": rationale})

        sign = 1.0 if desired >= 0 else -1.0
        # apply sign at time t (pointwise multiply)
        fsi.iloc[i] = fsi.iloc[i] * sign
        omega.iloc[i, :] = omega.iloc[i, :] * sign

    # audit
    audit = pd.DataFrame(flip_events)
    if audit.empty:
        logging.info("[ORIENT] No sign flips required across sample.")
    else:
        logging.warning(f"[ORIENT] {len(audit)} sign flip event(s).")
    return fsi, omega, audit


########################################################
########################################################
########################################################