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




def build_dynamic_group_map(df, window_pref=("260","252","250","126","125","63")):
    """
    Build {group -> [present columns]} from actually available engineered features.
    Prefers the longest available rolling window suffix.
    """
    # detect numeric window suffixes that exist in df (e.g., "..._126")
    suffixes = {
        c.rsplit("_", 1)[1] for c in df.columns
        if "_" in c and c.rsplit("_", 1)[1].isdigit()
    }
    suffix = next((s for s in window_pref if s in suffixes), None)
    suf = f"_{suffix}" if suffix else ""

    candidates = {
        "Volatility": [
            f"VIX_dev{suf}", f"MOVE_dev{suf}", f"OVX_dev{suf}", f"VIX3M_dev{suf}",
            "VIX_dev", "MOVE_dev", "OVX_dev", "VIX3M_dev"
        ],
        "Rates": [
            f"10Y_rate_dev{suf}", f"10Y_3M_inversion_dev{suf}",
            "10Y_rate_dev", "10Y_3M_inversion_dev"
        ],
        "Funding": [
            f"3M_TBill_stress{suf}", f"EFFR_stress{suf}",
            "3M_TBill_stress", "EFFR_stress"
        ],
        "Credit": [
            f"IG_OAS_dev{suf}", f"HY_OAS_dev{suf}", f"BBB_OAS_dev{suf}", f"HY_IG_spread{suf}",
            "IG_OAS_dev", "HY_OAS_dev", "BBB_OAS_dev", "HY_IG_spread"
        ],
        "FX/Safe_Haven": [
            f"Gold_dev{suf}", f"USDJPY_dev{suf}", f"USD_stress{suf}",
            "Gold_dev", "USDJPY_dev", "USD_stress"
        ],
    }

    present = set(df.columns)
    group_map = {}
    for g, options in candidates.items():
        cols = [c for c in options if c in present]
        # de-dup while preserving order
        seen, picked = set(), []
        for c in cols:
            if c not in seen:
                seen.add(c); picked.append(c)
        group_map[g] = picked
    return group_map




import logging

def _pick_anchor_columns(df, pref_windows=("250","252","260","126","125","63")):
    """
    Return best-available engineered anchors that are *present* in df.
    Preference: longer windows first.
    """
    base_opts = {
        # volatility
        "VIX_dev": ["VIX_dev"],
        "MOVE_dev": ["MOVE_dev"],
        "OVX_dev": ["OVX_dev"],
        "VIX3M_dev": ["VIX3M_dev"],
        # rates (your current names use *_dev and *_inversion_dev)
        "10Y_rate_dev": ["10Y_rate_dev"],
        "10Y_3M_inversion_dev": ["10Y_3M_inversion_dev"],
        # credit
        "HY_OAS_dev": ["HY_OAS_dev","US HY OAS_dev"],
        "IG_OAS_dev": ["IG_OAS_dev","US IG OAS_dev"],
        "BBB_OAS_dev": ["BBB_OAS_dev","US BBB OAS_dev"],
        # FX / safe haven + funding
        "Gold_dev": ["Gold_dev"],
        "USD_stress": ["USD_stress"],
        "USDJPY_dev": ["USDJPY_dev"],
        "EFFR_stress": ["EFFR_stress"],
        "3M_TBill_stress": ["3M_TBill_stress"],
        "HY_IG_spread": ["HY_IG_spread"]
    }
    # gather available window suffixes
    suffixes = {c.rsplit("_",1)[1] for c in df.columns if "_" in c and c.rsplit("_",1)[1].isdigit()}
    ordered_suffixes = [s for s in pref_windows if s in suffixes]

    present = []
    for s in ordered_suffixes:
        for _, stems in base_opts.items():
            for stem in stems:
                col = f"{stem}_{s}"
                if col in df.columns:
                    present.append(col)
        if present:
            break
    return present




def _make_stress_proxy(df: pd.DataFrame, window:int=126) -> pd.Series:
    """
    Signed stress proxy used ONLY for orientation.
    Higher = more stress. Uses stress-positive engineered features (no abs()).
    Normalizes each feature with a rolling z-score (same window as ALS) and then
    averages; finally smooths a bit with EWMA to reduce jitter.
    """
    # features engineered to be stress-positive in your pipeline
    picks = []
    picks += [c for c in df.columns if ("OAS_dev" in c or "HY_IG_spread" in c)]        # credit
    picks += [c for c in df.columns if c.startswith(("VIX_dev","MOVE_dev","OVX_dev","VIX3M_dev"))]  # vol
    picks += [c for c in df.columns if c.startswith(("USD_stress","USDJPY_dev","Gold_dev"))]        # FX/safe-haven
    picks += [c for c in df.columns if c.startswith(("3M_TBill_stress","EFFR_stress"))]             # funding

    if not picks:
        return pd.Series(0.0, index=df.index)

    X = df[picks].copy()

    # rolling z to match estimator standardization
    mu = X.rolling(window).mean()
    sd = X.rolling(window).std().replace(0, np.nan)
    Z  = (X - mu) / sd

    proxy = Z.mean(axis=1)
    # light EWMA smoothing for stability
    return proxy.ewm(span=21, min_periods=5).mean()




def orient_fsi_and_omega(
    fsi_series: pd.Series,
    omega_history: pd.DataFrame,
    df_engineered: pd.DataFrame,
    stability_series: pd.Series = None,
    stability_threshold: float = 0.7,
    freeze_after_days: int = 60,
    anchor_smooth_days: int = 21,
    corr_window_freeze: int = 126,
    corr_window_flip: int = 60,
    min_corr_to_freeze: float = 0.05,
    allow_flip_cosine_thresh: float = 0.2,
    flip_persist_days: int = 5,
    rho_guard: float = 0.05,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    import numpy as np, pandas as pd, logging

    fsi = fsi_series.copy()
    omega = omega_history.copy()
    idx = fsi.index
    df_e = df_engineered.reindex(idx)

    anchors = _pick_anchor_columns(omega)
    proxy = _make_stress_proxy(df_e).reindex(idx)

    # smoothed anchor sign series
    if anchors:
        anchor_mean = omega[anchors].mean(axis=1)
        anchor_sm = anchor_mean.rolling(anchor_smooth_days, min_periods=max(5, anchor_smooth_days//3)).median()
    else:
        anchor_sm = pd.Series(index=idx, data=np.nan)

    frozen = False
    flip_events = []
    sign = 1.0

    # helper: rolling correlation
    def rolling_corr(a, b, w):
        z = pd.concat([a, b], axis=1).dropna()
        if z.empty:
            return np.nan
        return z.iloc[-w:].corr().iloc[0,1] if len(z) >= 5 else np.nan

    for i, t in enumerate(idx):
        # desired sign from anchors if available, else from proxy corr up to t
        a_val = anchor_sm.loc[t] if t in anchor_sm.index else np.nan
        if pd.isna(a_val):
            # fallback: pick sign to make corr(FSI, proxy) over a short window positive
            w = min(126, i+1)
            r = rolling_corr(fsi.iloc[:i+1], proxy.iloc[:i+1], w)
            desired = 1.0 if (pd.isna(r) or r >= 0) else -1.0
            rationale = "proxy"
        else:
            desired = 1.0 if a_val >= 0 else -1.0
            rationale = "anchors(smoothed)"

        # check if we can freeze (stable + aligned with proxy)
        if not frozen and stability_series is not None and i >= freeze_after_days:
            # stable window
            stable_ok = (stability_series.reindex(idx)
                         .iloc[max(0, i-freeze_after_days+1):i+1]
                         .ge(stability_threshold)).all()
            # proxy alignment window
            r_freeze = rolling_corr(fsi.iloc[:i+1], proxy.iloc[:i+1], corr_window_freeze)
            align_ok = (not pd.isna(r_freeze)) and (r_freeze >= min_corr_to_freeze) and (np.sign(desired) == +1)
            if stable_ok and align_ok:
                frozen = True
                flip_events.append({"date": t, "reason": f"freeze(r={r_freeze:.3f})"})

        flip_now = (np.sign(desired) != np.sign(sign))
        if frozen and flip_now:
            # permissive compelling flip
            cos_ok = (stability_series is not None) and (stability_series.loc[t] < allow_flip_cosine_thresh)
            r_recent = rolling_corr(fsi.iloc[:i+1], proxy.iloc[:i+1], corr_window_flip)
            # require persistence: last K days corr < 0
            persist_mask = []
            for k in range(flip_persist_days):
                r_k = rolling_corr(fsi.iloc[:i+1-k], proxy.iloc[:i+1-k], corr_window_flip)
                persist_mask.append((not pd.isna(r_k)) and (r_k < -0.05))
            persistent_neg = all(persist_mask) if persist_mask else False
            compelling = cos_ok or persistent_neg
            if not compelling:
                desired = sign
                rationale += "|frozen"
            else:
                rationale += "|compelling_flip"

        if np.sign(desired) != np.sign(sign):
            flip_events.append({"date": t, "reason": rationale})

        sign = 1.0 if desired >= 0 else -1.0
        fsi.iloc[i] *= sign
        omega.iloc[i, :] *= sign

    # final safeguard on last-year correlation
    r_guard = rolling_corr(fsi, proxy, 252)
    if not pd.isna(r_guard) and r_guard < -rho_guard:
        fsi *= -1
        omega *= -1
        flip_events.append({"date": idx[-1], "reason": f"posthoc_guard_flip(r252={r_guard:.3f})"})
        logging.warning(f"[ORIENT] Post-hoc guard flip applied (r252={r_guard:.3f}).")

    audit = pd.DataFrame(flip_events)
    if audit.empty:
        logging.info("[ORIENT] No sign flips required across sample.")
    else:
        logging.warning(f"[ORIENT] {len(audit)} sign flip event(s).")
    return fsi, omega, audit






########################################################
########################################################
########################################################








def classify_regime_global_fsi(
    fsi_series: pd.Series,
    quantiles=(0.40, 0.80, 0.96)
) -> pd.Series:
    """
    Base 4-color regime from *global* FSI quantiles (no volatility or rolling window).
    This is the anchor regime: time-invariant thresholds.

    Green:  FSI <= q[0]
    Yellow: q[0] < FSI <= q[1]
    Amber:  q[1] < FSI <= q[2]
    Red:    FSI > q[2]
    """
    fsi = fsi_series.dropna()
    if fsi.empty:
        return pd.Series(index=fsi_series.index, dtype="object")

    q1, q2, q3 = fsi.quantile(quantiles)

    regimes = pd.Series(index=fsi_series.index, dtype="object")

    regimes[fsi <= q1] = "Green"
    regimes[(fsi > q1) & (fsi <= q2)] = "Yellow"
    regimes[(fsi > q2) & (fsi <= q3)] = "Amber"
    regimes[fsi > q3] = "Red"

    # Fill any leading/trailing NaNs conservatively as Yellow
    regimes = regimes.reindex(fsi_series.index).ffill().bfill().fillna("Yellow")
    return regimes




def fsi_vol_spike_flags(
    fsi_series: pd.Series,
    lambda_=0.97,
    change_quantile=0.95,
    level_quantile=0.60,
    min_history: int = 60
) -> pd.Series:
    """
    FSI-only volatility spike detector:
    - Compute EWMA volatility of FSI.
    - Flag spike when:
        * change in vol > change_quantile of its own history, AND
        * vol level > level_quantile of its own distribution.
    """
    vol = ewma_volatility(fsi_series, lambda_=lambda_)  # already FSI-only
    dvol = vol.diff()

    # Restrict to build thresholds only where we have enough history
    valid = dvol.dropna()
    if len(valid) < min_history:
        return pd.Series(False, index=fsi_series.index)

    change_thr = valid.quantile(change_quantile)
    level_thr  = vol.dropna().quantile(level_quantile)

    flags = (dvol > change_thr) & (vol > level_thr)
    return flags.reindex(fsi_series.index).fillna(False)





REGIME_ORDER = ["Green", "Yellow", "Amber", "Red"]
REGIME_TO_INT = {r: i for i, r in enumerate(REGIME_ORDER)}
INT_TO_REGIME = {i: r for i, r in enumerate(REGIME_ORDER)}


def _upgrade_one_notch(regime_series: pd.Series, spike_flags: pd.Series,
                       fsi_series: pd.Series, q1: float) -> pd.Series:
    """
    Upgrade regime by one notch on spike days, but only when FSI > q1
    (i.e., don't create Yellow in very low FSI environment).
    """
    out = regime_series.copy()
    base_int = regime_series.map(REGIME_TO_INT)

    mask = spike_flags & (fsi_series > q1)
    base_int_spike = base_int.where(~mask, np.minimum(base_int + 1, len(REGIME_ORDER) - 1))

    return base_int_spike.map(INT_TO_REGIME)



def classify_regime_fsi_improved(
    fsi_series: pd.Series,
    quantiles=(0.40, 0.80, 0.96),
    lambda_=0.97,
    change_quantile=0.95,
    level_quantile=0.60,
    min_history_spike: int = 60,
    min_run_length: int = 3,
) -> pd.Series:
    """
    Canonical FSI-only 4-color regime classifier.

    Steps:
      1) Base regime from *global* FSI quantiles.
      2) Detect FSI-only volatility spikes (EWMA vol level + change).
      3) On spike days with FSI > q1, upgrade regime by ONE notch (Green→Yellow→Amber→Red).
      4) Smooth regimes: any run shorter than min_run_length days is merged into the previous regime.
    """
    fsi = fsi_series.copy()

    # 1. Base global regime & thresholds
    fsi_nonnull = fsi.dropna()
    if fsi_nonnull.empty:
        return pd.Series(index=fsi.index, dtype="object")
    q1, q2, q3 = fsi_nonnull.quantile(quantiles)

    base_regime = classify_regime_global_fsi(fsi, quantiles=quantiles)

    # 2. Spike flags FSI-only
    spikes = fsi_vol_spike_flags(
        fsi,
        lambda_=lambda_,
        change_quantile=change_quantile,
        level_quantile=level_quantile,
        min_history=min_history_spike,
    )

    # 3. Upgrade by one notch where spike & FSI > q1
    upgraded_regime = _upgrade_one_notch(base_regime, spikes, fsi, q1=q1)

    # 4. Smooth/hysteresis
    smoothed_regime = smooth_regime_series(upgraded_regime, min_run=min_run_length)

    return smoothed_regime




def smooth_regime_series(regimes: pd.Series, min_run: int = 3) -> pd.Series:
    """
    Post-process regime series: any run shorter than min_run is merged into the previous regime.
    This reduces one-day whipsaws without changing the longer regime structure.
    """
    s = regimes.copy().reset_index(drop=True)
    out = s.copy()

    n = len(s)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s.iloc[j + 1] == s.iloc[i]:
            j += 1
        run_len = j - i + 1
        if run_len < min_run and i > 0:
            out.iloc[i:j + 1] = out.iloc[i - 1]
        i = j + 1

    # Put back original index
    out.index = regimes.index
    return out







