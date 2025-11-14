# main.py

import logging
import pandas as pd
import configparser
import numpy as np
import os
import time
from data_fetching import get_all_series
from fsi_estimation import (compute_variable_contributions, 
                            compute_timevarying_contributions, 
                            estimate_fsi_expanding_with_als)
from plotting import (
    plot_group_contributions_with_regime, plot_grouped_contributions,
    plot_pnl_with_regime_ribbons, save_fsi_charts_to_html
)
from utils import (
    aggregate_contributions_by_group, smooth_transition_regime, regime_from_smooth_weight, orient_fsi_and_omega,
    moving_average_deviation, absolute_deviation_rotated, absolute_deviation, build_dynamic_group_map,
    classify_risk_regime_hybrid, kalman_impute, impute_data, classify_adaptive_regime
)


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_configuration(config_file='config.ini'):
    """Load configuration from a .ini file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def merge_data(config, max_age_hours=0):
    """
    Loads processed data from cache if recent, otherwise runs full pipeline and caches result.
    """
    # ----- USE A PORTABLE CACHE DIRECTORY -----
    base_path = "./cache-directory"
    os.makedirs(base_path, exist_ok=True)
    cache_path = os.path.join(base_path, "fsi_data_latest.parquet")

    try:
        # --- 0. Use cache if recent ---
        if os.path.exists(cache_path):
            file_age = (time.time() - os.path.getmtime(cache_path)) / 3600.0
            if file_age < max_age_hours:
                logging.info(f"[merge_data] Loading cached data ({cache_path}), age: {file_age:.2f} hours")
                return pd.read_parquet(cache_path)
            else:
                logging.info(f"[merge_data] Cache too old ({file_age:.1f}h > {max_age_hours}h), refetching.")

        # --- 1. Fetch initial raw data ---
        df = get_all_series(config)
        if df.empty:
            logging.error("Merged data is empty right after fetching. Exiting.")
            return None

        # ---- SAVE CSVs TO LOCAL CACHE DIR (portable) ----
        df.to_csv(os.path.join(base_path, "Full_set_variables_brut.csv"))

        logging.info(f"Initial columns: {list(df.columns)}")
        logging.info(f"Initial shape: {df.shape}")

        # --- 2. Align to the latest first-valid date ---
        first_valid_dates = df.apply(lambda col: col.first_valid_index())
        cutoff_date = max(first_valid_dates)
        logging.info(f"Latest first-valid date across columns: {cutoff_date}")
        df = df[df.index >= cutoff_date]
        logging.info(f"Shape after aligning to cutoff date: {df.shape}")

        # --- 3. Forward/backward fill, drop columns with too much missing ---
        df = df.ffill().bfill()
        cols_before_drop = set(df.columns)
        df = df.dropna(axis=1, thresh=int(0.9 * len(df)))
        cols_after_drop = set(df.columns)
        dropped_cols = cols_before_drop - cols_after_drop
        logging.info(f"Dropped {dropped_cols} columns with >10% NaN. Shape now: {df.shape}")

        # # --- 3. Hybrid imputation for missing values ---
        # for col in df.columns:
        #     missing_ratio = df[col].isnull().mean()
        #     if missing_ratio < 0.05:
        #         df[col] = df[col].ffill().bfill()
        #     elif missing_ratio < 0.15:
        #         df[col] = kalman_impute(df[col])
        #     elif missing_ratio < 0.3:
        #         df[col] = impute_data(df[[col]])[col]
        #     else:
        #         df.drop(col, axis=1, inplace=True)
        #         logging.warning(f"Dropped column {col} due to excessive missing values ({missing_ratio:.1%}).")
        # logging.info(f"Shape after hybrid imputation: {df.shape}")

        # # Defensive: Drop any still-missing columns (should be none, but for safety)
        # df = df.dropna(axis=1, thresh=int(0.9 * len(df)))
        # df = df.dropna()  # Drop any remaining NaN rows (should be rare)
        # logging.info(f"Shape after dropping remaining NaN: {df.shape}")

        # --- 4. Final drop of any remaining NaN rows ---
        df = df.dropna()
        logging.info(f"Shape after dropping remaining NaN rows: {df.shape}")

        # --- 5. Defensive check for empty or too-narrow dataframe ---
        if df.empty or df.shape[1] < 2:
            logging.error(f"Final cleaned dataset is empty or too narrow for SVD. Columns: {df.columns.tolist()}")
            return None

        # === 6. Feature Engineering ===
        windows = [int(w) for w in config['fsi']['windows'].split(',')]




        for window in windows:
            # --- Volatility (stress-positive already) ---
            if 'VIX' in df.columns:
                df[f'VIX_dev_{window}'] = moving_average_deviation(df['VIX'], window)
            if 'MOVE Index' in df.columns:
                df[f'MOVE_dev_{window}'] = moving_average_deviation(df['MOVE Index'], window)
            if 'OVX' in df.columns:
                df[f'OVX_dev_{window}'] = moving_average_deviation(df['OVX'], window)
            if 'VIX3M' in df.columns:
                df[f'VIX3M_dev_{window}'] = moving_average_deviation(df['VIX3M'], window)
                # if 'VIX' in df.columns:
                #     # backwardation (VIX > VIX3M) ⇒ more stress
                #     spread = (df['VIX'] - df['VIX3M']).rename('VIX_minus_VIX3M')
                #     df[f'VIX_VIX3M_spread_dev_{window}'] = moving_average_deviation(spread, window)

            # --- FX / Safe Haven ---
            if 'Gold Price' in df.columns:
                df[f'Gold_dev_{window}'] = moving_average_deviation(df['Gold Price'], window)   # ↑gold ⇒ ↑stress
            if 'USD Index (DXY)' in df.columns:
                # risk-off USD bid ⇒ ↑ DXY ⇒ ↑stress  (remove invert)
                df[f'USD_stress_{window}'] = moving_average_deviation(df['USD Index (DXY)'], window)
            if 'USDJPY' in df.columns:
                # risk-off JPY strength ⇒ USDJPY ↓ ⇒ make stress-positive
                dev = moving_average_deviation(df['USDJPY'], window)
                df[f'USDJPY_dev_{window}'] = -dev   # invert here only

            # --- Rates (redefined to be stress-positive) ---
            # 1) Big absolute moves in yields are stress
            if '10Y Yield' in df.columns:
                ma10 = df['10Y Yield'].rolling(window).mean()
                df[f'10Y_rate_dev_{window}'] = (df['10Y Yield'] - ma10).abs()
            # if '2Y Yield' in df.columns:
            #     ma2 = df['2Y Yield'].rolling(window).mean()
            #     df[f'2Y_rate_stress_{window}'] = (df['2Y Yield'] - ma2).abs()

            # 2) Front-end jumps (funding pressures): only upside contributes
            if '3M T-Bill' in df.columns:
                ma3m = df['3M T-Bill'].rolling(window).mean()
                dev_up = (df['3M T-Bill'] - ma3m).clip(lower=0)
                df[f'3M_TBill_stress_{window}'] = dev_up

            # 3) Curve inversion stress: more negative 10Y-3M ⇒ higher stress
            if '10Y-3M Slope' in df.columns:
                ma_slope = df['10Y-3M Slope'].rolling(window).mean()
                slope_dev = df['10Y-3M Slope'] - ma_slope
                df[f'10Y_3M_inversion_dev_{window}'] = (-slope_dev).clip(lower=0)

            # --- Funding (policy/overnight) ---
            if 'EFFR' in df.columns:
                ma_effr = df['EFFR'].rolling(window).mean()
                df[f'EFFR_stress_{window}'] = (df['EFFR'] - ma_effr).clip(lower=0)

            # --- Credit / OAS (already stress-positive) ---
            if 'US IG OAS' in df.columns:
                df[f'IG_OAS_dev_{window}']  = absolute_deviation(df['US IG OAS'], window)
            if 'US HY OAS' in df.columns:
                df[f'HY_OAS_dev_{window}']  = absolute_deviation(df['US HY OAS'], window)
            if 'US BBB OAS' in df.columns:
                df[f'BBB_OAS_dev_{window}'] = absolute_deviation(df['US BBB OAS'], window)
            if 'HYG-LQD Spread' in df.columns:
                df[f'HY_IG_spread_{window}'] = moving_average_deviation(df['HYG-LQD Spread'], window)





        # for window in windows:
        #     # --- Volatility ---
        #     if 'VIX' in df.columns:
        #         df[f'VIX_dev_{window}'] = moving_average_deviation(df['VIX'], window)
        #     if 'MOVE Index' in df.columns:
        #         df[f'MOVE_dev_{window}'] = moving_average_deviation(df['MOVE Index'], window)
        #     if 'OVX' in df.columns:
        #         df[f'OVX_dev_{window}'] = moving_average_deviation(df['OVX'], window)
        #     if 'VIX3M' in df.columns:
        #         df[f'VIX3M_dev_{window}'] = moving_average_deviation(df['VIX3M'], window)
        #         if 'VIX' in df.columns:
        #             # optional term-structure measure
        #             spread = (df['VIX'] - df['VIX3M']).rename('VIX_minus_VIX3M')
        #             df[f'VIX_VIX3M_spread_dev_{window}'] = moving_average_deviation(spread, window)

        #     # --- Safe-Haven / FX ---
        #     if 'Gold Price' in df.columns:
        #         df[f'Gold_dev_{window}'] = moving_average_deviation(df['Gold Price'], window)
        #     if 'USD Index (DXY)' in df.columns:
        #         df[f'USD_stress_{window}'] = moving_average_deviation(df['USD Index (DXY)'], window, invert=True)
        #     if 'USDJPY' in df.columns:
        #         df[f'USDJPY_dev_{window}'] = moving_average_deviation(df['USDJPY'], window, invert=True)

        #     # --- Rates ---
        #     if '10Y Yield' in df.columns:
        #         df[f'10Y_rate_{window}'] = absolute_deviation(df['10Y Yield'], window, invert=True)
        #     if '2Y Yield' in df.columns:
        #         df[f'2Y_rate_{window}'] = absolute_deviation(df['2Y Yield'], window, invert=True)
        #     if '3M T-Bill' in df.columns:
        #         df[f'3M_TBill_stress_{window}'] = absolute_deviation_rotated(df['3M T-Bill'], window)
        #     if '10Y-3M Slope' in df.columns:
        #         df[f'10Y_3M_slope_dev_{window}'] = absolute_deviation(df['10Y-3M Slope'], window, invert=True)

        #     # --- Funding & Liquidity ---
        #     if 'EFFR' in df.columns:
        #         df[f'EFFR_stress_{window}'] = absolute_deviation(df['EFFR'], window)
        #     # if 'EFFR_VOLUME' in df.columns:
        #     #     df[f'EFFR_VOLUME_{window}'] = absolute_deviation(df['EFFR_VOLUME'], window)

        #     # --- Credit/OAS ---
        #     if 'US IG OAS' in df.columns:
        #         df[f'IG_OAS_dev_{window}'] = absolute_deviation(df['US IG OAS'], window)
        #     if 'US HY OAS' in df.columns:
        #         df[f'HY_OAS_dev_{window}'] = absolute_deviation(df['US HY OAS'], window)
        #     if 'US BBB OAS' in df.columns:
        #         df[f'BBB_OAS_dev_{window}'] = absolute_deviation(df['US BBB OAS'], window)
        #     if 'HYG-LQD Spread' in df.columns:
        #         df[f'HY_IG_spread_{window}'] = moving_average_deviation(df['HYG-LQD Spread'], window)




        # --- 7. Drop raw columns to keep only engineered features ---
        raw_cols = [
            'VIX', 'MOVE Index', 'USD Index (DXY)', 'Gold Price',
            '10Y Yield', 'USD Overnight Rate', 'FRED RRP Volume', '3M T-Bill', 'US BBB OAS', '2Y Yield',
            'US IG OAS', 'US HY OAS', '1Y Treasury Yield', '2Y Treasury Yield', '2Y Yield fmp', '10Y Yield',
            'SPY P/E', '10Y-2Y Slope', '10Y-3M Slope', 'VIX3M', 'VIX-VIX3M Spread', 'federalFunds', '10Y Yield fmp',
            'HYG-LQD Spread', 'SPY P/B', 'OVX', 'VXV', 'VIX-VXV Spread', 'USDJPY', "EFFR", "OBFR Rate",
            "3M T-Bill", "10Y Yield", "2Y Yield", "USD Index", "FRED RRP", "US Corp OAS", "EFFR_VOLUME"
        ]
        df.drop(raw_cols, axis=1, inplace=True, errors='ignore')

        logging.info(f"Shape after dropping raw columns: {df.shape}")
        if df.empty or df.shape[1] < 2:
            logging.error(f"Final engineered dataset is empty or too narrow for SVD. Columns: {df.columns.tolist()}")
            return None

        # --- 8. Final NaN drop (should be minimal) ---
        df = df.dropna()
        logging.info(f"Shape after final dropna: {df.shape}")

        if df.empty or df.shape[1] < 2:
            logging.error(f"Final processed dataset is empty or too narrow for SVD. Columns: {df.columns.tolist()}")
            return None

        # --- 9. Save as CSV and cache as Parquet ---
        df.to_csv(os.path.join(base_path, "Full_set_variables_std.csv"))
        df.to_parquet(cache_path)
        logging.info(f"Final merged and processed dataset saved and cached at {cache_path}.")

        return df

    except Exception as e:
        logging.error(f"Error merging data: {e}", exc_info=True)
        return None




def main():
    config = load_configuration()
    df = merge_data(config)
    if df is None:
        logging.error("Failed to merge data. Exiting.")
        return

    # fsi_series, omega_history, cos_sim_series, unstable_dates = estimate_fsi_recursive_rolling_with_stability(
    #     df,
    #     window_size=int(config['fsi']['window_size']),
    #     n_iter=int(config['fsi']['n_iter']),
    #     stability_threshold=float(config['fsi']['stability_threshold'])
    # )


    min_history = int(config['fsi']['window_size'])

    fsi_series, omega_history, cos_sim_series, unstable_dates = estimate_fsi_expanding_with_als(
        df,
        min_history=min_history,
        n_iter=int(config['fsi']['n_iter']),
        stability_threshold=float(config['fsi']['stability_threshold'])
    )

    # freeze_after_days = 90
    # stability_threshold = 0.8
    # min_corr_to_freeze = 0.10
    # allow_flip_cosine_thresh = 0.2

    fsi_series, omega_history, orient_audit = orient_fsi_and_omega(
        fsi_series=fsi_series,
        omega_history=omega_history,
        df_engineered=df.loc[fsi_series.index],
        stability_series=cos_sim_series,
        stability_threshold=float(config['fsi']['stability_threshold']),
        freeze_after_days=int(config['fsi'].get('freeze_after_days', 90)),
        min_corr_to_freeze=0.10,
        allow_flip_cosine_thresh=0.2
    )

    # # after obtaining fsi_series, omega_history we chack the correlation 
    # W = int(config['fsi']['window_size'])
    # idx_tail = fsi_series.index[-60:]
    # mu = df.rolling(W).mean().reindex(idx_tail)
    # sd = df.rolling(W).std().replace(0, np.nan).reindex(idx_tail)
    # Z = (df.reindex(idx_tail) - mu) / sd

    # common = omega_history.columns.intersection(Z.columns)
    # approx_fsi = (Z[common] * omega_history.reindex(idx_tail)[common]).sum(axis=1)
    # corr = approx_fsi.corr(fsi_series.reindex(idx_tail))
    # logging.info(f"[QC] Corr(FSI, Σ z⊙ω) last 60d = {corr:.3f}")

    min_history = int(config['fsi']['window_size'])
    idx_tail = fsi_series.index[-60:]

    mu = df.expanding(min_periods=min_history).mean().reindex(idx_tail)
    sd = df.expanding(min_periods=min_history).std().replace(0, np.nan).reindex(idx_tail)
    Z = (df.reindex(idx_tail) - mu) / sd

    common = omega_history.columns.intersection(Z.columns)
    approx_fsi = (Z[common] * omega_history.reindex(idx_tail)[common]).sum(axis=1)
    corr = approx_fsi.corr(fsi_series.reindex(idx_tail))
    logging.info(f"[QC] Corr(FSI, Σ z⊙ω) last 60d = {corr:.3f}")



    # Persist audit log
    base_path = "./cache-directory"
    orient_audit_path = os.path.join(base_path, "qc", "orientation_flip_audit.csv")
    os.makedirs(os.path.dirname(orient_audit_path), exist_ok=True)
    if not orient_audit.empty:
        orient_audit.to_csv(orient_audit_path, index=False)
        logging.warning(f"[ORIENT] Flip audit saved: {orient_audit_path}")
    else:
        pd.DataFrame(columns=["date","reason"]).to_csv(orient_audit_path, index=False)


    # # A1: leakage-free contributions using contemporaneous ω_t and window-standardization
    # logging.info("Computing contributions...")
    # variable_contribs = compute_timevarying_contributions(
    #     df.loc[fsi_series.index], omega_history, window_size=int(config['fsi']['window_size'])
    # )

    logging.info("Computing contributions...")
    min_history = int(config['fsi']['window_size'])
    variable_contribs = compute_timevarying_contributions(
        df.loc[fsi_series.index],
        omega_history,
        min_history=min_history
    )



    group_map = build_dynamic_group_map(variable_contribs)  # build from actually PRESENT columns
    grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)

    # Check that grouped sum equals FSI within tolerance (floating error)
    tol = 1e-8
    err = (grouped_contribs.drop(columns=['FSI']).sum(axis=1) - grouped_contribs['FSI']).abs().max()
    if pd.notna(err) and err > tol:
        logging.warning(f"[ATTR] Group attribution mismatch max={err:.2e} (tolerance {tol:.1e}).")

    # Regime classification on oriented FSI
    fsi = variable_contribs['FSI']
    regimes = classify_risk_regime_hybrid(fsi)
    print("Regime classification value counts:\n", regimes.value_counts())

    # Plot
    logging.info("Plotting results...")
    fig1 = plot_group_contributions_with_regime(variable_contribs, regimes=regimes)
    fig2 = plot_grouped_contributions(grouped_contribs, regimes=regimes)

    # PnL chart (pass regimes so ribbons match)
    try:
        pnl_df = pd.read_excel(config['data']['pnl_file'], index_col=0, sheet_name='PnL')
        fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series, regimes=regimes)
    except Exception as e:
        logging.error(f"Error loading or plotting PnL data: {e}", exc_info=True)
        fig_pnl = None

    save_fsi_charts_to_html(fig1, fig2, fig_pnl)

if __name__ == '__main__':
    main()




