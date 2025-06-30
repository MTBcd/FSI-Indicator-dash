# # main.py
# import logging
# import pandas as pd
# import configparser
# import logging
# import numpy as np
# import os
# import time
# from data_fetching import get_all_series
# # from data_fetching import get_ibkr_series, get_fred_series, load_extended_csv_data, scrape_investing_data
# from fsi_estimation import estimate_fsi_recursive_rolling_with_stability, compute_variable_contributions
# from plotting import (
#     plot_group_contributions_with_regime, plot_grouped_contributions,
#     plot_pnl_with_regime_ribbons, save_fsi_charts_to_html
# )
# from utils import (
#     aggregate_contributions_by_group, smooth_transition_regime, regime_from_smooth_weight,
#     moving_average_deviation, absolute_deviation_rotated, absolute_deviation,
#     classify_risk_regime_hybrid
# )

# # # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def load_configuration(config_file='config.ini'):
#     """Load configuration from a .ini file."""
#     config = configparser.ConfigParser()
#     config.read(config_file)
#     return config


# def merge_data(config, max_age_hours=12):
#     """
#     Loads processed data from cache if recent, otherwise runs full pipeline and caches result.
#     """
#     base_path = config['data']['csv_base_path']
#     cache_path = os.path.join(base_path, "fsi_data_latest.parquet")

#     try:
#         # --- 0. Use cache if recent ---
#         if os.path.exists(cache_path):
#             file_age = (time.time() - os.path.getmtime(cache_path)) / 3600.0
#             if file_age < max_age_hours:
#                 logging.info(f"[merge_data] Loading cached data ({cache_path}), age: {file_age:.2f} hours")
#                 return pd.read_parquet(cache_path)
#             else:
#                 logging.info(f"[merge_data] Cache too old ({file_age:.1f}h > {max_age_hours}h), refetching.")

#         # --- 1. Fetch initial raw data ---
#         df = get_all_series(config)
#         if df.empty:
#             logging.error("Merged data is empty right after fetching. Exiting.")
#             return None

#         df.to_csv(f"{base_path}\\Full_set_variables_brut.csv")

#         logging.info(f"Initial columns: {list(df.columns)}")
#         logging.info(f"Initial shape: {df.shape}")

#         # --- 2. Align to the latest first-valid date ---
#         first_valid_dates = df.apply(lambda col: col.first_valid_index())
#         cutoff_date = max(first_valid_dates)
#         logging.info(f"Latest first-valid date across columns: {cutoff_date}")
#         df = df[df.index >= cutoff_date]
#         logging.info(f"Shape after aligning to cutoff date: {df.shape}")

#         # --- 3. Forward/backward fill, drop columns with too much missing ---
#         df = df.ffill().bfill()
#         cols_before_drop = set(df.columns)
#         df = df.dropna(axis=1, thresh=int(0.9 * len(df)))
#         cols_after_drop = set(df.columns)
#         dropped_cols = cols_before_drop - cols_after_drop
#         logging.info(f"Dropped {dropped_cols} columns with >10% NaN. Shape now: {df.shape}")

#         # --- 4. Final drop of any remaining NaN rows ---
#         df = df.dropna()
#         logging.info(f"Shape after dropping remaining NaN rows: {df.shape}")

#         # --- 5. Defensive check for empty or too-narrow dataframe ---
#         if df.empty or df.shape[1] < 2:
#             logging.error(f"Final cleaned dataset is empty or too narrow for SVD. Columns: {df.columns.tolist()}")
#             return None

#         # === 6. Feature Engineering ===
#         windows = [int(w) for w in config['fsi']['windows'].split(',')]

#         for window in windows:
#             # --- Volatility ---
#             if 'VIX' in df.columns:
#                 df[f'VIX_dev_{window}'] = moving_average_deviation(df['VIX'], window)
#             if 'MOVE Index' in df.columns:
#                 df[f'MOVE_dev_{window}'] = moving_average_deviation(df['MOVE Index'], window)
#             if 'OVX' in df.columns:
#                 df[f'OVX_dev_{window}'] = moving_average_deviation(df['OVX'], window)
#             if 'VXV' in df.columns:
#                 df[f'VXV_dev_{window}'] = moving_average_deviation(df['VXV'], window)
#             if 'VIX-VXV Spread' in df.columns:
#                 df[f'VIX_VXV_spread_dev_{window}'] = moving_average_deviation(df['VIX-VXV Spread'], window)

#             # --- Safe-Haven / FX ---
#             if 'USD Index (DXY)' in df.columns:
#                 df[f'USD_stress_{window}'] = moving_average_deviation(df['USD Index (DXY)'], window, invert=True)
#             if 'Gold Price' in df.columns:
#                 df[f'Gold_dev_{window}'] = moving_average_deviation(df['Gold Price'], window)

#             # --- Rates ---
#             if 'US 10Y Treasury Yield' in df.columns:
#                 df[f'10Y_rate_{window}'] = absolute_deviation(df['US 10Y Treasury Yield'], window, invert=True)
#             # if '1Y Treasury Yield' in df.columns:
#             #     df[f'1Y_rate_{window}'] = absolute_deviation(df['1Y Treasury Yield'], window, invert=True)
#             if '2Y Treasury Yield' in df.columns:
#                 df[f'2Y_rate_{window}'] = absolute_deviation(df['2Y Treasury Yield'], window, invert=True)
#             if '3M T-Bill Yield' in df.columns:
#                 df[f'3M_TBill_stress_{window}'] = absolute_deviation_rotated(df['3M T-Bill Yield'], window)
#             if 'FRED RRP' in df.columns:
#                 df[f'FRED_RRP_stress_{window}'] = absolute_deviation_rotated(df['FRED RRP'], window)

#             # --- Credit & OAS ---
#             if 'US IG OAS' in df.columns:
#                 df[f'IG_OAS_dev_{window}'] = absolute_deviation(df['US IG OAS'], window)
#             if 'US HY OAS' in df.columns:
#                 df[f'HY_OAS_dev_{window}'] = absolute_deviation(df['US HY OAS'], window)
#             if 'HYG-LQD Spread' in df.columns:
#                 df[f'HY_IG_spread_{window}'] = moving_average_deviation(df['HYG-LQD Spread'], window)

#             # --- Funding & Liquidity ---
#             if 'USD Overnight Rate' in df.columns:
#                 df[f'USDO_rate_dev_{window}'] = moving_average_deviation(df['USD Overnight Rate'], window, invert=True)
#             if 'FRED RRP Volume' in df.columns:
#                 df[f'Fed_RRP_stress_{window}'] = absolute_deviation(df['FRED RRP Volume'], window, invert=True)

#             # --- Slope & Spreads ---
#             if '10Y-2Y Slope' in df.columns:
#                 df[f'10Y_2Y_slope_dev_{window}'] = absolute_deviation(df['10Y-2Y Slope'], window, invert=True)
#             if '10Y-3M Slope' in df.columns:
#                 df[f'10Y_3M_slope_dev_{window}'] = absolute_deviation(df['10Y-3M Slope'], window, invert=True)

#             # --- Valuation ---
#             # if 'SPY P/E' in df.columns:
#             #     df[f'SPY_PE_dev_{window}'] = moving_average_deviation(df['SPY P/E'], window)
#             # if 'SPY P/B' in df.columns:
#             #     df[f'SPY_PB_dev_{window}'] = moving_average_deviation(df['SPY P/B'], window)

#         # --- 7. Drop raw columns to keep only engineered features ---
#         raw_cols = [
#             'VIX', 'MOVE Index', 'USD Index (DXY)', 'Gold Price',
#             'US 10Y Treasury Yield', 'USD Overnight Rate', 'FRED RRP Volume', '3M T-Bill Yield',
#             'US IG OAS', 'US HY OAS', '1Y Treasury Yield', '2Y Treasury Yield', 
#             'SPY P/E', '10Y-2Y Slope', '10Y-3M Slope',
#             'HYG-LQD Spread', 'SPY P/B', 'OVX', 'VXV', 'VIX-VXV Spread',
#             "3M T-Bill", "10Y Yield", "2Y Yield", "USD Index", "FRED RRP", "US Corp OAS"
#         ]
#         df.drop(raw_cols, axis=1, inplace=True, errors='ignore')

#         logging.info(f"Shape after dropping raw columns: {df.shape}")
#         if df.empty or df.shape[1] < 2:
#             logging.error(f"Final engineered dataset is empty or too narrow for SVD. Columns: {df.columns.tolist()}")
#             return None

#         # --- 8. Final NaN drop (should be minimal) ---
#         df = df.dropna()
#         logging.info(f"Shape after final dropna: {df.shape}")

#         if df.empty or df.shape[1] < 2:
#             logging.error(f"Final processed dataset is empty or too narrow for SVD. Columns: {df.columns.tolist()}")
#             return None

#         # --- 9. Save as CSV and cache as Parquet ---
#         df.to_csv(f"{base_path}\\Full_set_variables_std.csv")
#         df.to_parquet(cache_path)
#         logging.info(f"Final merged and processed dataset saved and cached at {cache_path}.")

#         return df

#     except Exception as e:
#         logging.error(f"Error merging data: {e}", exc_info=True)
#         return None




# def main():
#     """Main function to orchestrate the FSI estimation and plotting."""
#     config = load_configuration()
#     # config = configparser.ConfigParser()
#     # config.read('config.ini')

#     df = merge_data(config)
#     if df is None:
#         logging.error("Failed to merge data. Exiting.")
#         return

#     fsi_series, omega_history, cos_sim_series, unstable_dates = estimate_fsi_recursive_rolling_with_stability(
#         df,
#         window_size=int(config['fsi']['window_size']),
#         n_iter=int(config['fsi']['n_iter']),
#         stability_threshold=float(config['fsi']['stability_threshold'])
#     )

#     # Broader anchor set with known positive relation to stress
#     anchor_vars = ['VIX_dev_250', 'MOVE_dev_250', 'HY_OAS_dev_250', 'IG_OAS_dev_250']

#     # Check if these exist in omega
#     available_anchors = [col for col in anchor_vars if col in omega_history.columns]

#     # Average their weights
#     anchor_sign = np.sign(omega_history.iloc[-1][available_anchors].mean())

#     # Flip if they're collectively negative
#     if anchor_sign < 0:
#         fsi_series *= -1
#         omega_history *= -1

#     # === ω Stability Diagnostics ===
#     if unstable_dates:
#         logging.warning(f"Detected unstable ω estimates on {len(unstable_dates)} days:")
#         for date in unstable_dates:
#             logging.warning(f" - {date.strftime('%Y-%m-%d')} (cos_sim = {cos_sim_series.loc[date]:.3f})")

#     # === Compute contributions using latest omega ===
#     logging.info("Computing contributions...")
#     latest_omega = omega_history.iloc[-1]
#     variable_contribs = compute_variable_contributions(df.loc[fsi_series.index], latest_omega)

#     # === Group attribution ===
#     logging.info("Aggregating and plotting group-level contributions...")

#     group_map = {
#         'Volatility': ['VIX_dev_250', 'MOVE_dev_250', 'OVX_dev_250', 'VXV_dev_250', 'VIX_VXV_spread_dev_250'],
#         'Rates': ['10Y_rate_250', '2Y_rate_250', '10Y_2Y_slope_dev_250', '10Y_3M_slope_dev_250', 'USDO_rate_dev_250'],
#         'Funding': ['USD_stress_250', '3M_TBill_stress_250', 'Fed_RRP_stress_250', 'FRED_RRP_stress_250'],
#         'Credit': ['IG_OAS_dev_250', 'HY_OAS_dev_250', 'HY_IG_spread_250'],
#         'Safe_Haven': ['Gold_dev_250']
#     }

#     grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)

#     # === Regime Classification ===
#     fsi = variable_contribs['FSI']
#     regimes = classify_risk_regime_hybrid(fsi)

#     logging.info("Plotting results...")
#     fig1 = plot_group_contributions_with_regime(variable_contribs)
#     fig2 = plot_grouped_contributions(grouped_contribs)

#     # Load PnL data and plot
#     try:
#         pnl_df = pd.read_excel(config['data']['pnl_file'], index_col=0, sheet_name='PnL')
#         fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series)
#     except Exception as e:
#         logging.error(f"Error loading or plotting PnL data: {e}", exc_info=True)
#         fig_pnl = None

#     # Save charts to HTML
#     output_file = config['output']['output_file']
#     save_fsi_charts_to_html(fig1, fig2, fig_pnl)

# if __name__ == '__main__':
#     main()





# main.py

import logging
import pandas as pd
import configparser
import numpy as np
import os
import time
from data_fetching import get_all_series
from fsi_estimation import estimate_fsi_recursive_rolling_with_stability, compute_variable_contributions
from plotting import (
    plot_group_contributions_with_regime, plot_grouped_contributions,
    plot_pnl_with_regime_ribbons, save_fsi_charts_to_html
)
from utils import (
    aggregate_contributions_by_group, smooth_transition_regime, regime_from_smooth_weight,
    moving_average_deviation, absolute_deviation_rotated, absolute_deviation,
    classify_risk_regime_hybrid
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_configuration(config_file='config.ini'):
    """Load configuration from a .ini file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config


def merge_data(config, max_age_hours=12):
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
            # --- Volatility ---
            if 'VIX' in df.columns:
                df[f'VIX_dev_{window}'] = moving_average_deviation(df['VIX'], window)
            if 'MOVE Index' in df.columns:
                df[f'MOVE_dev_{window}'] = moving_average_deviation(df['MOVE Index'], window)
            if 'OVX' in df.columns:
                df[f'OVX_dev_{window}'] = moving_average_deviation(df['OVX'], window)
            if 'VXV' in df.columns:
                df[f'VXV_dev_{window}'] = moving_average_deviation(df['VXV'], window)
            if 'VIX-VXV Spread' in df.columns:
                df[f'VIX_VXV_spread_dev_{window}'] = moving_average_deviation(df['VIX-VXV Spread'], window)

            # --- Safe-Haven / FX ---
            if 'USD Index (DXY)' in df.columns:
                df[f'USD_stress_{window}'] = moving_average_deviation(df['USD Index (DXY)'], window, invert=True)
            if 'Gold Price' in df.columns:
                df[f'Gold_dev_{window}'] = moving_average_deviation(df['Gold Price'], window)

            # --- Rates ---
            if 'US 10Y Treasury Yield' in df.columns:
                df[f'10Y_rate_{window}'] = absolute_deviation(df['US 10Y Treasury Yield'], window, invert=True)
            # if '1Y Treasury Yield' in df.columns:
            #     df[f'1Y_rate_{window}'] = absolute_deviation(df['1Y Treasury Yield'], window, invert=True)
            if '2Y Treasury Yield' in df.columns:
                df[f'2Y_rate_{window}'] = absolute_deviation(df['2Y Treasury Yield'], window, invert=True)
            if '3M T-Bill Yield' in df.columns:
                df[f'3M_TBill_stress_{window}'] = absolute_deviation_rotated(df['3M T-Bill Yield'], window)
            if 'FRED RRP' in df.columns:
                df[f'FRED_RRP_stress_{window}'] = absolute_deviation_rotated(df['FRED RRP'], window)

            # --- Credit & OAS ---
            if 'US IG OAS' in df.columns:
                df[f'IG_OAS_dev_{window}'] = absolute_deviation(df['US IG OAS'], window)
            if 'US HY OAS' in df.columns:
                df[f'HY_OAS_dev_{window}'] = absolute_deviation(df['US HY OAS'], window)
            if 'HYG-LQD Spread' in df.columns:
                df[f'HY_IG_spread_{window}'] = moving_average_deviation(df['HYG-LQD Spread'], window)

            # --- Funding & Liquidity ---
            if 'USD Overnight Rate' in df.columns:
                df[f'USDO_rate_dev_{window}'] = moving_average_deviation(df['USD Overnight Rate'], window, invert=True)
            if 'FRED RRP Volume' in df.columns:
                df[f'Fed_RRP_stress_{window}'] = absolute_deviation(df['FRED RRP Volume'], window, invert=True)

            # --- Slope & Spreads ---
            if '10Y-2Y Slope' in df.columns:
                df[f'10Y_2Y_slope_dev_{window}'] = absolute_deviation(df['10Y-2Y Slope'], window, invert=True)
            if '10Y-3M Slope' in df.columns:
                df[f'10Y_3M_slope_dev_{window}'] = absolute_deviation(df['10Y-3M Slope'], window, invert=True)

            # --- Valuation ---
            # if 'SPY P/E' in df.columns:
            #     df[f'SPY_PE_dev_{window}'] = moving_average_deviation(df['SPY P/E'], window)
            # if 'SPY P/B' in df.columns:
            #     df[f'SPY_PB_dev_{window}'] = moving_average_deviation(df['SPY P/B'], window)

        # --- 7. Drop raw columns to keep only engineered features ---
        raw_cols = [
            'VIX', 'MOVE Index', 'USD Index (DXY)', 'Gold Price',
            'US 10Y Treasury Yield', 'USD Overnight Rate', 'FRED RRP Volume', '3M T-Bill Yield',
            'US IG OAS', 'US HY OAS', '1Y Treasury Yield', '2Y Treasury Yield',
            'SPY P/E', '10Y-2Y Slope', '10Y-3M Slope',
            'HYG-LQD Spread', 'SPY P/B', 'OVX', 'VXV', 'VIX-VXV Spread',
            "3M T-Bill", "10Y Yield", "2Y Yield", "USD Index", "FRED RRP", "US Corp OAS"
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
    """Main function to orchestrate the FSI estimation and plotting."""
    config = load_configuration()
    df = merge_data(config)
    if df is None:
        logging.error("Failed to merge data. Exiting.")
        return

    fsi_series, omega_history, cos_sim_series, unstable_dates = estimate_fsi_recursive_rolling_with_stability(
        df,
        window_size=int(config['fsi']['window_size']),
        n_iter=int(config['fsi']['n_iter']),
        stability_threshold=float(config['fsi']['stability_threshold'])
    )

    # Broader anchor set with known positive relation to stress
    anchor_vars = ['VIX_dev_250', 'MOVE_dev_250', 'HY_OAS_dev_250', 'IG_OAS_dev_250']

    # Check if these exist in omega
    available_anchors = [col for col in anchor_vars if col in omega_history.columns]

    # Average their weights
    anchor_sign = np.sign(omega_history.iloc[-1][available_anchors].mean())

    # Flip if they're collectively negative
    if anchor_sign < 0:
        fsi_series *= -1
        omega_history *= -1

    # === ω Stability Diagnostics ===
    if unstable_dates:
        logging.warning(f"Detected unstable ω estimates on {len(unstable_dates)} days:")
        for date in unstable_dates:
            logging.warning(f" - {date.strftime('%Y-%m-%d')} (cos_sim = {cos_sim_series.loc[date]:.3f})")

    # === Compute contributions using latest omega ===
    logging.info("Computing contributions...")
    latest_omega = omega_history.iloc[-1]
    variable_contribs = compute_variable_contributions(df.loc[fsi_series.index], latest_omega)

    # === Group attribution ===
    logging.info("Aggregating and plotting group-level contributions...")

    group_map = {
        'Volatility': ['VIX_dev_250', 'MOVE_dev_250', 'OVX_dev_250', 'VXV_dev_250', 'VIX_VXV_spread_dev_250'],
        'Rates': ['10Y_rate_250', '2Y_rate_250', '10Y_2Y_slope_dev_250', '10Y_3M_slope_dev_250', 'USDO_rate_dev_250'],
        'Funding': ['USD_stress_250', '3M_TBill_stress_250', 'Fed_RRP_stress_250', 'FRED_RRP_stress_250'],
        'Credit': ['IG_OAS_dev_250', 'HY_OAS_dev_250', 'HY_IG_spread_250'],
        'Safe_Haven': ['Gold_dev_250']
    }

    grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)

    # === Regime Classification ===
    fsi = variable_contribs['FSI']
    regimes = classify_risk_regime_hybrid(fsi)

    logging.info("Plotting results...")
    fig1 = plot_group_contributions_with_regime(variable_contribs)
    fig2 = plot_grouped_contributions(grouped_contribs)

    # Load PnL data and plot
    try:
        pnl_df = pd.read_excel(config['data']['pnl_file'], index_col=0, sheet_name='PnL')
        fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series)
    except Exception as e:
        logging.error(f"Error loading or plotting PnL data: {e}", exc_info=True)
        fig_pnl = None

    # Save charts to HTML
    output_file = config['output']['output_file']
    save_fsi_charts_to_html(fig1, fig2, fig_pnl)

if __name__ == '__main__':
    main()
