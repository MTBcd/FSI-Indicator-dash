# main.py
import logging
import pandas as pd
import configparser
import numpy as np
import logging
from data_fetching import get_all_series
# from data_fetching import get_ibkr_series, get_fred_series, load_extended_csv_data, scrape_investing_data
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

# # Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_configuration(config_file='config.ini'):
    """Load configuration from a .ini file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    return config



def merge_data(config):
    try:
        # --- 1. Fetch initial raw data ---
        df = get_all_series(config)
        if df.empty:
            logging.error("Merged data is empty right after fetching. Exiting.")
            return None

        base_path = config['data']['csv_base_path']
        df.to_csv(f"{base_path}\\Full_set_variables_brut.csv")

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
        pre_dropna_shape = df.shape
        df = df.dropna(axis=1, thresh=int(0.9 * len(df)))
        dropped_cols = set(df.columns) ^ set(df.columns)
        logging.info(f"Dropped {pre_dropna_shape[1] - df.shape[1]} columns with >10% NaN. Shape now: {df.shape}")

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
            if '1Y Treasury Yield' in df.columns:
                df[f'1Y_rate_{window}'] = absolute_deviation(df['1Y Treasury Yield'], window, invert=True)
            if '2Y Treasury Yield' in df.columns:
                df[f'2Y_rate_{window}'] = absolute_deviation(df['2Y Treasury Yield'], window, invert=True)
            if '3M T-Bill Yield' in df.columns:
                df[f'3M_TBill_stress_{window}'] = absolute_deviation_rotated(df['3M T-Bill Yield'], window)

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
            if 'SPY P/E' in df.columns:
                df[f'SPY_PE_dev_{window}'] = moving_average_deviation(df['SPY P/E'], window)
            if 'SPY P/B' in df.columns:
                df[f'SPY_PB_dev_{window}'] = moving_average_deviation(df['SPY P/B'], window)

        # --- 7. Drop raw columns to keep only engineered features ---
        raw_cols = [
            'VIX', 'MOVE Index', 'USD Index (DXY)', 'Gold Price',
            'US 10Y Treasury Yield', 'USD Overnight Rate', 'FRED RRP Volume', '3M T-Bill Yield',
            'US IG OAS', 'US HY OAS', '1Y Treasury Yield', '2Y Treasury Yield', 
            'SPY P/E', '10Y-2Y Slope', '10Y-3M Slope',
            'HYG-LQD Spread', 'SPY P/B', 'OVX', 'VXV', 'VIX-VXV Spread'
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

        # --- 9. Save and return ---
        df.to_csv(f"{base_path}\\Full_set_variables_std.csv")
        logging.info("Final merged and processed dataset saved and ready.")

        return df

    except Exception as e:
        logging.error(f"Error merging data: {e}", exc_info=True)
        return None





# def merge_data(config):
#     """Merge data from different sources."""
#     try:
#         market_data = get_ibkr_series(config)
#         fred_data = get_fred_series(config)

#         if not market_data or not fred_data:
#             logging.error("One or more data sources returned empty data. Check API keys and connectivity.")
#             return None

#         # Add Yield Slope
#         fred_data['10Y-2Y Yield Slope'] = fred_data['10Y Yield'] - fred_data['2Y Yield']

#         df = pd.concat({**market_data, **fred_data}, axis=1)
#         df = df.sort_index()

#         base_path = config['data']['csv_base_path']
#         df.to_csv(f"{base_path}\\Full_set_variables_brut.csv")

#         # Align to the latest first-valid point
#         first_valid_dates = df.apply(lambda col: col.first_valid_index())
#         cutoff_date = max(first_valid_dates)
#         df = df[df.index >= cutoff_date]

#         df = df.ffill().bfill()  # Dual fill
#         df = df.dropna(axis=1, thresh=int(0.9 * len(df)))  # Drop weak series
#         df = df.dropna()

#         # Feature Engineering
#         windows = [int(w) for w in config['fsi']['windows'].split(',')]
#         for window in windows:
#             # Volatility
#             df[f'VIX_dev_{window}'] = moving_average_deviation(df['VIX'], window)
#             df[f'VXV_dev_{window}'] = moving_average_deviation(df['VXV'], window)
#             df[f'OVX_dev_{window}'] = moving_average_deviation(df['OVX'], window)
#             df[f'GVZ_dev_{window}'] = moving_average_deviation(df['GVZ'], window)

#             # Safe Assets
#             df[f'USD_stress_{window}'] = moving_average_deviation(df['USD Index'], window, invert=True)
#             df[f'3M_TBill_stress_{window}'] = absolute_deviation_rotated(df['3M T-Bill'], window)

#             # Funding
#             df[f'Credit_spread_{window}'] = absolute_deviation(df['Credit Spread (HYG - LQD)'], window)
#             df[f'USDO_rate_dev_{window}'] = moving_average_deviation(df['USD Overnight Rate'], window, invert=True)
#             df[f'Fed_RRP_stress_{window}'] = absolute_deviation(df['FRED RRP'], window, invert=True)

#             # Rate Stress
#             df[f'10Y_rate_{window}'] = absolute_deviation(df['10Y Yield'], window, invert=True)
#             df[f'1Y_rate_{window}'] = absolute_deviation(df['1Y Yield'], window, invert=True)

#             # Credit stress
#             df[f'Yield_slope_{window}'] = absolute_deviation(df['10Y-2Y Yield Slope'], window, invert=True)
#             df[f'Corp_OAS_dev_{window}'] = absolute_deviation(df['US Corp OAS'], window)
#             df[f'HY_OAS_dev_{window}'] = absolute_deviation(df['US HY OAS'], window)

#         # Drop raw columns after feature engineering is complete (outside loop)
#         df.drop([
#             'USD Index', '3M T-Bill', 'Credit Spread (HYG - LQD)', '10Y-2Y Yield Slope',
#             'VIX', 'VXV', '10Y Yield', '2Y Yield', 'OVX', 'GVZ', '1Y Yield',
#             'USD Overnight Rate', 'FRED RRP', 'US Corp OAS', 'US HY OAS'
#         ], axis=1, inplace=True, errors='ignore')

#         # Save final processed dataset
#         df.to_csv(f"{base_path}\\Full_set_variables_std.csv")

#         logging.info("Final merged and processed dataset.")
#         return df

#     except Exception as e:
#         logging.error(f"Error merging data: {e}", exc_info=True)
#         return None







def main():
    """Main function to orchestrate the FSI estimation and plotting."""
    config = load_configuration()
    # config = configparser.ConfigParser()
    # config.read('config.ini')

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

    # Optional: Flip sign based on VIX/SPX logic for interpretability
    vix_cols = [c for c in df.columns if 'VIX' in c or 'SPX_vol' in c]
    for col in vix_cols:
        if col in omega_history.columns and omega_history.iloc[-1][col] < 0:
            fsi_series *= -1
            omega_history *= -1
            break

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
    # group_map = {
    #     'Volatility': ['VIX_dev', 'VXV_dev', 'OVX_dev', 'GVZ_dev'],
    #     'Rates': ['10Y_rate', '1Y_rate', 'Yield_slope', 'USDO_rate_dev'],
    #     'Funding': ['USD_stress', '3M_TBill_stress', 'Fed_RRP_stress'],
    #     'Credit': ['Credit_spread', 'Corp_OAS_dev', 'HY_OAS_dev'],
    # }

    group_map = {
        'Volatility': ['VIX_dev', 'MOVE_dev', 'OVX_dev', 'VXV_dev', 'VIX_VXV_spread_dev'],
        'Rates': ['10Y_rate', '1Y_rate', '2Y_rate', '10Y_2Y_slope_dev', '10Y_3M_slope_dev', 'USDO_rate_dev'],
        'Funding': ['USD_stress', '3M_TBill_stress', 'Fed_RRP_stress',],
        'Credit': ['IG_OAS_dev', 'HY_OAS_dev', 'HY_IG_spread'],
        'Valuation': ['SPY_PE_dev', 'SPY_PB_dev'],
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














# import logging
# import pandas as pd
# import configparser
# from data_fetching import get_fmp_series
# from fsi_estimation import estimate_fsi_recursive_rolling_with_stability, compute_variable_contributions
# from plotting import (
#     plot_group_contributions_with_regime, plot_grouped_contributions,
#     plot_pnl_with_regime_ribbons, save_fsi_charts_to_html
# )
# from utils import (
#     aggregate_contributions_by_group,
#     moving_average_deviation, absolute_deviation,
#     classify_risk_regime_hybrid
# )

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def load_configuration(config_file='config.ini'):
#     config = configparser.ConfigParser()
#     config.read(config_file)
#     return config

# def merge_data(config):
#     """Merge data using only FMP as the source."""
#     try:
#         fmp_data = get_fmp_series(config)

#         # If fmp_data is a dict, convert to DataFrame
#         if isinstance(fmp_data, dict):
#             df = pd.DataFrame(fmp_data)
#         else:
#             df = fmp_data.copy()

#         if df.empty:
#             logging.error("FMP data returned empty data. Check API keys and connectivity.")
#             return None

#         # Align index (optional: sort if you fetch timeseries)
#         df = df.sort_index()

#         base_path = config['data']['csv_base_path']
#         df.to_csv(f"{base_path}/Full_set_variables_brut.csv")

#         # Align to the latest first-valid point
#         first_valid_dates = df.apply(lambda col: col.first_valid_index())
#         cutoff_date = max(first_valid_dates)
#         df = df[df.index >= cutoff_date]

#         df = df.ffill().bfill()
#         df = df.dropna(axis=1, thresh=int(0.9 * len(df)))
#         df = df.dropna()

#         # Feature Engineering
#         windows = [int(w) for w in config['fsi']['windows'].split(',')]
#         for window in windows:
#             if 'VIX' in df.columns:
#                 df[f'VIX_dev_{window}'] = moving_average_deviation(df['VIX'], window)
#             if 'VXV' in df.columns:
#                 df[f'VXV_dev_{window}'] = moving_average_deviation(df['VXV'], window)
#             if 'USD Index (DXY)' in df.columns:
#                 df[f'USD_stress_{window}'] = moving_average_deviation(df['USD Index (DXY)'], window, invert=True)
#             if 'Gold Price' in df.columns:
#                 df[f'Gold_dev_{window}'] = moving_average_deviation(df['Gold Price'], window)
#             if 'US 10Y Treasury Yield' in df.columns:
#                 df[f'10Y_Yield_dev_{window}'] = moving_average_deviation(df['US 10Y Treasury Yield'], window)
#             if 'USD Overnight Rate' in df.columns:
#                 df[f'USDO_rate_dev_{window}'] = moving_average_deviation(df['USD Overnight Rate'], window, invert=True)
#             if 'TED Spread (3M LIBOR - 3M T-Bill)' in df.columns:
#                 df[f'TED_spread_dev_{window}'] = absolute_deviation(df['TED Spread (3M LIBOR - 3M T-Bill)'], window)
#             if 'S&P 500 P/E Ratio' in df.columns:
#                 df[f'SP500_PE_dev_{window}'] = moving_average_deviation(df['S&P 500 P/E Ratio'], window)

#         df.to_csv(f"{base_path}/Full_set_variables_std.csv")

#         logging.info("Final merged and processed dataset ready.")
#         return df

#     except Exception as e:
#         logging.error(f"Error merging data: {e}", exc_info=True)
#         return None

# def main():
#     """Main function to orchestrate the FSI estimation and plotting."""
#     config = load_configuration()

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

#     latest_omega = omega_history.iloc[-1]
#     variable_contribs = compute_variable_contributions(df.loc[fsi_series.index], latest_omega)

#     group_map = {
#         'Volatility': ['VIX_dev', 'VXV_dev'],
#         'Safe Assets': ['USD_stress', 'Gold_dev', '10Y_Yield_dev'],
#         'Funding': ['USDO_rate_dev', 'TED_spread_dev'],
#         'Valuation': ['SP500_PE_dev'],
#     }

#     grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)

#     fsi = variable_contribs['FSI']
#     regimes = classify_risk_regime_hybrid(fsi)

#     fig1 = plot_group_contributions_with_regime(variable_contribs)
#     fig2 = plot_grouped_contributions(grouped_contribs)

#     try:
#         pnl_df = pd.read_excel(config['data']['pnl_file'], index_col=0, sheet_name='PnL')
#         fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series)
#     except Exception as e:
#         logging.error(f"Error loading or plotting PnL data: {e}", exc_info=True)
#         fig_pnl = None

#     output_file = config['output']['output_file']
#     save_fsi_charts_to_html(fig1, fig2, fig_pnl, filename=output_file)

# if __name__ == '__main__':
#     main()
