


# data_fetching.py
import pandas as pd
import numpy as np
import requests
import logging
from fredapi import Fred
import configparser

# ======== Config Loader ==========

config = configparser.ConfigParser()
config.read('config.ini')

# ======== FMP PRICE DATA ==========

def get_price_series(ticker, api_key, start_date="2017-01-01"):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&apikey={api_key}"
    try:
        resp = requests.get(url)
        data = resp.json()
        if 'historical' not in data:
            logging.warning(f"No price data found for {ticker}.")
            return pd.Series(dtype=float, name=f"{ticker} Price")
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        return df['close']
    except Exception as e:
        logging.error(f"Error fetching FMP price data for {ticker}: {e}")
        return pd.Series(dtype=float, name=f"{ticker} Price")

def fetch_fmp_data(ticker, api_key, start_date="2017-01-01"):
    return get_price_series(ticker, api_key, start_date)

def get_hyg_lqd_spread(api_key, start_date="2017-01-01"):
    hyg = get_price_series('HYG', api_key, start_date=start_date)
    lqd = get_price_series('LQD', api_key, start_date=start_date)
    common_idx = hyg.index.intersection(lqd.index)
    if common_idx.empty:
        return pd.Series(name='HYG-LQD Spread')
    spread = hyg.loc[common_idx] - lqd.loc[common_idx]
    spread.name = 'HYG-LQD Spread'
    return spread

# ======== FRED MULTI-SERIES FETCHER ==========

def get_fred_series(fred_api_key, start_date, series_map=None):
    """
    Fetches multiple FRED series as a dict of pandas Series, clipped to start_date.
    :param fred_api_key: Your FRED API key (str)
    :param start_date: Start date as 'YYYY-MM-DD' or datetime
    :param series_map: dict {output_col: FRED_series_id}
    :return: dict {output_col: pd.Series}
    """
    fred = Fred(api_key=fred_api_key)
    start_date = pd.to_datetime(start_date)
    if series_map is None:
        series_map = {
            'VXV': 'VXVCLS',
            'VIX': 'VIXCLS',
            'OVX': 'OVXCLS',
            'USD Overnight Rate': 'OBFR',
            '3M T-Bill': 'DTB3',
            '10Y Yield': 'DGS10',
            '2Y Yield': 'DGS2',
            'USD Index': 'DTWEXBGS',
            'FRED RRP': 'RRPONTSYD',
            'US Corp OAS': 'BAMLC0A0CM',
            'US HY OAS': 'BAMLH0A0HYM2'
        }
    result = {}
    for out_col, fred_id in series_map.items():
        try:
            s = fred.get_series(fred_id)
            s.index = pd.to_datetime(s.index)
            s = s[s.index >= start_date]
            s.name = out_col
            result[out_col] = s
        except Exception as e:
            logging.warning(f"Could not fetch FRED series {fred_id} for {out_col}: {e}")
            result[out_col] = pd.Series(dtype=float, name=out_col)
    return result

# ======== MASTER DATA FETCHER ==========

def get_all_series(config):
    fred_api_key = config['data']['fred_api_key']
    fmp_api_key = config['data']['fmp_api_key']
    start_date = pd.to_datetime(config['data']['start_date'])
    start_date_str = str(start_date.date())

    data = {}

    # --- Fetch all FRED time series at once ---
    fred_data = get_fred_series(fred_api_key, start_date)
    data.update(fred_data)

    # --- Add FMP or other data ---
    data['MOVE Index'] = fetch_fmp_data('^MOVE', fmp_api_key, start_date=start_date_str)
    data['Gold Price'] = fetch_fmp_data('GC=F', fmp_api_key, start_date=start_date_str)
    data['USD Index (DXY)'] = fetch_fmp_data('DX-Y.NYB', fmp_api_key, start_date=start_date_str)
    data['HYG-LQD Spread'] = get_hyg_lqd_spread(fmp_api_key, start_date=start_date_str)

    # --- Example spreads using FRED data already loaded ---
    if '10Y Yield' in data and '2Y Yield' in data:
        data['10Y-2Y Slope'] = data['10Y Yield'] - data['2Y Yield']
    if '10Y Yield' in data and '3M T-Bill' in data:
        data['10Y-3M Slope'] = data['10Y Yield'] - data['3M T-Bill']
    if 'VIX' in data and 'VXV' in data:
        data['VIX-VXV Spread'] = data['VIX'] - data['VXV']

    # --- Combine all into DataFrame, filter to start_date ---
    df_final = pd.concat(data, axis=1)
    df_final = df_final[df_final.index >= start_date].sort_index()

    return df_final
