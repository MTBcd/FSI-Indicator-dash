# data_fetching.py

import pandas as pd
import numpy as np
import requests
import logging
from fredapi import Fred
import configparser
from datetime import datetime

# ======== Config Loader ==========

config = configparser.ConfigParser()
config.read('config.ini')

Start_Date = pd.to_datetime(config['data']['start_date'])
Start_date = config['data']['start_date']
today_date = pd.to_datetime(datetime.today())

# ======== FMP PRICE DATA ==========

def get_nyfed_rates_from_excel(start_date=Start_Date, end_date=None):
    import pandas as pd
    import logging
    from datetime import datetime

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    url = (
        f"https://markets.newyorkfed.org/read?startDt={start_date}&endDt={end_date}"
        "&eventCodes=500&productCode=50&sort=postDt:-1,eventCode:1&format=xlsx"
    )

    try:
        df = pd.read_excel(url)
        df['Effective Date'] = pd.to_datetime(df['Effective Date'])

        # Pivot for rates (EFFR will be one of the Rate Types)
        rates = df.pivot(index='Effective Date', columns='Rate Type', values='Rate (%)')

        # Keep only EFFR column
        if 'EFFR' not in rates.columns:
            rates['EFFR'] = float("nan")

        rates = rates[['EFFR']].sort_index()
        return rates

    except Exception as e:
        logging.error(f"Failed to fetch NY Fed Excel rates: {e}")
        return pd.DataFrame()


def get_treasury_yield_series(maturity='year2', api_key=None, start_date=Start_Date):
    """
    Fetches the specified Treasury yield series (e.g., 'year2', 'year10') from FMP.
    Args:
        maturity: str, one of ['month1', 'month2', ..., 'year1', 'year2', ..., 'year30']
        api_key: str, your FMP API key
        start_date: str, earliest date (YYYY-MM-DD)
    Returns:
        pd.Series with Date index and yield values
    """
    url = f"https://financialmodelingprep.com/stable/treasury-rates?from={start_date}&to={today_date}&apikey={api_key}"   # https://financialmodelingprep.com/stable/treasury-rates?from={Start_Date}&to={today_date}&apikey={api_key}
    try:
        resp = requests.get(url)
        data = resp.json()
        if not isinstance(data, list) or not data:
            logging.warning(f"No treasury yield data found for {maturity} at v4 endpoint.")
            return pd.Series(dtype=float, name=f"{maturity} Treasury Yield")
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= pd.to_datetime(start_date)]
        if maturity not in df.columns:
            logging.warning(f"Maturity {maturity} not found in data columns: {df.columns.tolist()}")
            return pd.Series(dtype=float, name=f"{maturity} Treasury Yield")
        # Convert string values to float
        df[maturity] = pd.to_numeric(df[maturity], errors='coerce')
        df = df.set_index('date').sort_index()
        s = df[maturity].dropna()
        s.name = f"{maturity.upper()} FMP V4"
        return s
    except Exception as e:
        logging.error(f"Error fetching FMP v4 treasury yield data for {maturity}: {e}")
        return pd.Series(dtype=float, name=f"{maturity} Treasury Yield")


def get_price_series(ticker, api_key, start_date=Start_Date):
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

def fetch_fmp_data(ticker, api_key, start_date=Start_Date):
    return get_price_series(ticker, api_key, start_date)

def get_hyg_lqd_spread(api_key, start_date=Start_Date):
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
            # 'USD Overnight Rate': 'OBFR',
            # '2Y Yield': 'DGS2',
            # 'FRED RRP': 'RRPONTSYD',
            'US Corp OAS': 'BAMLC0A0CM',
            'US HY OAS': 'BAMLH0A0HYM2',
            'US BBB OAS': 'BAMLC0A4CBBBEY'
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
    start_date = Start_Date
    start_date_str = str(start_date.date())

    data = {}

    # --- Fetch all FRED time series at once ---
    fred_data = get_fred_series(fred_api_key, start_date)
    data.update(fred_data)

    # --- Add FMP or other data ---
    data['MOVE Index'] = fetch_fmp_data('^MOVE', fmp_api_key, start_date=start_date_str)
    data['Gold Price'] = fetch_fmp_data('GC=F', fmp_api_key, start_date=start_date_str)
    data['VIX'] = fetch_fmp_data('^VIX', fmp_api_key, start_date=start_date_str)
    data['VIX3M'] = fetch_fmp_data('^VIX3M', fmp_api_key, start_date=start_date_str)
    # data['10Y Yield'] = fetch_fmp_data('^TNX', fmp_api_key, start_date=start_date_str)
    data['3M T-Bill'] = fetch_fmp_data('^IRX', fmp_api_key, start_date=start_date_str)
    data['USDJPY'] = fetch_fmp_data('USDJPY', fmp_api_key, start_date=start_date_str)
    # data['federalFunds'] = fetch_fmp_data('federalFunds', fmp_api_key, start_date=start_date_str)
    data['10Y Yield'] = get_treasury_yield_series('year10', fmp_api_key, start_date=start_date_str)
    data['2Y Yield'] = get_treasury_yield_series('year2', fmp_api_key, start_date=start_date_str)

    # --- NY Fed Official Rates from Excel (preferred over CSV API) ---
    nyfed_rates = get_nyfed_rates_from_excel(start_date=start_date_str)
    if not nyfed_rates.empty:
        for col in ['EFFR']:    #'OBFR', "EFFR_VOLUME"
            data[col] = nyfed_rates[col]

    # --- Example spreads using FRED data already loaded ---
    if '10Y Yield' in data and '2Y Yield' in data:
        data['10Y-2Y Slope'] = data['10Y Yield'] - data['2Y Yield']
    if '10Y Yield' in data and '3M T-Bill' in data:
        data['10Y-3M Slope'] = data['10Y Yield'] - data['3M T-Bill']
    if 'VIX' in data and 'VIX3M' in data:
        data['VIX-VIX3M Spread'] = data['VIX'] - data['VIX3M']

    # --- Combine all into DataFrame, filter to start_date ---
    df_final = pd.concat(data, axis=1)
    df_final = df_final[df_final.index >= start_date].sort_index()

    return df_final
