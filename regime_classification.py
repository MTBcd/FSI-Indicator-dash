import pandas as pd
from scipy.stats import rankdata

def classify_regimes(fsi_series, vol_window=20, spike_quantile=0.9):
    percentiles = rankdata(fsi_series) / len(fsi_series)
    vol = fsi_series.rolling(vol_window).std()
    spikes = vol.diff() > vol.diff().quantile(spike_quantile)

    def regime(pct, spike):
        if spike:
            return 'Red' if pct > 0.75 else 'Amber' if pct > 0.35 else 'Yellow'
        return 'Green' if pct <= 0.35 else 'Yellow' if pct <= 0.75 else 'Amber'

    return pd.Series([
        regime(percentiles[i], spikes.iloc[i]) for i in range(len(fsi_series))
    ], index=fsi_series.index)
