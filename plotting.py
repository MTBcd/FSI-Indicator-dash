# plotting.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import chart_studio
import chart_studio.plotly as py
import logging
from utils import smooth_transition_regime, regime_from_smooth_weight, classify_adaptive_regime, classify_adaptive_regime_hybrid_fallback, classify_risk_regime_hybrid
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, skew, kurtosis
from numpy import trapz
 
# chart_studio.tools.set_credentials_file(username='Tuler', api_key='EOdkt6iCFZgZvJtTdFc6')


# --- Shared axis styles (taken from PnL chart) ---
AXIS_TITLE_FONT = dict(family="Arial Black", size=16, color="#163A7B")
AXIS_TICK_FONT  = dict(family="Arial Black", size=12, color="#163A7B")

# --- Shared figure geometry (height + margins) ---
FIG_HEIGHT_MAIN   = 550   # main time-series charts (FSI + PnL)
FIG_HEIGHT_SECOND = 550   # secondary charts (cumret, distribution, HHI if you want)

# Tight margins so the plot fills vertically
FIG_MARGIN_MAIN   = dict(l=70, r=200, t=40, b=60)   # FSI charts (legend on right)
FIG_MARGIN_PNL    = dict(l=80, r=60,  t=40, b=70)   # PnL scatter
FIG_MARGIN_CUMRET = dict(l=70, r=200, t=60, b=60)   # cum-ret (extra top for title)


def _compute_date_ticks(index: pd.DatetimeIndex):
    """
    Build x-axis tick positions and labels based on the selected time window:
      - > 3 years  -> yearly ticks: 2019, 2020, ...
      - <= 3 years -> quarterly ticks: Q1 2023, Q2 2023, ...
    """
    idx = pd.to_datetime(index)
    if idx.empty:
        return [], []

    start = idx.min()
    end   = idx.max()

    span_years = (end - start).days / 365.25

    if span_years > 3:
        # Yearly ticks
        years = range(start.year, end.year + 1)
        tickvals = [pd.Timestamp(f"{y}-01-01") for y in years]
        ticktext = [str(y) for y in years]
    else:
        # Quarterly ticks
        quarters = pd.period_range(start=start, end=end, freq="Q")
        tickvals = [p.start_time for p in quarters]
        ticktext = [f"Q{p.quarter} {p.year}" for p in quarters]

    return tickvals, ticktext


def apply_standard_date_axis(fig: go.Figure, index, title_text: str = "<b>Date</b>"):
    """
    Apply the standard x-axis style + date frequency rule to a figure.
    """
    tickvals, ticktext = _compute_date_ticks(pd.to_datetime(index))

    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        title=dict(text=title_text, font=AXIS_TITLE_FONT),
        tickfont=AXIS_TICK_FONT,
        type="date",
        rangeslider=dict(visible=False),
        showgrid=False,
        gridwidth=1.2,
        gridcolor="black",
    )



market_events = {
    # "2018-12-24": "<b>Fed hikes<br>market panic</b>",
    # "2019-08-14": "<b>Yield curve<br>inversion</b>",
    # "2020-02-24": "<b>COVID<br>Crisis</b>",
    # "2022-02-24": "<b>Russia<br>invades Ukraine</b>",
    # "2022-05-11": "<b>Fed hikes<br>to control<br>inflation</b>",
    # "2022-07-28": "<b>US GDP<br>recession fears</b>",
    # "2023-03-10": "<b>SVB<br>collapse</b>",
    # "2024-08-01": "<b>Fed starts<br>cutting rates</b>",
    # "2025-04-15": "<b>Trump tariffs</b>"
}

event_heights = {
    # "2018-12-24": 0.90,
    # "2019-08-14": 0.67,
    # "2020-02-24": 0.98,
    # "2022-02-24": 0.89,
    # "2022-05-11": 0.69,
    # "2022-07-28": 0.78,
    # "2023-03-10": 0.60,
    # "2024-08-01": 0.78,
    # "2025-04-15": 0.68
}

event_heights_pnl = {
    # "2018-12-24": 0.90,
    # "2019-08-14": 0.15,
    # "2020-02-24": 0.94,
    # "2022-02-24": 0.86,
    # "2022-05-11": 0.15,
    # "2022-07-28": 0.03,
    # "2023-03-10": 0.78,
    # "2024-08-01": 0.78,
    # "2025-04-15": 0.71
}

def add_event_annotations(fig, events_dict, event_heights=None):
    x0 = fig.layout.xaxis.range[0] if fig.layout.xaxis.range else None
    x1 = fig.layout.xaxis.range[1] if fig.layout.xaxis.range else None

    for date_str, label in sorted(events_dict.items()):
        x = pd.to_datetime(date_str)
        if x0 and x1 and not (pd.to_datetime(x0) <= x <= pd.to_datetime(x1)):
            continue  # skip out-of-range
        y = event_heights.get(date_str, 1.01) if event_heights else 1.01
        fig.add_annotation(
            x=x,
            y=y,
            xref='x', yref='paper',
            text=label,
            showarrow=False,
            font=dict(size=14, family="Arial"),
            xanchor="center",
            align="center",
            bgcolor="rgba(240,240,240,0.9)",
            bordercolor="grey",
            borderwidth=1,
            borderpad=4,
        )

def add_regime_ribbons(fig, fsi_series, regimes, row=1, col=1, regime_filter=None):
    """Add regime-based colored ribbons to the plot."""
    df = pd.DataFrame({'FSI': fsi_series, 'Regime': regimes})
    df['RegimeShift'] = (df['Regime'] != df['Regime'].shift()).cumsum()
    colors = {
        'Green': 'rgba(0, 200, 0, 0.3)',
        'Yellow': 'rgba(255, 255, 0, 0.3)',
        'Amber': 'rgba(255, 165, 0, 0.3)',
        'Red': 'rgba(255, 0, 0, 0.3)'
    }

    # NEW: filter which regimes to draw
    allowed = set(regime_filter) if regime_filter else set(colors.keys())

    for _, seg in df.groupby('RegimeShift'):
        regime = seg['Regime'].iloc[0]
        if regime not in allowed:
            continue
        fig.add_vrect(
            x0=seg.index[0],
            x1=seg.index[-1] + pd.Timedelta(days=1),
            fillcolor=colors.get(regime, 'rgba(100,100,100,0.1)'),
            opacity=1, layer="below", line_width=0, row=row, col=col
        )



def reindex_to_daily(series_or_df, fill_method="ffill"):
    """Reindex a Series or DataFrame to full daily calendar (including weekends/holidays)."""
    daily_index = pd.date_range(series_or_df.index.min(), series_or_df.index.max(), freq="D")
    return series_or_df.reindex(daily_index).fillna(method=fill_method)

def fix_axis_minus(fig, y_min, y_max, n_ticks=5):
    """Fix the display of minus signs on the y-axis."""
    import numpy as np
    tick_vals = np.linspace(y_min, y_max, n_ticks)
    tick_texts = [f"{v:.2f}".replace("-", "-") for v in tick_vals]  # Ensure using standard hyphen
    fig.update_yaxes(tickvals=tick_vals, ticktext=tick_texts, tickfont=dict(family="Arial", size=12))

def make_tz_naive(dt):
    return dt.tz_localize(None) if getattr(dt, "tzinfo", None) is not None else dt

def _prepare_ribbons(fsi, regimes):
    # Make sure regimes is a Series and align to fsi
    if not isinstance(regimes, pd.Series):
        regimes = pd.Series(regimes, index=fsi.index)
    regimes = regimes.reindex(fsi.index).ffill().bfill()
    fsi_daily = reindex_to_daily(fsi)
    regimes_daily = reindex_to_daily(regimes)
    return fsi_daily, regimes_daily

def _pick_fsi_for_ribbons(contribs_by_group, fsi_for_ribbons):
    if fsi_for_ribbons is not None:
        # Align to the contribs index
        s = pd.to_datetime(contribs_by_group.index)
        f = pd.to_datetime(fsi_for_ribbons.index)
        return fsi_for_ribbons.reindex(contribs_by_group.index).ffill().bfill()
    else:
        return contribs_by_group['FSI']


def plot_group_contributions_with_regime(contribs_by_group, regimes=None, regime_filter=None):
    """Plot variable-level contributions to the FSI with regime ribbons (no proximity subplot)."""
    try:
        contribs_by_group.index = pd.to_datetime(contribs_by_group.index)
        fsi = contribs_by_group['FSI']

        fig = go.Figure()

        fig.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86400000.0], value="%Y-%m-%d"),  # < 1 day
                dict(dtickrange=[86400000.0, 604800000.0], value="%Y-%m-%d"),  # < 1 week
                dict(dtickrange=[604800000.0, "M1"], value="%Y-%m-%d"),  # < 1 month
                dict(dtickrange=["M1", "M12"], value="%b %Y"),           # < 1 year
                dict(dtickrange=["M12", None], value="%Y")               # >= 1 year
            ]
        )

        # Stacked area by variable (everything except FSI)
        for col in [c for c in contribs_by_group.columns if c != 'FSI']:
            fig.add_trace(go.Scatter(
                x=contribs_by_group.index,
                y=contribs_by_group[col],
                stackgroup='one',
                name=col,
                legendgroup=col
            ))

        # FSI line
        fig.add_trace(go.Scattergl(
            x=contribs_by_group.index,
            y=fsi,
            name='FSI (Total)',
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            legendgroup='FSI'
        ))

        # Regime ribbons + events

        fsi_daily, regimes_daily = _prepare_ribbons(fsi, regimes)

        add_regime_ribbons(fig, fsi_daily, regimes=regimes_daily, regime_filter=regime_filter)
        add_event_annotations(fig, market_events, event_heights=event_heights)

        # Year markers
        index_min = make_tz_naive(contribs_by_group.index.min())
        index_max = make_tz_naive(contribs_by_group.index.max())
        year_starts = pd.to_datetime([f"{y}-01-01" for y in sorted(set(contribs_by_group.index.year))])
        year_starts = [d for d in map(make_tz_naive, year_starts) if index_min <= d <= index_max]
        for d in year_starts:
            fig.add_vline(x=d, line_width=1.2, line_color="black", opacity=0.5)


        # fig.update_layout(
        #     template="plotly_white",
        #     showlegend=True,
        #     font=dict(family="Arial", size=13),  # body text
        #     xaxis=dict(
        #         type='date',
        #         showgrid=False,
        #         gridwidth=1.2,
        #         gridcolor='black',
        #         rangeslider=dict(visible=False),
        #     ),
        #     yaxis=dict(
        #         title=dict(text="<b>Contribution to FSI</b>", font=AXIS_TITLE_FONT),
        #         tickfont=AXIS_TICK_FONT,
        #         showgrid=False,
        #         gridwidth=1,
        #         gridcolor='lightgray'
        #     ),
        # )


        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            font=dict(family="Arial", size=13),

            # 🔹 Harmonised geometry
            height=FIG_HEIGHT_MAIN,
            margin=FIG_MARGIN_MAIN,

            xaxis=dict(
                type='date',
                showgrid=False,
                gridwidth=1.2,
                gridcolor='black',
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                title=dict(text="<b>Contribution to FSI</b>", font=AXIS_TITLE_FONT),
                tickfont=AXIS_TICK_FONT,
                showgrid=False,
                gridwidth=1,
                gridcolor='lightgray'
            ),
        )


        y_min = float(np.nanmin(fsi))
        y_max = float(np.nanmax(fsi))
        fix_axis_minus(fig, y_min, y_max)

        fig.update_yaxes(
            tickformat=".2f",
            separatethousands=False,
            exponentformat="none",
            showexponent="none",
        )

        # 🔹 Standardized x-axis (same style + frequency rule as PnL)
        apply_standard_date_axis(fig, contribs_by_group.index, title_text="<b>Date</b>")

        return fig

    except Exception as e:
        logging.error(f"Error plotting variable-level contributions: {e}", exc_info=True)
        return None





    #     fig.update_layout(
    #         template="plotly_white",
    #         showlegend=True,
    #         font=dict(family="Arial", size=13),
    #         xaxis=dict(
    #             title="Date",
    #             rangeslider=dict(visible=False),
    #             type='date',
    #             showgrid=False,
    #             gridwidth=1.2,
    #             gridcolor='black',
    #             tickformat='%Y'
    #         ),
    #         yaxis=dict(
    #             title="Contribution to FSI",
    #             showgrid=False,
    #             gridwidth=1,
    #             gridcolor='lightgray'
    #         ),
    #     )

    #     y_min = float(np.nanmin(fsi))
    #     y_max = float(np.nanmax(fsi))
    #     fix_axis_minus(fig, y_min, y_max)

    #     fig.update_yaxes(
    #         tickformat=".2f",
    #         separatethousands=False,
    #         exponentformat="none",
    #         showexponent="none",
    #         tickfont=dict(family="Arial", size=13)
    #     )

    #     return fig

    # except Exception as e:
    #     logging.error(f"Error plotting variable-level contributions: {e}", exc_info=True)
    #     return None


def plot_grouped_contributions(contribs_by_group, regimes=None, regime_filter=None):
    """Plot grouped contributions to the FSI with regime ribbons (no proximity subplot)."""
    try:
        contribs_by_group.index = pd.to_datetime(contribs_by_group.index)
        fsi = contribs_by_group['FSI']

        fig = go.Figure()

        fig.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86400000.0], value="%Y-%m-%d"),  # < 1 day
                dict(dtickrange=[86400000.0, 604800000.0], value="%Y-%m-%d"),  # < 1 week
                dict(dtickrange=[604800000.0, "M1"], value="%Y-%m-%d"),  # < 1 month
                dict(dtickrange=["M1", "M12"], value="%b %Y"),           # < 1 year
                dict(dtickrange=["M12", None], value="%Y")               # >= 1 year
            ]
        )

        # Stacked area by group (everything except FSI)
        for col in [c for c in contribs_by_group.columns if c != 'FSI']:
            fig.add_trace(go.Scatter(
                x=contribs_by_group.index,
                y=contribs_by_group[col],
                stackgroup='one',
                name=col,
                legendgroup=col
            ))

        # FSI line
        fig.add_trace(go.Scattergl(
            x=contribs_by_group.index,
            y=fsi,
            name='FSI (Total)',
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            legendgroup='FSI'
        ))

        # Regime ribbons + events

        fsi_daily, regimes_daily = _prepare_ribbons(fsi, regimes)

        add_regime_ribbons(fig, fsi_daily, regimes=regimes_daily, regime_filter=regime_filter)
        add_event_annotations(fig, market_events, event_heights=event_heights)

        # Year markers
        index_min = make_tz_naive(contribs_by_group.index.min())
        index_max = make_tz_naive(contribs_by_group.index.max())
        year_starts = pd.to_datetime([f"{y}-01-01" for y in sorted(set(contribs_by_group.index.year))])
        year_starts = [d for d in map(make_tz_naive, year_starts) if index_min <= d <= index_max]
        for d in year_starts:
            fig.add_vline(x=d, line_width=1.2, line_color="black", opacity=0.5)


        # fig.update_layout(
        #     template="plotly_white",
        #     showlegend=True,
        #     xaxis=dict(
        #         type='date',
        #         showgrid=False,
        #         gridwidth=1.2,
        #         gridcolor='black',
        #         rangeslider=dict(visible=False),
        #     ),
        #     yaxis=dict(
        #         title=dict(text="<b>Contribution to FSI</b>", font=AXIS_TITLE_FONT),
        #         tickfont=AXIS_TICK_FONT,
        #         showgrid=False,
        #         gridwidth=1,
        #         gridcolor='lightgray'
        #     ),
        # )


        fig.update_layout(
            template="plotly_white",
            showlegend=True,

            # 🔹 Same thickness as other main charts
            height=FIG_HEIGHT_MAIN,
            margin=FIG_MARGIN_MAIN,

            xaxis=dict(
                type='date',
                showgrid=False,
                gridwidth=1.2,
                gridcolor='black',
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                title=dict(text="<b>Contribution to FSI</b>", font=AXIS_TITLE_FONT),
                tickfont=AXIS_TICK_FONT,
                showgrid=False,
                gridwidth=1,
                gridcolor='lightgray'
            ),
        )



        y_min = float(np.nanmin(fsi))
        y_max = float(np.nanmax(fsi))
        fix_axis_minus(fig, y_min, y_max)

        fig.update_yaxes(
            tickformat=".2f",
            separatethousands=False,
            exponentformat="none",
            showexponent="none",
        )

        # 🔹 Standardized x-axis
        apply_standard_date_axis(fig, contribs_by_group.index, title_text="<b>Date</b>")

        return fig

    except Exception as e:
        logging.error(f"Error plotting grouped contributions: {e}", exc_info=True)
        return None




    #     fig.update_layout(
    #         template="plotly_white",
    #         showlegend=True,
    #         xaxis=dict(
    #             title="Date",
    #             rangeslider=dict(visible=False),
    #             type='date',
    #             showgrid=False,
    #             gridwidth=1.2,
    #             gridcolor='black',
    #             tickformat='%Y'
    #         ),
    #         yaxis=dict(
    #             title="Contribution to FSI",
    #             showgrid=False,
    #             gridwidth=1,
    #             gridcolor='lightgray'
    #         ),
    #     )

    #     y_min = float(np.nanmin(fsi))
    #     y_max = float(np.nanmax(fsi))
    #     fix_axis_minus(fig, y_min, y_max)

    #     fig.update_yaxes(
    #         tickformat=".2f",
    #         separatethousands=False,
    #         exponentformat="none",
    #         showexponent="none",
    #         tickfont=dict(family="Arial", size=13)
    #     )

    #     return fig

    # except Exception as e:
    #     logging.error(f"Error plotting grouped contributions: {e}", exc_info=True)
    #     return None


def plot_hhi_bar(ranking_shares: pd.Series, top_n: int = 15, title_suffix: str = ""):
    """
    Bar chart of top-N contributor shares (summing to 1 over variables used
    to compute HHI). Assumes `ranking_shares` is sorted desc.
    """
    if ranking_shares is None or ranking_shares.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for HHI.")
        return fig

    s = ranking_shares.head(top_n)
    fig = go.Figure(go.Bar(
        x=s.index.tolist(),
        y=(s.values * 100.0),  # percent
        name="Share (%)",
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_white",
        title=f"Top {min(top_n, len(s))} Contributors by Share {title_suffix}",
        xaxis=dict(title="", tickangle=-30),
        yaxis=dict(title="Share (%)", rangemode="tozero"),
        margin=dict(l=50, r=110, t=35, b=40), # (l=40, r=20, t=50, b=100)
        showlegend=False
    )
    return fig


def plot_pnl_with_regime_ribbons(pnl_df, contribs_by_group, fsi_series, regimes=None):
    """PnL scatter with regime ribbons from FSI classification (no proximity trace)."""
    try:
        contribs_by_group.index = pd.to_datetime(contribs_by_group.index)
        fsi_series.index = pd.to_datetime(fsi_series.index)

        # Filter to a consistent start (optional)
        start_chart_date = pd.to_datetime("2019-01-01")
        if 'Date' in pnl_df.columns:
            pnl_df = pnl_df.set_index(pd.to_datetime(pnl_df['Date']))
        pnl_df.index = pd.to_datetime(pnl_df.index)

        pnl_df = pnl_df.loc[pnl_df.index >= start_chart_date]
        contribs_by_group = contribs_by_group.loc[contribs_by_group.index >= start_chart_date]
        fsi_series = fsi_series.loc[fsi_series.index >= start_chart_date]

        # FSI & regimes (take FSI from the passed contributions DF)
        fsi = contribs_by_group['FSI']

        # --- NEW: create the figure FIRST (bug fix) ---
        fig = go.Figure()

        fig.update_xaxes(
            tickformatstops=[
                dict(dtickrange=[None, 86400000.0], value="%Y-%m-%d"),
                dict(dtickrange=[86400000.0, 604800000.0], value="%Y-%m-%d"),
                dict(dtickrange=[604800000.0, "M1"], value="%Y-%m-%d"),
                dict(dtickrange=["M1", "M12"], value="%b %Y"),
                dict(dtickrange=["M12", None], value="%Y")
            ]
        )

        # --- Align PnL strictly to overlapping dates with FSI ---
        # (reindex-to-FSI can produce all-NaN if there’s no overlap)
        common_idx = fsi_series.index.intersection(pnl_df.index)
        if common_idx.empty:
            # No overlap -> show a helpful empty chart
            fig.update_layout(title="PnL dates do not overlap the FSI sample.")
            return fig
        pnl_series = pnl_df.loc[common_idx, 'P/L']

        # --- NEW: limit FSI + regimes to the visible PnL window ---
        pnl_start = pnl_series.index.min()
        pnl_end   = pnl_series.index.max()

        fsi_window = fsi.loc[pnl_start:pnl_end]
        regimes_window = regimes
        if regimes_window is not None:
            if not isinstance(regimes_window, pd.Series):
                regimes_window = pd.Series(regimes_window, index=fsi.index)
            regimes_window = regimes_window.loc[fsi_window.index]

        # Y-axis grid at 3% spacing (guard against empty/NaN)
        if pnl_series.dropna().empty:
            fig.update_layout(title="PnL series is empty after aligning to FSI dates.")
            return fig

        y_min = float(np.nanmin(pnl_series))
        y_max = float(np.nanmax(pnl_series))
        max_abs = max(abs(y_min), abs(y_max), 0.06)
        max_abs = np.ceil(max_abs * 100 / 3) * 3 / 100
        yticks = np.round(np.arange(-max_abs, max_abs + 0.001, 0.03), 2)
        yticktext = [f"{int(v*100)}%" for v in yticks]

        # PnL points
        fig.add_trace(go.Scattergl(
            x=pnl_series.index,
            y=pnl_series.values,
            mode='markers',
            marker=dict(size=5, color='Darkblue'),
            name='PnL'
        ))

        # # Regime ribbons (match FSI chart)
        # fsi_daily, regimes_daily = _prepare_ribbons(fsi, regimes)
        # add_regime_ribbons(fig, fsi_daily, regimes=regimes_daily)

        # Regime ribbons (restricted to PnL window)
        fsi_daily, regimes_daily = _prepare_ribbons(fsi_window, regimes_window)
        add_regime_ribbons(fig, fsi_daily, regimes=regimes_daily)


        # VaR guard rails
        custom_color_dark = '#3096B9'
        fig.add_hline(y=0.03, line_color=custom_color_dark, line_dash="dash", layer="below")
        fig.add_hline(y=-0.03, line_color=custom_color_dark, line_dash="dash", layer="below")

        # Year lines
        index_min = make_tz_naive(pnl_series.index.min())
        index_max = make_tz_naive(pnl_series.index.max())
        year_starts = pd.to_datetime([f"{year}-01-01" for year in sorted(set(pnl_series.index.year))])
        year_starts = [d for d in map(make_tz_naive, year_starts) if index_min <= d <= index_max]
        for d in year_starts:
            fig.add_vline(x=d, line_width=1.2, line_color="black", opacity=0.5)

        # Optional annotations (as you had)
        fig.add_annotation(
            x=pd.to_datetime("2018-08-31"), y=0, xref='x', yref='y',
            text="PRE-<br>AQUAE", showarrow=False,
            font=dict(size=14, color='red'), align="center",
            bgcolor="rgba(255, 255, 255, 0.5)", bordercolor="red", borderwidth=1, borderpad=4,
        )
        fig.add_annotation(
            x=pd.to_datetime("2023-01-01"), y=-0.2, xref='x', yref='paper',
            text="<b>New Risk<br>Controls</b>", showarrow=False,
            font=dict(size=12, color="black"), align="center",
            bordercolor="red", borderwidth=1, borderpad=4, bgcolor="rgba(255, 255, 255, 0.5)"
        )

        neptune_end = pnl_series.index.max()
        # PORTFOLIO arrow
        fig.add_shape(type="line", x0="2019-01-01", x1="2024-02-01", y0=-0.13, y1=-0.13,
                      line=dict(color="darkblue", width=3), xref='x', yref='y', layer="above")
        fig.add_annotation(x="2019-01-01", y=-0.13, xref='x', yref='y',
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                           arrowcolor="darkblue", ax=30, ay=0)
        fig.add_annotation(x="2024-02-01", y=-0.13, xref='x', yref='y',
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                           arrowcolor="darkblue", ax=-30, ay=0)
        fig.add_annotation(x="2021-07-01", y=-0.155, xref='x', yref='y',
                           text="<b>PORTFOLIO</b>", showarrow=False,
                           font=dict(family="Arial Black", size=16, color="darkblue"), align="center")

        # NEPTUNE arrow
        fig.add_shape(type="line", x0="2024-02-01", x1=neptune_end, y0=-0.13, y1=-0.13,
                      line=dict(color="#3096B9", width=3), xref='x', yref='y', layer="above")
        fig.add_annotation(x="2024-02-01", y=-0.13, xref='x', yref='y',
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                           arrowcolor="#3096B9", ax=30, ay=0)
        fig.add_annotation(x=neptune_end, y=-0.13, xref='x', yref='y',
                           showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
                           arrowcolor="#3096B9", ax=-30, ay=0)
        fig.add_annotation(
            x=pd.to_datetime("2024-02-01") + (neptune_end - pd.to_datetime("2024-02-01")) / 2,
            y=-0.155, xref='x', yref='y',
            text="<b>NEPTUNE</b>", showarrow=False,
            font=dict(family="Arial Black", size=16, color="#3096B9"), align="center"
        )

        # fig.update_layout(
        #     template="plotly_white",
        #     showlegend=True,
        #     xaxis=dict(
        #         # title / ticks will be set by apply_standard_date_axis
        #         type='date',
        #         showgrid=False,
        #         gridwidth=1.2,
        #         gridcolor='black',
        #         rangeslider=dict(visible=False),
        #     ),
        #     yaxis=dict(
        #         title=dict(text="<b>PnL (%)</b>", font=AXIS_TITLE_FONT),
        #         tickfont=AXIS_TICK_FONT,
        #         tickvals=yticks,
        #         ticktext=yticktext,
        #         tickmode="array",
        #         showgrid=False,
        #         range=[yticks[0], yticks[-1]],
        #         fixedrange=True,
        #     )
        # )


        fig.update_layout(
            template="plotly_white",
            showlegend=True,

            # 🔹 Same thickness as FSI charts
            height=FIG_HEIGHT_MAIN,
            margin=FIG_MARGIN_PNL,

            xaxis=dict(
                # title / ticks overridden by apply_standard_date_axis
                type='date',
                showgrid=False,
                gridwidth=1.2,
                gridcolor='black',
                rangeslider=dict(visible=False),
            ),
            yaxis=dict(
                title=dict(text="<b>PnL (%)</b>", font=AXIS_TITLE_FONT),
                tickfont=AXIS_TICK_FONT,
                tickvals=yticks,
                ticktext=yticktext,
                tickmode="array",
                showgrid=False,
                range=[yticks[0], yticks[-1]],
                fixedrange=True,
            )
        )



        # Apply standardized date axis using PnL dates
        apply_standard_date_axis(fig, pnl_series.index, title_text="<b>Date</b>")

        # # Keep x-range consistent with FSI window
        # fig.update_xaxes(range=[fsi.index.min(), fsi.index.max()])
        # return fig

        # 🔹 Keep x-range consistent with the *selected PnL* window
        fig.update_xaxes(range=[pnl_start, pnl_end])
        return fig

    except Exception as e:
        logging.error(f"Error plotting PnL with regime ribbons: {e}", exc_info=True)
        # return an empty, informative figure instead of None
        fig = go.Figure()
        fig.update_layout(title="Could not render PnL chart. Check logs for details.")
        return fig






    #     fig.update_layout(
    #         template="plotly_white",
    #         showlegend=True,
    #         xaxis=dict(
    #             title=dict(text="<b>Date</b>", font=dict(family="Arial Black", size=16, color="#163A7B")),
    #             tickfont=dict(family="Arial Black", size=12, color="#163A7B"),
    #             rangeslider=dict(visible=False),
    #             type='date',
    #             showgrid=False,
    #             gridwidth=1.2,
    #             gridcolor='black',
    #             tickformatstops=[
    #                 dict(dtickrange=[None, 1000 * 60 * 60 * 24 * 366], value="%Y"),
    #                 dict(dtickrange=[1000 * 60 * 60 * 24 * 28, 1000 * 60 * 60 * 24 * 366], value="%b-%Y"),
    #             ]
    #         ),
    #         yaxis=dict(
    #             title=dict(text="<b>PnL (%)</b>", font=dict(family="Arial Black", size=16, color="#163A7B")),
    #             tickfont=dict(family="Arial Black", size=12, color="#163A7B"),
    #             tickvals=yticks,
    #             ticktext=yticktext,
    #             tickmode="array",
    #             showgrid=False,
    #             range=[yticks[0], yticks[-1]],
    #             fixedrange=True,
    #         )
    #     )

    #     # Keep x-range consistent with FSI window
    #     fig.update_xaxes(range=[fsi.index.min(), fsi.index.max()])
    #     return fig

    # except Exception as e:
    #     logging.error(f"Error plotting PnL with regime ribbons: {e}", exc_info=True)
    #     # return an empty, informative figure instead of None
    #     fig = go.Figure()
    #     fig.update_layout(title="Could not render PnL chart. Check logs for details.")
    #     return fig


def save_fsi_charts_to_html(fig1, fig2, fig3=None, filename="fsi_combined_report.html"):
    with open(filename, "w") as f:
        f.write("<html><head><title>FSI Report</title></head><body>\n")
        f.write("<h1>FSI Variable-Level Contributions</h1>\n")
        f.write(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write("<hr><h1>FSI Group-Level Contributions</h1>\n")
        f.write(fig2.to_html(full_html=False, include_plotlyjs=False))
        if fig3:
            f.write("<hr><h1>Realized NEPTUNE PnL with Regimes</h1>\n")
            f.write(fig3.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")
    print(f"✅ Combined HTML saved to: {filename}")


def plot_distribution_plotly(pnl_values, period_title, pnl_range=None):
    """Plotly version for Dash. Returns a go.Figure."""
    # Optionally filter by range
    if pnl_range:
        pnl_values = pnl_values[(pnl_values >= pnl_range[0]) & (pnl_values <= pnl_range[1])]
    pnl_values = pnl_values.dropna()

    # Stats
    pnl_skewness = skew(pnl_values)
    pnl_kurtosis = kurtosis(pnl_values)
    kde = gaussian_kde(pnl_values, bw_method='scott')
    x_vals = np.linspace(min(pnl_values), max(pnl_values), 500)
    y_vals = kde(x_vals)

    # Area
    pos_mask = x_vals >= 0
    neg_mask = x_vals < 0
    area_positive = trapz(y_vals[pos_mask], x_vals[pos_mask])
    area_negative = trapz(y_vals[neg_mask], x_vals[neg_mask])

    # Histogram
    hist = np.histogram(pnl_values, bins=50, density=True)
    hist_x = (hist[1][1:] + hist[1][:-1]) / 2
    hist_y = hist[0]

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Bar(
        x=hist_x, y=hist_y,
        marker=dict(color='#3096B9'),
        opacity=0.75,
        name='PnL Histogram',
        hoverinfo='skip'
    ))

    # KDE
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='lines',
        line=dict(color='#002060', width=3),
        name='Smoothed PnL'
    ))

    # Metrics (as annotation)
    metrics = (
        f"<b>Skewness</b>: {pnl_skewness:.2f}<br>"
        f"<b>Kurtosis</b>: {pnl_kurtosis:.2f}<br>"
        f"<b>Area (Pos)</b>: {area_positive:.2f}<br>"
        f"<b>Area (Neg)</b>: {area_negative:.2f}"
    )

    fig.add_annotation(
        x=1.01, y=1.04, xref="paper", yref="paper",
        text=metrics, showarrow=False,
        align='right', font=dict(size=11, family="Arial", color='#002060')
    )

    # Subtitle
    total_obs = len(pnl_values)
    subtitle = f'{total_obs} PnL observations<br>from {period_title}'
    fig.add_annotation(
        x=0, y=1.04, xref="paper", yref="paper",
        text=f"<b>{subtitle}</b>", showarrow=False,
        align='left', font=dict(size=11, family="Arial", color='#002060')
    )

    fig.update_layout(
        xaxis_title="PnL Values",
        yaxis_title="Density",
        margin=dict(l=50, r=160, t=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.3,
            xanchor="center", x=0.5,
            font=dict(size=11, family="Arial")
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(tickfont=dict(family="Arial", color='#002060', size=12)),
        yaxis=dict(tickfont=dict(family="Arial", color='#3096B9', size=12)),
    )

    return fig


def make_cumret_figure(
    neptune_returns: pd.Series,
    benchmark_returns: pd.DataFrame,
    start_date=None,
    end_date=None,
    fsi_series: pd.Series | None = None,
    regimes: pd.Series | None = None,
) -> go.Figure:
    """
    Build a cumulative return chart for NEPTUNE vs benchmarks.

    NEPTUNE + benchmarks are rebased to 0% at start_date.
    """

    COLOR_MAP = {
        "NEPTUNE":  "#0077b6",  # darkest
        "ACWI":     "#1fa4ff",
        "SPX":      "#7fc8ff",
        "SPXETWR":  "#cfe9ff",  # lightest
    }
    DEFAULT_COLOR = "#555555"

    if neptune_returns is None or neptune_returns.empty:
        return go.Figure()

    if benchmark_returns is None or benchmark_returns.empty:
        bench = pd.DataFrame(index=neptune_returns.index)
    else:
        bench = benchmark_returns.copy()

    # Combine NEPTUNE + benchmarks
    all_data = pd.concat(
        [neptune_returns.rename("NEPTUNE"), bench],
        axis=1
    ).sort_index().ffill().dropna(how="all")

    if all_data.empty:
        return go.Figure()

    # Apply date filters
    if start_date is not None:
        all_data = all_data[all_data.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        all_data = all_data[all_data.index <= pd.to_datetime(end_date)]

    if all_data.empty:
        return go.Figure()

    # Daily returns → cumulative returns, rebased at start
    cum = (1.0 + all_data).cumprod() - 1.0
    cum_pct = cum * 100.0  # in percent

    fig = go.Figure()
    for col in cum_pct.columns:
        fig.add_trace(
            go.Scatter(
                x=cum_pct.index,
                y=cum_pct[col],
                mode="lines",
                name=col,
                line=dict(
                    width=2,
                    color=COLOR_MAP.get(col, DEFAULT_COLOR),
                ),
            )
        )

    # 🔹 Add FSI regime ribbons as background (like PnL chart)
    try:
        if fsi_series is not None and regimes is not None:
            # Make sure they're Series with DateTimeIndex
            fsi_series = pd.Series(fsi_series)
            fsi_series.index = pd.to_datetime(fsi_series.index)

            if not isinstance(regimes, pd.Series):
                regimes = pd.Series(regimes, index=fsi_series.index)
            else:
                regimes.index = pd.to_datetime(regimes.index)

            # Restrict FSI/regimes to the cumret visible window
            idx_min = cum_pct.index.min()
            idx_max = cum_pct.index.max()
            fsi_window = fsi_series.loc[idx_min:idx_max]
            regimes_window = regimes.reindex(fsi_window.index).ffill().bfill()

            if not fsi_window.empty:
                fsi_daily, regimes_daily = _prepare_ribbons(fsi_window, regimes_window)
                add_regime_ribbons(fig, fsi_daily, regimes=regimes_daily)
    except Exception as e:
        logging.error(f"Error adding regime ribbons to cumret chart: {e}", exc_info=True)


    # fig.update_layout(
    #     title="Cumulative Returns (rebased to 0% at selected start date)",
    #     xaxis_title=None,  # we set via apply_standard_date_axis
    #     yaxis_title=None,
    #     hovermode="x unified",
    #     legend=dict(
    #         orientation="v",
    #         yanchor="top",
    #         y=1.0,
    #         xanchor="left",
    #         x=1.02,
    #     ),
    #     margin=dict(l=50, r=160, t=20, b=40),
    # )
    # fig.update_xaxes(showgrid=False, zeroline=False)
    # fig.update_yaxes(
    #     title=dict(text="<b>Cumulative Return (%)</b>", font=AXIS_TITLE_FONT),
    #     tickfont=AXIS_TICK_FONT,
    #     showgrid=False,
    #     zeroline=False,
    # )


    fig.update_layout(
        title="Cumulative Returns (rebased to 0% at selected start date)",
        xaxis_title=None,  # handled by apply_standard_date_axis
        yaxis_title=None,
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
        ),

        # 🔹 Slightly thinner but still harmonised
        height=FIG_HEIGHT_SECOND,
        margin=FIG_MARGIN_CUMRET,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(
        title=dict(text="<b>Cumulative Return (%)</b>", font=AXIS_TITLE_FONT),
        tickfont=AXIS_TICK_FONT,
        showgrid=False,
        zeroline=False,
    )



    # 🔹 Standardized x-axis using cumret index
    apply_standard_date_axis(fig, cum_pct.index, title_text="<b>Date</b>")

    return fig




    # fig.update_layout(
    #     title="Cumulative Returns (rebased to 0% at selected start date)",
    #     xaxis_title="Date",
    #     yaxis_title="Cumulative Return (%)",
    #     hovermode="x unified",
    #     legend=dict(
    #         orientation="v",   # vertical legend
    #         yanchor="top",
    #         y=1.0,             # top
    #         xanchor="left",
    #         x=1.02,            # just to the right of the plot area
    #     ),
    #     margin=dict(l=50, r=160, t=20, b=40),
    # )
    # fig.update_xaxes(showgrid=False, zeroline=False)
    # fig.update_yaxes(showgrid=False, zeroline=False)

    # return fig
