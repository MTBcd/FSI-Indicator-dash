# plotting.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import chart_studio
import chart_studio.plotly as py
import logging
from utils import smooth_transition_regime, regime_from_smooth_weight

# chart_studio.tools.set_credentials_file(username='Tuler', api_key='EOdkt6iCFZgZvJtTdFc6')

market_events = {
    "2018-12-24": "<b>Fed hikes<br>market panic</b>",
    "2019-08-14": "<b>Yield curve<br>inversion</b>",
    "2020-02-24": "<b>COVID<br>Crisis</b>",
    "2022-02-24": "<b>Russia<br>invades Ukraine</b>",
    "2022-05-11": "<b>Fed hikes<br>to control<br>inflation</b>",
    "2022-07-28": "<b>US GDP<br>recession fears</b>",
    "2023-03-10": "<b>SVB<br>collapse</b>",
    "2024-08-01": "<b>Fed starts<br>cutting rates</b>",
    "2025-04-15": "<b>Trump tariffs</b>"
}

event_heights = {
    "2018-12-24": 0.90,
    "2019-08-14": 0.67,
    "2020-02-24": 0.98,
    "2022-02-24": 0.89,
    "2022-05-11": 0.69,
    "2022-07-28": 0.78,
    "2023-03-10": 0.60,
    "2024-08-01": 0.78,
    "2025-04-15": 0.68
}

event_heights_pnl = {
    "2018-12-24": 0.90,
    "2019-08-14": 0.15,
    "2020-02-24": 0.94,
    "2022-02-24": 0.86,
    "2022-05-11": 0.15,
    "2022-07-28": 0.03,
    "2023-03-10": 0.78,
    "2024-08-01": 0.78,
    "2025-04-15": 0.71
}

def add_event_annotations(fig, events_dict, event_heights=None):
    """Add annotations for market events to the plot."""
    for date_str, label in sorted(events_dict.items()):
        x = pd.to_datetime(date_str)
        y = event_heights.get(date_str, 1.01) if event_heights else 1.01

        fig.add_annotation(
            x=x,
            y=y,
            xref='x',
            yref='paper',
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

def add_regime_ribbons(fig, fsi_series, regimes, row=1, col=1):
    """Add regime-based colored ribbons to the plot."""
    df = pd.DataFrame({'FSI': fsi_series, 'Regime': regimes})
    df['RegimeShift'] = (df['Regime'] != df['Regime'].shift()).cumsum()
    colors = {
        'Green': 'rgba(0, 200, 0, 0.3)',
        'Yellow': 'rgba(255, 255, 0, 0.3)',
        'Amber': 'rgba(255, 165, 0, 0.3)',
        'Red': 'rgba(255, 0, 0, 0.3)'
    }
    for _, segment in df.groupby('RegimeShift'):
        regime = segment['Regime'].iloc[0]
        fig.add_vrect(
            x0=segment.index[0], x1=segment.index[-1],
            fillcolor=colors.get(regime, 'rgba(100,100,100,0.1)'),
            opacity=1, layer="below",
            line_width=0,
            row=row, col=col
        )


def fix_axis_minus(fig, y_min, y_max, n_ticks=5):
    """Fix the display of minus signs on the y-axis."""
    import numpy as np
    tick_vals = np.linspace(y_min, y_max, n_ticks)
    tick_texts = [f"{v:.2f}".replace("-", "-") for v in tick_vals]  # Ensure using standard hyphen
    fig.update_yaxes(tickvals=tick_vals, ticktext=tick_texts, tickfont=dict(family="Arial", size=12))


def plot_group_contributions_with_regime(contribs_by_group):
    """Plot group-level contributions to the FSI with regime highlighting."""
    try:
        contribs_by_group.index = pd.to_datetime(contribs_by_group.index)
        fsi = contribs_by_group['FSI']
        smooth_weight = smooth_transition_regime(fsi, gamma=2.5, c=0.5)
        regimes = regime_from_smooth_weight(smooth_weight)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
            # subplot_titles=["FSI Group-Level Contributions", "Transition Proximity"]
        )

        # === Top Plot: Stacked Area Contributions ===
        for col in [c for c in contribs_by_group.columns if c != 'FSI']:
            fig.add_trace(go.Scatter(
                x=contribs_by_group.index,
                y=contribs_by_group[col],
                stackgroup='one',
                name=col,
                legendgroup=col
            ), row=1, col=1)

        # Add FSI line
        fig.add_trace(go.Scatter(
            x=contribs_by_group.index,
            y=fsi,
            name='FSI (Total)',
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            legendgroup='FSI'
        ), row=1, col=1)

        # === Bottom Plot: Transition Proximity ===
        fig.add_trace(go.Scatter(
            x=contribs_by_group.index,
            y=smooth_weight,
            name='Transition Proximity',
            mode='lines',
            line=dict(color='purple', width=2),
            legendgroup='Proximity'
        ), row=2, col=1)

        # Add regime ribbons to top chart only
        add_regime_ribbons(fig, fsi, regimes=regimes, row=1, col=1)
        add_event_annotations(fig, market_events, event_heights=event_heights)

        # Vertical lines and year labels for every Jan 1st
        year_starts = pd.to_datetime([f"{year}-01-01" for year in sorted(set(contribs_by_group.index.year))])
        year_starts = [d for d in year_starts if d >= contribs_by_group.index.min() and d <= contribs_by_group.index.max()]

        for d in year_starts:
            fig.add_vline(
                x=d,
                line_width=1.2,
                line_color="black",
                opacity=0.5,
                row="all"
            )
            fig.add_annotation(
                x=d, y=0.61,
                xref='x', yref='paper',
                text=str(d.year),
                showarrow=False,
                font=dict(size=14, color='black', family='Arial'),
                xanchor="center",
                align="center",
                opacity=0.6,
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="black",
                borderwidth=0.5,
                borderpad=2,
            )

        fig.update_layout(
            height=750,
            # title="FSI Group-Level Contributions with Transition Proximity",
            template="plotly_white",
            showlegend=True,
            font=dict(family="Arial", size=13),
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=False),
                type='date',
                showgrid=True,
                gridwidth=1.2,
                gridcolor='black',
                tickformat='%Y'
            ),
            yaxis=dict(
                title="Contribution to FSI",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                # title="Transition Proximity",
                range=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
        )

        y_min = float(np.nanmin(fsi))
        y_max = float(np.nanmax(fsi))
        fix_axis_minus(fig, y_min, y_max)

        # Standardize all y-axis labels (avoid unicode minus)
        fig.update_yaxes(
            tickformat=".2f",
            separatethousands=False,
            exponentformat="none",
            showexponent="none",
            tickfont=dict(family="Arial", size=13)
        )

        return fig
    except Exception as e:
        logging.error(f"Error plotting group contributions: {e}", exc_info=True)
        return None

def plot_grouped_contributions(contribs_by_group):
    """Plot grouped contributions to the FSI."""
    try:
        contribs_by_group.index = pd.to_datetime(contribs_by_group.index)
        fsi = contribs_by_group['FSI']
        smooth_weight = smooth_transition_regime(fsi, gamma=2.5, c=0.5)
        regimes = regime_from_smooth_weight(smooth_weight)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
            # subplot_titles=["FSI Group-Level Contributions", "Transition Proximity"]
        )

        # === Top Plot: Stacked Contributions ===
        for col in [c for c in contribs_by_group.columns if c != 'FSI']:
            fig.add_trace(go.Scatter(
                x=contribs_by_group.index,
                y=contribs_by_group[col],
                stackgroup='one',
                name=col,
                legendgroup=col
            ), row=1, col=1)

        # FSI Line
        fig.add_trace(go.Scatter(
            x=contribs_by_group.index,
            y=fsi,
            name='FSI (Total)',
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            legendgroup='FSI'
        ), row=1, col=1)

        # === Bottom Plot: Transition Proximity ===
        fig.add_trace(go.Scatter(
            x=contribs_by_group.index,
            y=smooth_weight,
            name='Transition Proximity',
            mode='lines',
            line=dict(color='purple', width=2),
            legendgroup='Proximity'
        ), row=2, col=1)

        # === Add regime ribbons only to top chart ===
        add_regime_ribbons(fig, fsi, regimes=regimes, row=1, col=1)
        add_event_annotations(fig, market_events, event_heights=event_heights)

        # Vertical lines for every Jan 1st (no event lines)
        year_starts = pd.to_datetime([f"{year}-01-01" for year in sorted(set(contribs_by_group.index.year))])
        year_starts = [d for d in year_starts if d >= contribs_by_group.index.min() and d <= contribs_by_group.index.max()]

        for d in year_starts:
            fig.add_vline(
                x=d,
                line_width=1.2,
                line_color="black",
                opacity=0.5,
                row="all"
            )
            fig.add_annotation(
                x=d, y=0.61,
                xref='x', yref='paper',
                text=str(d.year),
                showarrow=False,
                font=dict(size=14, color='black', family='Arial'),
                xanchor="center",
                align="center",
                opacity=0.6,
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="black",
                borderwidth=0.5,
                borderpad=2,
            )

        fig.update_layout(
            height=750,
            # title="FSI Group-Level Contributions with Transition Proximity",
            template="plotly_white",
            showlegend=True,
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=False),
                type='date',
                showgrid=True,
                gridwidth=1.2,
                gridcolor='black',
                tickformat='%Y'
            ),
            yaxis=dict(
                title="Contribution to FSI",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis2=dict(
                # title="Transition Proximity",
                range=[0, 1],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
        )

        y_min = float(np.nanmin(fsi))
        y_max = float(np.nanmax(fsi))
        fix_axis_minus(fig, y_min, y_max)

        # Standardize all y-axis labels (avoid unicode minus)
        fig.update_yaxes(
            tickformat=".2f",
            separatethousands=False,
            exponentformat="none",
            showexponent="none",
            tickfont=dict(family="Arial", size=13)
        )

        return fig
    except Exception as e:
        logging.error(f"Error plotting grouped contributions: {e}", exc_info=True)
        return None



def plot_pnl_with_regime_ribbons(pnl_df, contribs_by_group, fsi_series):
    """Plot PnL scatter with *identical* regime background as FSI group chart."""
    try:
        contribs_by_group.index = pd.to_datetime(contribs_by_group.index)
        fsi = contribs_by_group['FSI']
        smooth_weight = smooth_transition_regime(fsi, gamma=2.5, c=0.5)
        regimes = regime_from_smooth_weight(smooth_weight)

        # Pull / align the PnL series
        if 'Date' in pnl_df.columns:  # allow either column or index
            pnl_df = pnl_df.set_index(pd.to_datetime(pnl_df['Date']))
        pnl_df.index = pd.to_datetime(pnl_df.index)
        pnl_series = pnl_df['P/L'].reindex(fsi_series.index)

        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True
        )

        # Add PnL scatter
        fig.add_trace(
            go.Scatter(
                x=pnl_series.index,
                y=pnl_series.values,
                mode='markers',
                marker=dict(size=5, color='Darkblue'),
                name='PnL'
            ),
            row=1, col=1
        )

        # Add regime ribbons
        add_regime_ribbons(fig, fsi, regimes=regimes, row=1, col=1)
        add_event_annotations(fig, market_events, event_heights=event_heights_pnl)

        # Vertical lines for every Jan 1st
        year_starts = pd.to_datetime([f"{year}-01-01" for year in sorted(set(pnl_series.index.year))])
        year_starts = [d for d in year_starts if d >= pnl_series.index.min() and d <= pnl_series.index.max()]

        for d in year_starts:
            fig.add_vline(
                x=d,
                line_width=1.2,
                line_color="black",
                opacity=0.5,
                row="all"
            )
            fig.add_annotation(
                x=d, y=0.91,
                xref='x', yref='paper',
                text=str(d.year),
                showarrow=False,
                font=dict(size=14, color='black', family='Arial'),
                xanchor="center",
                align="center",
                opacity=0.6,
                bgcolor="rgba(255,255,255,0.1)",
                bordercolor="black",
                borderwidth=0.5,
                borderpad=2,
            )

        # Add "NO DATA" annotation before 2019
        fig.add_annotation(
            x=pd.to_datetime("2018-08-31"),  # Position the annotation at the end of 2018
            y=0,  # Position it on the zero axis
            xref='x',
            yref='y',
            text="PRE-<br>AQUAE",
            showarrow=False,
            font=dict(size=16, color='red'),  # Red color for the text
            align="center",
            bgcolor="rgba(255, 255, 255, 0.5)",  # Optional background color
            bordercolor="red",
            borderwidth=1,
            borderpad=4,
        )

        fig.add_annotation(
            x=pd.to_datetime("2023-01-01"),  # Position the annotation at the end of 2018
            y=-0.18,  # Adjust this value to position it below the chart
            xref='x',
            yref='paper',  # Use 'paper' to position relative to the entire plot area
            text="New Risk<br>Control<br>Implemented",
            showarrow=False,
            font=dict(size=12, color='#3096B9'),  # Color for the text
            align="center",
            bordercolor="red",
            borderwidth=1,
            borderpad=4,
            bgcolor="rgba(255, 255, 255, 0.5)"  # Optional background color for better visibility
        )

        # Add target VaR lines
        x_start = pnl_series.index.min()
        x_end = pnl_series.index.max()
        custom_color_dark = '#3096B9'  # Define your custom color

        fig.add_hline(y=0.03, line_color=custom_color_dark, line_dash="dash", annotation_text="3%", annotation_position="top right")
        fig.add_hline(y=-0.03, line_color=custom_color_dark, line_dash="dash", annotation_text="-3%", annotation_position="bottom right")

        fig.update_layout(
            height=600,
            # title="PnL Scatter Plot with FSI Regimes",
            template="plotly_white",
            showlegend=True,
            xaxis=dict(
                title="Date",
                rangeslider=dict(visible=False),
                type='date',
                showgrid=True,
                gridwidth=1.2,
                gridcolor='black',
                tickformat='%Y'
            ),
            yaxis=dict(
                title="PnL",
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            )
        )

        y_min = float(np.nanmin(pnl_series))
        y_max = float(np.nanmax(pnl_series))
        fix_axis_minus(fig, y_min, y_max)

        # Standardize all y-axis labels (avoid unicode minus)
        fig.update_yaxes(
            tickformat=".2f",
            separatethousands=False,
            exponentformat="none",
            showexponent="none",
            tickfont=dict(family="Arial", size=13)
        )

        return fig
    except Exception as e:
        logging.error(f"Error plotting PnL with regime ribbons: {e}", exc_info=True)
        return None



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
