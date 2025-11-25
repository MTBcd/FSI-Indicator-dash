# app.py

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import base64
import io
import diskcache
import time
import numpy as np
import logging
  
# --- Your framework imports ---
from main import load_configuration, merge_data
from fsi_estimation import compute_timevarying_contributions, estimate_fsi_expanding_with_als
from plotting import (
    plot_group_contributions_with_regime,
    plot_grouped_contributions,
    plot_pnl_with_regime_ribbons,
    plot_distribution_plotly, plot_hhi_bar, make_cumret_figure
)
from data_fetching import get_benchmark_returns
from utils import (
    aggregate_contributions_by_group, classify_regime_fsi_improved,
    get_current_regime, run_hmm, predict_regime_probability, compute_transition_matrix, build_dynamic_group_map, orient_fsi_and_omega,
    classify_risk_regime_hybrid, average_time_in_regime, classify_adaptive_regime_hybrid_fallback,
    compute_hhi_ranking
)

# --- Consistent Regime Colors ---
REGIME_COLORS = {
    "Green": "#27ae60",
    "Yellow": "#f7ca18",
    "Amber": "#f39c12",
    "Red": "#e74c3c"
}

cache = diskcache.Cache("./cache-directory")

app = dash.Dash(__name__)
server = app.server

# --- Benchmark returns for cumulative chart (ACWI, SPX, SPXETWR) ---
try:
    # Already a DateTimeIndex named "Date"
    benchmark_returns = get_benchmark_returns().sort_index()

    # Rename human-readable FMP names to short labels used in the legend
    rename_map = {
        "MSCI ACWI": "ACWI",
        "S&P 500": "SPX",
        "S&P 500 EW": "SPXETWR",
    }
    benchmark_returns = benchmark_returns.rename(columns=rename_map)

    # (They are already daily simple returns in decimal form; no rescaling needed)

    logging.info(f"Loaded benchmark returns with columns: {list(benchmark_returns.columns)}")

except Exception as e:
    logging.warning(f"Could not load benchmark returns: {e}")
    benchmark_returns = pd.DataFrame()

def regime_color_text(regime):
    # Returns an html span with correct color and bold text for regime
    return html.Span(
        regime,
        style={
            'color': REGIME_COLORS.get(regime, 'black'),
            'fontWeight': 'bold'
        }
    )

def info_icon(text):
    # Simple tooltip icon
    return html.Span(
        " ⓘ", title=text,
        style={"cursor": "pointer", "color": "#888", "font-size": "0.8em", "margin-left": "4px"}
    )


def build_pnl_stats_table(pnl_series: pd.Series):
    s = pnl_series.dropna()
    if s.empty:
        return html.Div("No PnL data for the selected period.")

    total_n = len(s)

    # Define bins (in decimal)
    pos_bins = [(0.0, 0.03), (0.03, 0.05), (0.05, np.inf)]
    neg_bins = [(-0.03, 0.0), (-0.05, -0.03), (-np.inf, -0.05)]
    pos_labels = ["0% to 3%", "3% to 5%", "5% or more"]
    neg_labels = ["-3% to 0%", "-5% to -3%", "-5% or less"]

    def stats_for_bins(series, bins):
        counts, pct_total, avg_pnl = [], [], []
        for lo, hi in bins:
            if lo == -np.inf:
                mask = (series < hi)
            elif hi == np.inf:
                mask = (series >= lo)
            else:
                mask = (series >= lo) & (series < hi)
            c = mask.sum()
            counts.append(c)
            pct_total.append(100.0 * c / total_n if total_n > 0 else 0.0)
            if c > 0:
                avg_pnl.append(100.0 * series[mask].mean())
            else:
                avg_pnl.append(np.nan)
        return counts, pct_total, avg_pnl

    pos_mask = (s > 0)
    neg_mask = (s < 0)

    pos_counts, pos_pct_total, pos_avg = stats_for_bins(s[pos_mask], pos_bins)
    neg_counts, neg_pct_total, neg_avg = stats_for_bins(s[neg_mask], neg_bins)

    total_pos = sum(pos_counts)
    total_neg = sum(neg_counts)

    total_pos_pct = 100.0 * total_pos / total_n if total_n > 0 else 0.0
    total_neg_pct = 100.0 * total_neg / total_n if total_n > 0 else 0.0
    total_pos_avg = 100.0 * s[pos_mask].mean() if total_pos > 0 else np.nan
    total_neg_avg = 100.0 * s[neg_mask].mean() if total_neg > 0 else np.nan

    total_pos_pnl = s[pos_mask].sum()
    pnl_share = []
    for (lo, hi) in pos_bins:
        if lo == 0.0:
            mask = (s > 0) & (s < hi)
        elif hi == np.inf:
            mask = (s >= lo)
        else:
            mask = (s >= lo) & (s < hi)
        share = 100.0 * s[mask].sum() / total_pos_pnl if total_pos_pnl != 0 else np.nan
        pnl_share.append(share)

    def fmt_pct(x):
        return "" if np.isnan(x) else f"{x:.2f}%"

    table = html.Table([
        html.Thead(
            html.Tr([
                html.Th("", style={"padding": "6px 10px", "background": "#003366",
                                   "color": "white", "textAlign": "center"}),
                *[html.Th(lbl, style={"padding": "6px 10px", "background": "#003366",
                                      "color": "white", "textAlign": "center"})
                  for lbl in pos_labels],
                html.Th("Total", style={"padding": "6px 10px", "background": "#003366",
                                        "color": "white", "textAlign": "center"})
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Th("PnL Positive", colSpan=5,
                        style={"background": "#005b96", "color": "white",
                               "padding": "6px 10px", "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("Instances", style={"fontWeight": "bold", "padding": "4px 10px",
                                            "textAlign": "center"}),
                *[html.Td(str(c), style={"padding": "4px 10px", "textAlign": "center"})
                  for c in pos_counts],
                html.Td(str(total_pos), style={"padding": "4px 10px", "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("% of Total", style={"fontWeight": "bold", "padding": "4px 10px",
                                             "textAlign": "center"}),
                *[html.Td(fmt_pct(x), style={"padding": "4px 10px", "textAlign": "center"})
                  for x in pos_pct_total],
                html.Td(fmt_pct(total_pos_pct), style={"padding": "4px 10px", "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("Avg PnL of Instances", style={"fontWeight": "bold", "padding": "4px 10px",
                                                       "textAlign": "center"}),
                *[html.Td(fmt_pct(x), style={"padding": "4px 10px", "textAlign": "center"})
                  for x in pos_avg],
                html.Td(fmt_pct(total_pos_avg), style={"padding": "4px 10px", "textAlign": "center"})
            ]),

            html.Tr([
                html.Th("PnL Negative", colSpan=5,
                        style={"background": "#005b96", "color": "white",
                               "padding": "6px 10px", "paddingTop": "10px",
                               "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("Instances", style={"fontWeight": "bold", "padding": "4px 10px",
                                            "textAlign": "center"}),
                *[html.Td(str(c), style={"padding": "4px 10px", "textAlign": "center"})
                  for c in neg_counts],
                html.Td(str(total_neg), style={"padding": "4px 10px", "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("% of Total", style={"fontWeight": "bold", "padding": "4px 10px",
                                             "textAlign": "center"}),
                *[html.Td(fmt_pct(x), style={"padding": "4px 10px", "textAlign": "center"})
                  for x in neg_pct_total],
                html.Td(fmt_pct(total_neg_pct), style={"padding": "4px 10px", "textAlign": "center"})
            ]),
            html.Tr([
                html.Td("Avg PnL of Instances", style={"fontWeight": "bold", "padding": "4px 10px",
                                                       "textAlign": "center"}),
                *[html.Td(fmt_pct(x), style={"padding": "4px 10px", "textAlign": "center"})
                  for x in neg_avg],
                html.Td(fmt_pct(total_neg_avg), style={"padding": "4px 10px", "textAlign": "center"})
            ]),

            html.Tr([
                html.Td("% of P/L per positive range", style={
                    "fontWeight": "bold", "padding": "6px 10px",
                    "borderTop": "2px solid #003366", "textAlign": "center"
                }),
                *[html.Td(fmt_pct(x), style={"padding": "6px 10px",
                                             "borderTop": "2px solid #003366",
                                             "textAlign": "center"})
                  for x in pnl_share],
                html.Td("100.00%", style={"padding": "6px 10px",
                                          "borderTop": "2px solid #003366",
                                          "textAlign": "center"})
            ])
        ])
    ], style={
        "borderCollapse": "collapse",
        "minWidth": "520px",
        "background": "white",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
        "marginTop": "8px",
        "textAlign": "center"
    })

    return table


# --- App Layout ---
app.layout = html.Div([
    # --- Header, Timestamp, Controls ---
    html.H1("Financial Stress Dashboard", style={"margin-bottom": "5px"}),
    html.Div([
        dcc.Loading(
            id="loading-refresh",
            type="circle",
            color="#396aff",
            children=[
                html.Button(
                    "Run/Refresh Analysis",
                    id="run-btn",
                    n_clicks=0,
                    disabled=False,
                    style={"margin-bottom": "7px", "margin-right": "18px"}
                ),
                html.Span(id='run-message', style={'color': 'green', "font-size": "1em"}),
                html.Div(id='timestamp-label', className="timestamp-label", style={'font-size': '1em', "margin-bottom": "12px"})
            ]
        )
    ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
    dcc.Store(id='fsi-store'),
    # dcc.Store(id='fsi-events-store', data=[]),

    # --- Main Chart Panels ---
    dcc.Loading(
        id="loading-fsi",
        type="circle",
        children=[
            html.Div([
                # Variable-Level FSI
                html.Div([
                    html.H2([
                        "Variable-Level FSI",
                        info_icon("Shows each variable’s weighted contribution to the overall Financial Stress Index.")
                    ]),
                    html.Label("Select FSI date range:"),
                    dcc.DatePickerRange(
                        id='fsi-date-range',
                        min_date_allowed=None,  # Set after data load
                        max_date_allowed=None,
                        start_date=None,
                        end_date=None,
                        display_format="YYYY-MM-DD",
                        style={"marginBottom": "12px"}
                        ),
                    # html.Label("Show Y-Axis Ticks:"),
                    dcc.Checklist(
                        id='fsi-yaxis-ticks',
                        options=[{'label': 'Show Y-Ticks', 'value': 'show'}],
                        value=[],   #'show'
                        style={'marginBottom': '12px'}
                    ),
                    html.Label("Shade regimes:"),
                    dcc.Checklist(
                        id='ribbon-filter',
                        options=[{'label': r, 'value': r} for r in ['Green','Yellow','Amber','Red']],
                        value=['Green','Yellow','Amber','Red'],  # default: risk-only
                        inline=True,
                        style={'marginBottom': '12px'}
                    ),
                    dcc.Graph(id='fig1'),
                    html.Button("Download as Image", id="dl-fig1", n_clicks=0, className="download-btn")
                ], style={"margin-bottom": "10px"}),

                # Group-Level FSI
                html.Div([
                    html.H2([
                        "Group-Level FSI",
                        info_icon("Aggregated by risk group: Volatility, Rates, Credit, etc.")
                    ]),
                    dcc.Graph(id='fig2'),
                    html.Button("Download as Image", id="dl-fig2", n_clicks=0, className="download-btn")
                ]),

                # Herfindahl–Hirschman Index of variable contribution
                html.Div([
                    html.H2([
                        "Concentration of Contributors (HHI, last 20 days)",
                        info_icon("Herfindahl–Hirschman Index of variable contribution shares over the last 20 days. Higher HHI = more concentrated.")
                    ]),
                    html.Div(id="hhi-metrics", style={"margin": "6px 0 10px 0", "fontSize": "1.05em"}),
                    # dcc.Graph(id='fig-hhi', style={"marginBottom": "10px"}),
                    dash_table.DataTable(
                        id='hhi-table',
                        columns=[{"name": "Variable", "id": "Variable"},
                                {"name": "Share (%)", "id": "Share"}],
                        data=[],
                        style_table={'maxWidth': '800px'},
                        style_cell={'font-family': 'Segoe UI, Arial', 'textAlign': 'left'},
                        style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
                        page_size=15
                    )
                ], style={"marginBottom": "30px"}),
            ], style={'width': '95%', 'margin': 'auto'})
        ]
    ),

                #     # --- FSI Event Annotations Controls ---
                # html.Div([
                #     html.H3("Add Events on FSI Charts", style={"marginTop": "10px"}),
                #     html.Div([
                #         dcc.DatePickerSingle(
                #             id='fsi-event-date',
                #             display_format="YYYY-MM-DD",
                #             placeholder="Event date",
                #             style={"marginRight": "10px"}
                #         ),
                #         dcc.Input(
                #             id='fsi-event-label',
                #             type='text',
                #             placeholder='Event label (e.g. "SVB failure")',
                #             style={"marginRight": "10px", "width": "260px"}
                #         ),
                #         html.Button(
                #             "Add FSI Event",
                #             id='add-fsi-event-btn',
                #             n_clicks=0,
                #             className="download-btn"
                #         )
                #     ], style={"display": "flex", "flexDirection": "row", "alignItems": "center"}),
                #     html.Div(
                #         id='fsi-events-list',
                #         style={"marginTop": "6px", "fontSize": "0.9em", "color": "#555"}
                #     )
                # ], style={"marginBottom": "20px"}),

            # --- Improved PnL Chart Section ---
            html.Div([
                html.H2([
                    "PnL Chart with Regime Ribbons",
                    info_icon("Upload your PnL file. Regimes are highlighted along the PnL curve.")
                ]),

                # 👉 PnL date range selector here
                html.Label("Select PnL date range (chart):"),
                dcc.DatePickerRange(
                    id='pnl-chart-date-range',
                    min_date_allowed=None,
                    max_date_allowed=None,
                    start_date=None,
                    end_date=None,
                    display_format="YYYY-MM-DD",
                    style={"marginBottom": "12px"}
                ),

                dcc.Loading(
                    id="loading-pnl",
                    type="circle",
                    color="#396aff",
                    children=dcc.Graph(
                        id='fig-pnl',
                        style={"margin-bottom": "8px"}
                    )
                ),

                # Row for buttons
                html.Div([
                    html.Button(
                        "Download as Image",
                        id="dl-pnl",
                        n_clicks=0,
                        className="download-btn",
                        style={"margin-right": "auto"}
                    ),
                    dcc.Upload(
                        id='upload-pnl',
                        children=html.Button(
                            'Upload PnL File',
                            className="download-btn",
                            style={"background": "#666", "color": "#fff"}
                        ),
                        accept='.xlsx,.csv',
                        multiple=False,
                        className="dash-uploader",
                        style={"display": "inline-block", "margin-left": "auto"}
                    )
                ],
                    style={
                        "display": "flex",
                        "flexDirection": "row",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "width": "100%",
                        "margin-bottom": "5px"
                    }
                ),

                # Message BELOW upload button, right aligned
                html.Div(
                    html.Span(id='upload-message', style={'color': 'red'}),
                    style={"display": "flex", "flexDirection": "row", "justifyContent": "flex-end", "width": "100%"}
                ),

                # Preview always below everything
                html.Div(id="pnl-preview", style={"margin": "7px 0 7px 0", "font-size": "0.95em"}),

            ], style={'width': '95%', 'margin': 'auto', 'margin-bottom': '30px'}),

            # ========= NEW: Cumulative Returns vs Benchmarks =========
            html.H3([
                "Cumulative Returns vs Benchmarks",
                info_icon("Cumulative performance of NEPTUNE vs S&P 500, S&P 500 equal-weighted, and MSCI ACWI.")
            ], style={"marginTop": "18px"}),

            html.Label("Select date range (cumulative returns):"),
            dcc.DatePickerRange(
                id='cumret-date-range',
                min_date_allowed=None,
                max_date_allowed=None,
                start_date=None,
                end_date=None,
                display_format="YYYY-MM-DD",
                style={"marginBottom": "10px"}
            ),

            dcc.Graph(
                id='fig-cumret',
                style={"margin-bottom": "20px"}
            ),

    html.Hr(),

    # --- Forward-Looking & Regime Metrics ---
    html.Div([
        html.H2([
            "Forward-Looking & Regime Risk Metrics",
            info_icon("Regimes and probability forecasts based on current model results.")
        ]),
        html.Div([
            html.Div([   # LEFT COLUMN
                html.Div([
                    html.H4([
                        "Current Regime (FSI Regime):",
                        info_icon("Classified by FSI volatility and thresholds.")
                    ]),
                    html.Div(
                        id='current-regime',
                        style={
                            'font-size': '1.7em',
                            'font-weight': 'bold',
                            "margin-bottom": "12px"
                        }
                    )
                ], className="card-metric", style={"margin-bottom": "28px"}),
                html.Div([
                    html.H4([
                        "Current Benchmark Regime (HMM Regime):",
                        info_icon("Market regime inferred by a Hidden Markov Model. <br>This reflects the most likely underlying market state, based on a statistical regime-switching model fitted to market data.")
                    ]),
                    html.Div(
                        id='current-hmm',
                        style={
                            'font-size': '1.7em',
                            'font-weight': 'bold'
                        }
                    )
                ], className="card-metric"),
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "flex-start",
                "minWidth": "260px",
                "minHeight": "260px",
                "marginRight": "60px"
            }),
            html.Div([
                html.Div([
                    html.H4([
                        "Probability of Red Regime",
                        info_icon("Predicted probability of entering a high-risk (Red) regime in the next lookahead window, based on current market features.<br>Logit P(Red): Interpretable linear probability model.<br>XGBoost P(Red): Nonlinear, machine-learning-based probability model.")
                    ], style={"margin-bottom": "18px", "text-align": "center"})
                ]),
                html.Div([
                    dcc.Graph(
                        id='prob-red-logit',
                        config={'displayModeBar': False},
                        style={'height': '220px', 'minWidth': "300px", "marginRight": "50px"}
                    ),
                    dcc.Graph(
                        id='prob-red-xgb',
                        config={'displayModeBar': False},
                        style={'height': '220px', 'minWidth': "300px"}
                    )
                ], style={
                    "display": "flex",
                    "flexDirection": "row",
                    "alignItems": "center"
                }),
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "flex": "1"
            })
        ], style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "gap": "65px"
        }),
        html.H4([
            "Historical Regime Transition Matrix",
            info_icon("Rows: FROM regime; Cols: TO regime. Shows likelihood of switching between risk regimes.")
        ]),
        html.Div(
            dcc.Graph(id='regime-transition-matrix', style={"height": "600px"}),
            style={"textAlign": "center", "width": "95%", "margin": "0 auto"}
        ),
        html.H4([
            "Average Time Spent in Each Regime",
            info_icon("Mean number of consecutive days spent in each regime before switching.")
        ]),
        html.Div(id='avg-time-table'),
    ], style={'width': '95%', 'margin': 'auto'}),


    # --- PnL Distribution Analysis Section ---
    html.Div([
        html.H2("PnL Distribution Analysis", style={"marginTop": "30px"}),

        html.Label("Select PnL date range (distribution):"),
        dcc.DatePickerRange(
            id='pnl-dist-date-range',
            min_date_allowed=None,
            max_date_allowed=None,
            start_date=None,
            end_date=None,
            display_format="YYYY-MM-DD",
            style={"marginBottom": "12px"}
        ),

        html.Div([
            dcc.Graph(
                id='dist-pnl-range',
                style={"flex": "1", "minWidth": "380px", "height": "360px"}
            ),
            dcc.Graph(
                id='dist-pnl-full',
                style={"flex": "1", "minWidth": "380px", "height": "360px"}
            )
        ], style={"display": "flex", "flexDirection": "row", "gap": "30px"}),

        # 👉 Container for dynamic PnL stats table at the end of this section
        html.Div(
            id='pnl-stats-table',
            style={"marginTop": "20px", "display": "flex", "justifyContent": "center"}
        ),
    ], style={"margin": "30px 0"}),

    dcc.Download(id="download-image"),
], style={
    'font-family': 'Segoe UI, Arial, sans-serif',
    'background-color': '#f7f8fa',
    "padding-bottom": "35px"
})



# @app.callback(
#     Output('fsi-events-store', 'data'),
#     Output('fsi-events-list', 'children'),
#     Input('add-fsi-event-btn', 'n_clicks'),
#     State('fsi-event-date', 'date'),
#     State('fsi-event-label', 'value'),
#     State('fsi-events-store', 'data'),
#     prevent_initial_call=True
# )

def add_fsi_event(n_clicks, date_str, label, events):
    if not date_str or not label:
        raise dash.exceptions.PreventUpdate

    events = events or []
    events.append({"date": date_str, "label": label})

    # Simple text list of current events
    items = [
        html.Div(f"{i+1}. {e['date']} – {e['label']}")
        for i, e in enumerate(events)
    ]

    return events, items


# --- 1. RUN/REFRESH BUTTON: Pipeline Callback with Caching and Button Disable ---
@app.callback(
    [Output('fsi-store', 'data'), 
     Output('run-message', 'children'), 
     Output('run-btn', 'disabled'), 
     Output('timestamp-label', 'children')],
    Input('run-btn', 'n_clicks'),
    prevent_initial_call=True
)

def run_full_pipeline(n_clicks):
    import time

    cache_key = "fsi_analysis_latest_hybrid_v2"
    msg = "⏳ Analysis running, please wait..."
    timestamp_label = ""

    # Serve from cache if present
    result = cache.get(cache_key)
    if result is not None:
        msg = f"✅ Served from cache (last computed at {result.get('timestamp', 'unknown')})"
        timestamp_label = f"Last update: {result.get('timestamp', 'unknown')}"
        return result, msg, False, timestamp_label

    # --- Load & merge data ---
    config = load_configuration()
    df = merge_data(config)
    if df is None or df.empty:
        return dash.no_update, "❌ Data loading failed", False, ""

    # --- Estimate FSI & ω with stability diagnostics (no leakage) ---
    min_history = int(config['fsi']['window_size'])

    fsi_series, omega_history, cos_sim_series, _ = estimate_fsi_expanding_with_als(
        df,
        min_history=min_history,
        n_iter=int(config['fsi']['n_iter']),
        stability_threshold=float(config['fsi']['stability_threshold'])
    )


    # --- C1: robust orientation + freeze + audit (already implemented in utils) ---
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


    # Persist orientation audit (optional)
    base_path = "./cache-directory"
    os.makedirs(os.path.join(base_path, "qc"), exist_ok=True)
    try:
        orient_audit.to_csv(os.path.join(base_path, "qc", "orientation_flip_audit_app.csv"), index=False)
    except Exception:
        pass

    # # --- A1: leakage-free time-varying contributions using contemporaneous ω_t ---
    # variable_contribs = compute_timevarying_contributions(
    #     df.loc[fsi_series.index], omega_history, window_size=int(config['fsi']['window_size'])
    # )


    # --- A1: leakage-free time-varying contributions using contemporaneous ω_t ---
    min_history = int(config['fsi']['window_size'])
    variable_contribs = compute_timevarying_contributions(
        df.loc[fsi_series.index],
        omega_history,
        min_history=min_history
    )


    # --- C2: dynamic group mapping from present columns ---
    group_map = build_dynamic_group_map(variable_contribs)
    grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)

    # Small attribution sanity check
    try:
        err = (grouped_contribs.drop(columns=['FSI']).sum(axis=1) - variable_contribs['FSI']).abs().max()
        if pd.notna(err) and err > 1e-8:
            logging.warning(f"[ATTR] Group attribution mismatch max={err:.2e}")
    except Exception:
        pass

    # --- Regime classification (fixed, not recomputed on UI tweaks) ---
    regimes_full = classify_regime_fsi_improved(variable_contribs['FSI'])

    # Align DF and attach regimes (for any lightweight displays later)
    df_aligned = df.loc[fsi_series.index].copy()
    df_aligned["Regime"] = regimes_full.astype(str)

    # =======================
    # Precompute ALL heavy stuff here (one time per refresh)
    # =======================
    # 1) HMM regime today
    hmm_state, _, _ = run_hmm(
        df_aligned, n_states=4,
        columns=[c for c in df_aligned.columns if ('FSI' in c or 'dev' in c or 'stress' in c or 'OAS' in c)]
    )
    hmm_regime_map = {0: "Green", 1: "Yellow", 2: "Amber", 3: "Red"}
    hmm_regime_today = hmm_regime_map.get(hmm_state, f"Unknown ({hmm_state})")

    # 2) Forward-looking probabilities (logit & XGB) with purged CV
    prob_logit, _, _, _, auc_logit = predict_regime_probability(
        df_aligned, model_type='logit', lookahead=20, n_splits=5, use_purged=True, embargo=20)
    prob_xgb, _, _, _, auc_xgb = predict_regime_probability(
        df_aligned, model_type='xgboost', lookahead=20, n_splits=5, use_purged=True, embargo=20)

    # 3) Transition matrix & average time in regime
    tm_df = compute_transition_matrix(df_aligned['Regime'])
    avg_time_series = average_time_in_regime(df_aligned['Regime'])

    # =======================
    # Pack everything for the UI (render-only later)
    # =======================
    result = {
        # time series
        "fsi_series": fsi_series.to_json(date_format="iso", orient="split"),
        "variable_contribs": variable_contribs.to_json(date_format="iso", orient="split"),
        "grouped_contribs": grouped_contribs.to_json(date_format="iso", orient="split"),
        "df": df_aligned.to_json(date_format="iso", orient="split"),
        "regime_series": regimes_full.astype(str).to_json(date_format="iso", orient="split"),

        # heavy precomputations for static display
        "hmm_state_today": int(hmm_state),
        "hmm_regime_today": hmm_regime_today,
        "prob_red_logit": float(prob_logit),
        "auc_logit": float(auc_logit),
        "prob_red_xgb": float(prob_xgb),
        "auc_xgb": float(auc_xgb),
        "transition_matrix": tm_df.to_dict(),          # rows: FROM, cols: TO
        "avg_time_in_regime": avg_time_series.to_dict(),

        # housekeeping
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    cache.set(cache_key, result, expire=3600)
    msg = f"✅ Analysis completed and cached at {result['timestamp']}"
    timestamp_label = f"Last update: {result['timestamp']}"
    return result, msg, False, timestamp_label

#####

# --- 2. Update Main Charts/Stats When Data Is Available ---
@app.callback(
    [
        Output('fig1', 'figure'),
        Output('fig2', 'figure'),
        Output('current-regime', 'children'),
        Output('current-hmm', 'children'),
        Output('prob-red-logit', 'figure'),
        Output('prob-red-xgb', 'figure'),
        Output('regime-transition-matrix', 'figure'),
        Output('avg-time-table', 'children'),
        Output('hhi-metrics', 'children'),
        Output('hhi-table', 'data'),
    ],
    [
        Input('fsi-store', 'data'),
        Input('fsi-date-range', 'start_date'),
        Input('fsi-date-range', 'end_date'),
        Input('fsi-yaxis-ticks', 'value'),
        Input('ribbon-filter', 'value'),
        # Input('fsi-events-store', 'data'),
    ],
)
def update_all_from_store(data, start_date, end_date, ytick_opts, ribbon_filter, fsi_events=False):
    if data is None:
        raise dash.exceptions.PreventUpdate

    # --- Safely format AUC labels (handle None) ---
    auc_logit = data.get("auc_logit", None)
    auc_xgb   = data.get("auc_xgb", None)

    def fmt_auc(auc_value):
        try:
            if auc_value is None:
                return "n/a"
            return f"{float(auc_value):.2f}"
        except (TypeError, ValueError):
            return "n/a"

    auc_logit_label = fmt_auc(auc_logit)
    auc_xgb_label   = fmt_auc(auc_xgb)

    # --- Load precomputed series/frames ---
    variable_contribs = pd.read_json(io.StringIO(data["variable_contribs"]), orient="split")
    grouped_contribs  = pd.read_json(io.StringIO(data["grouped_contribs"]), orient="split")
    regimes_full      = pd.read_json(io.StringIO(data["regime_series"]), orient="split", typ="series")

    # Make sure indices are DateTimeIndex
    variable_contribs.index = pd.to_datetime(variable_contribs.index)
    grouped_contribs.index  = pd.to_datetime(grouped_contribs.index)
    regimes_full.index      = pd.to_datetime(regimes_full.index)

    # --- Date filtering (render-only, no heavy computation) ---
    idx = variable_contribs.index

    if start_date:
        start_ts = pd.to_datetime(start_date)
    else:
        start_ts = idx.min()

    if end_date:
        end_ts = pd.to_datetime(end_date)
    else:
        end_ts = idx.max()

    # Slice by date (inclusive)
    variable_contribs_window = variable_contribs.loc[start_ts:end_ts]
    grouped_contribs_window  = grouped_contribs.loc[start_ts:end_ts]
    regimes_filtered         = regimes_full.loc[start_ts:end_ts]

    # If the window has no data, return empty figs with a message
    if variable_contribs_window.empty or grouped_contribs_window.empty:
        fig1 = go.Figure()
        fig1.update_layout(title="No FSI data in selected date range.")
        fig2 = go.Figure()
        fig2.update_layout(title="No FSI data in selected date range.")

        curr_regime_html = regime_color_text(regimes_full.iloc[-1])
        hmm_regime_html  = regime_color_text(data.get("hmm_regime_today", "Unknown"))

        # Gauges (precomputed)
        def make_prob_gauge(prob, label):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(prob) * 100.0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': label, "font": {"size": 13}, "align": "center"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#e74c3c" if prob > 0.6 else "#f1c40f" if prob > 0.3 else "#27ae60"},
                    'steps': [
                        {'range': [0, 30], 'color': "#d4efdf"},
                        {'range': [30, 60], 'color': "#f7f7a4"},
                        {'range': [60, 100], 'color': "#f9c9c4"}
                    ],
                },
                number={'suffix': "%"}
            ))
            fig.update_layout(margin=dict(l=10, r=10, t=45, b=14),
                              paper_bgcolor="#f7f8fa", height=210)
            return fig

        # fig_prob_logit = make_prob_gauge(
        #     data["prob_red_logit"],
        #     f"Logit P(Red) (AUC: {data['auc_logit']:.2f})"
        # )
        # fig_prob_xgb = make_prob_gauge(
        #     data["prob_red_xgb"],
        #     f"XGBoost P(Red) (AUC: {data['auc_xgb']:.2f})"
        # )

        fig_prob_logit = make_prob_gauge(
            data["prob_red_logit"],
            f"Logit P(Red) (AUC: {auc_logit_label})"
        )
        fig_prob_xgb = make_prob_gauge(
            data["prob_red_xgb"],
            f"XGBoost P(Red) (AUC: {auc_xgb_label})"
        )

        # Transition matrix (precomputed, unchanged)
        tm = pd.DataFrame(data["transition_matrix"]).fillna(0.0)
        regimes_order = ["Green", "Yellow", "Amber", "Red"]
        tm = tm.reindex(index=regimes_order, columns=regimes_order, fill_value=0.0)

        if (tm.values.sum() == 0) or (tm.shape != (4, 4)):
            fig_matrix = go.Figure()
            fig_matrix.update_layout(
                title="Transition matrix unavailable for this sample.",
                plot_bgcolor="#f7f8fa", paper_bgcolor="#f7f8fa",
                xaxis_visible=False, yaxis_visible=False
            )
        else:
            z = tm.values
            x = list(tm.columns)
            y = list(tm.index)
            hovertext = [[f"From <b>{y[i]}</b> to <b>{x[j]}</b>: {z[i][j]:.2%}"
                          for j in range(len(x))] for i in range(len(y))]
            fig_matrix = go.Figure(data=go.Heatmap(
                z=z, x=x, y=y,
                colorscale='RdYlGn', reversescale=True,
                hoverinfo='text', text=hovertext, zmin=0, zmax=1,
                colorbar=dict(title="Prob.")
            ))
            fig_matrix.update_layout(
                title="<b>Regime Transition Matrix (Rows: FROM, Cols: TO)</b>",
                xaxis_title="To Regime", yaxis_title="From Regime",
                margin=dict(l=25, r=25, t=45, b=40),
                font=dict(size=12),
                plot_bgcolor="#f7f8fa", paper_bgcolor="#f7f8fa"
            )

        # Average time in regime (precomputed)
        avg_time = data.get("avg_time_in_regime", {})
        def cell(reg): return f"{float(avg_time.get(reg, 0.0)):.1f}"

        avg_time_table = html.Div([
            html.Div([
                html.Span("The average time spent in regime is measured in number of days.", style={
                    "fontWeight": "bold", "fontSize": "1.17em", "marginRight": "7px"
                }),
                info_icon("Mean number of consecutive days spent in each regime before switching.")
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th(reg, style={
                            "padding": "8px 18px",
                            "background": "#f2f3f5",
                            "color": REGIME_COLORS[reg],
                            "fontWeight": "bold",
                            "fontSize": "1.09em",
                            "borderRadius": "7px 7px 0 0"
                        }) for reg in regimes_order
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(cell(reg), style={
                            "padding": "8px 18px",
                            "fontWeight": "500",
                            "fontSize": "1.09em"
                        }) for reg in regimes_order
                    ])
                ])
            ], style={
                "borderCollapse": "separate",
                "borderSpacing": "0",
                "marginTop": "3px",
                "background": "#fff",
                "borderRadius": "8px",
                "boxShadow": "0 2px 8px 0 rgba(0,0,0,0.05)",
                "width": "auto",
                "minWidth": "350px"
            })
        ], style={"marginTop": "15px", "marginBottom": "20px"})

        # HHI message for empty range
        hhi_text = "HHI unavailable for the selected range."
        table_data = []

        return (fig1, fig2, curr_regime_html, hmm_regime_html,
                fig_prob_logit, fig_prob_xgb, fig_matrix, avg_time_table,
                hhi_text, table_data)

    # --- Normal case: non-empty window, build figures on the sliced data ---
    fig1 = plot_group_contributions_with_regime(
        variable_contribs_window, regimes=regimes_filtered, regime_filter=ribbon_filter
    )
    fig2 = plot_grouped_contributions(
        grouped_contribs_window, regimes=regimes_filtered, regime_filter=ribbon_filter
    )

    # Y-axis ticks visibility toggle
    show_ticks = 'show' in (ytick_opts or [])
    fig1.update_yaxes(showticklabels=show_ticks)
    fig2.update_yaxes(showticklabels=show_ticks)

    # Optional FSI events
    if fsi_events and not variable_contribs_window.empty:
        idx_min = variable_contribs_window.index.min()
        idx_max = variable_contribs_window.index.max()
        for ev in fsi_events:
            try:
                ev_date = pd.to_datetime(ev.get("date"))
                label = ev.get("label", "")
                if ev_date < idx_min or ev_date > idx_max:
                    continue
                for fig in (fig1, fig2):
                    fig.add_vline(
                        x=ev_date,
                        line_dash="dash",
                        line_color="#333",
                        opacity=0.7
                    )
                    fig.add_annotation(
                        x=ev_date,
                        y=1.02,
                        xref="x",
                        yref="paper",
                        text=label,
                        showarrow=False,
                        textangle=90,
                        font=dict(size=10, color="#333"),
                        align="left"
                    )
            except Exception:
                continue

    # Current regime = last of precomputed regimes (not sliced)
    curr_regime = regimes_full.iloc[-1]
    curr_regime_html = regime_color_text(curr_regime)

    # HMM regime (precomputed)
    hmm_regime_html = regime_color_text(data.get("hmm_regime_today", "Unknown"))

    # Gauges (precomputed)
    def make_prob_gauge(prob, label):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(prob) * 100.0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': label, "font": {"size": 13}, "align": "center"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c" if prob > 0.6 else "#f1c40f" if prob > 0.3 else "#27ae60"},
                'steps': [
                    {'range': [0, 30], 'color': "#d4efdf"},
                    {'range': [30, 60], 'color': "#f7f7a4"},
                    {'range': [60, 100], 'color': "#f9c9c4"}
                ],
            },
            number={'suffix': "%"}
        ))
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=14),
                          paper_bgcolor="#f7f8fa", height=210)
        return fig

    # fig_prob_logit = make_prob_gauge(
    #     data["prob_red_logit"],
    #     f"Logit P(Red) (AUC: {data['auc_logit']:.2f})"
    # )
    # fig_prob_xgb = make_prob_gauge(
    #     data["prob_red_xgb"],
    #     f"XGBoost P(Red) (AUC: {data['auc_xgb']:.2f})"
    # )

    fig_prob_logit = make_prob_gauge(
        data["prob_red_logit"],
        f"Logit P(Red) (AUC: {auc_logit_label})"
    )
    fig_prob_xgb = make_prob_gauge(
        data["prob_red_xgb"],
        f"XGBoost P(Red) (AUC: {auc_xgb_label})"
    )

    # Transition matrix (precomputed)
    tm = pd.DataFrame(data["transition_matrix"]).fillna(0.0)
    regimes_order = ["Green", "Yellow", "Amber", "Red"]
    tm = tm.reindex(index=regimes_order, columns=regimes_order, fill_value=0.0)

    if (tm.values.sum() == 0) or (tm.shape != (4, 4)):
        fig_matrix = go.Figure()
        fig_matrix.update_layout(
            title="Transition matrix unavailable for this sample.",
            plot_bgcolor="#f7f8fa", paper_bgcolor="#f7f8fa",
            xaxis_visible=False, yaxis_visible=False
        )
    else:
        z = tm.values
        x = list(tm.columns)
        y = list(tm.index)
        hovertext = [[f"From <b>{y[i]}</b> to <b>{x[j]}</b>: {z[i][j]:.2%}"
                      for j in range(len(x))] for i in range(len(y))]
        fig_matrix = go.Figure(data=go.Heatmap(
            z=z, x=x, y=y,
            colorscale='RdYlGn', reversescale=True,
            hoverinfo='text', text=hovertext, zmin=0, zmax=1,
            colorbar=dict(title="Prob.")
        ))
        fig_matrix.update_layout(
            title="<b>Regime Transition Matrix (Rows: FROM, Cols: TO)</b>",
            xaxis_title="To Regime", yaxis_title="From Regime",
            margin=dict(l=25, r=25, t=45, b=40),
            font=dict(size=12),
            plot_bgcolor="#f7f8fa", paper_bgcolor="#f7f8fa"
        )

    # Average time in regime (precomputed)
    avg_time = data.get("avg_time_in_regime", {})
    def cell(reg): return f"{float(avg_time.get(reg, 0.0)):.1f}"

    avg_time_table = html.Div([
        html.Div([
            html.Span("The average time spent in regime is measured in number of days.", style={
                "fontWeight": "bold", "fontSize": "1.17em", "marginRight": "7px"
            }),
            info_icon("Mean number of consecutive days spent in each regime before switching.")
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
        html.Table([
            html.Thead(
                html.Tr([
                    html.Th(reg, style={
                        "padding": "8px 18px",
                        "background": "#f2f3f5",
                        "color": REGIME_COLORS[reg],
                        "fontWeight": "bold",
                        "fontSize": "1.09em",
                        "borderRadius": "7px 7px 0 0"
                    }) for reg in regimes_order
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(cell(reg), style={
                        "padding": "8px 18px",
                        "fontWeight": "500",
                        "fontSize": "1.09em"
                    }) for reg in regimes_order
                ])
            ])
        ], style={
            "borderCollapse": "separate",
            "borderSpacing": "0",
            "marginTop": "3px",
            "background": "#fff",
            "borderRadius": "8px",
            "boxShadow": "0 2px 8px 0 rgba(0,0,0,0.05)",
            "width": "auto",
            "minWidth": "350px"
        })
    ], style={"marginTop": "15px", "marginBottom": "20px"})

    # --- HHI over last 20 rows of the *window* ---
    hhi, eff_n, ranking = compute_hhi_ranking(variable_contribs_window, window=20)
    if np.isnan(hhi):
        hhi_text = "HHI unavailable for the selected range."
    else:
        hhi_text = f"HHI = {hhi:.3f}  |  Effective number of contributors ≈ {eff_n:.1f} (= 1/HHI)"

    table_data = []
    if ranking is not None and not ranking.empty:
        top = ranking.head(20) * 100.0
        table_data = [{"Variable": k, "Share": f"{v:.2f}%"} for k, v in top.items()]

    return (fig1, fig2, curr_regime_html, hmm_regime_html,
            fig_prob_logit, fig_prob_xgb, fig_matrix, avg_time_table,
            hhi_text, table_data)


# --- 3. PnL Upload Logic (now supports CSV and preview, error feedback) ---

@app.callback(
    [Output('fig-pnl', 'figure'),
     Output('upload-message', 'children'),
     Output('pnl-preview', 'children')],
    [Input('upload-pnl', 'contents'),
     Input('pnl-chart-date-range', 'start_date'),
     Input('pnl-chart-date-range', 'end_date')],
    [State('upload-pnl', 'filename'),
     State('fsi-store', 'data')]
)

def update_pnl(upload_contents, start_date, end_date, upload_filename, fsi_data):
    if fsi_data is None:
        return go.Figure(), "Please run analysis first.", ""

    variable_contribs = pd.read_json(io.StringIO(fsi_data["variable_contribs"]), orient="split")
    fsi_series        = pd.read_json(io.StringIO(fsi_data["fsi_series"]), typ="series", orient="split")
    df_all            = pd.read_json(io.StringIO(fsi_data["df"]), orient="split")  # <- has Regime
    regimes_full      = df_all["Regime"].astype(str)

    msg = ""
    pnl_df = None
    preview_table = ""   # we no longer show the stats table here

    if upload_contents is not None:
        try:
            # Decode uploaded file
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)

            if upload_filename and upload_filename.lower().endswith('.csv'):
                pnl_df = pd.read_csv(io.BytesIO(decoded))
            else:
                pnl_df = pd.read_excel(io.BytesIO(decoded))

            # Normalize and strip column names
            pnl_df.columns = [c.strip() for c in pnl_df.columns]
            lower_map = {c.lower(): c for c in pnl_df.columns}

            # Accept common variants for date and pnl column names
            date_candidates = ['date', 'datetime', 'timestamp']
            pnl_candidates  = ['p/l', 'p&l', 'pnl', 'pl', 'return', 'ret', 'pnl%', 'p/l %', 'p&l %']

            date_col = next((lower_map[c] for c in date_candidates if c in lower_map), None)
            pnl_col  = next((lower_map[c] for c in pnl_candidates  if c in lower_map), None)

            if date_col is None or pnl_col is None:
                msg = ("File must contain a Date column and a PnL column "
                       "(accepted names: Date/Datetime/Timestamp + P/L, P&L, PnL, Return, etc.).")
                pnl_df = None
            else:
                # Parse dates
                pnl_df['Date'] = pd.to_datetime(pnl_df[date_col], errors='coerce')
                pnl_df = pnl_df.dropna(subset=['Date'])

                # Parse PnL values, handle percent strings and scaling
                pnl_series_raw = pnl_df[pnl_col]
                if pnl_series_raw.dtype == object:
                    pnl_series_clean = pnl_series_raw.astype(str).str.replace('%', '', regex=False).str.replace(',', '')
                    pnl_series = pd.to_numeric(pnl_series_clean, errors='coerce')
                    if pnl_series.dropna().abs().max() > 1.0:
                        pnl_series = pnl_series / 100.0
                else:
                    pnl_series = pnl_series_raw.astype(float)

                # Assign standardized column names and index
                pnl_df = pnl_df.assign(**{'P/L': pnl_series}).set_index('Date').sort_index()

                # Filter by selected date range (for chart)
                if start_date and end_date:
                    mask = (pnl_df.index >= pd.to_datetime(start_date)) & \
                           (pnl_df.index <= pd.to_datetime(end_date))
                    pnl_df = pnl_df.loc[mask]

                msg = "PnL file loaded."

        except Exception as e:
            msg = f"Error reading file: {e}"
            pnl_df = None

    # Plot PnL if available
    if pnl_df is not None and not pnl_df.empty:
        fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series, regimes=regimes_full)
    else:
        fig_pnl = go.Figure()
        fig_pnl.update_layout(title="PnL Chart (Upload file to see data)")

    return fig_pnl, msg, preview_table

# --- Set available date range after PnL upload ---
@app.callback(
    [
        Output('pnl-chart-date-range', 'min_date_allowed'),
        Output('pnl-chart-date-range', 'max_date_allowed'),
        Output('pnl-chart-date-range', 'start_date'),
        Output('pnl-chart-date-range', 'end_date'),

        Output('pnl-dist-date-range', 'min_date_allowed'),
        Output('pnl-dist-date-range', 'max_date_allowed'),
        Output('pnl-dist-date-range', 'start_date'),
        Output('pnl-dist-date-range', 'end_date'),

        Output('cumret-date-range', 'min_date_allowed'),
        Output('cumret-date-range', 'max_date_allowed'),
        Output('cumret-date-range', 'start_date'),
        Output('cumret-date-range', 'end_date'),
    ],
    Input('upload-pnl', 'contents'),
    State('upload-pnl', 'filename')
)
def set_datepicker_limits(upload_contents, upload_filename):
    if not upload_contents:
        # 12 Nones (for 3 DatePickerRange components)
        return (None, None, None, None,
                None, None, None, None,
                None, None, None, None)

    content_type, content_string = upload_contents.split(',')
    decoded = base64.b64decode(content_string)
    if upload_filename.lower().endswith('.csv'):
        pnl_df = pd.read_csv(io.BytesIO(decoded))
    else:
        pnl_df = pd.read_excel(io.BytesIO(decoded))

    pnl_df.columns = [c.strip() for c in pnl_df.columns]
    col_map = {c.lower(): c for c in pnl_df.columns}
    if 'date' not in col_map:
        return (None, None, None, None,
                None, None, None, None,
                None, None, None, None)

    pnl_df['Date'] = pd.to_datetime(pnl_df[col_map['date']])
    min_date = pnl_df['Date'].min().date()
    max_date = pnl_df['Date'].max().date()

    from datetime import date
    default_start = max(min_date, date(2019, 1, 1))

    # Use same defaults for chart, distribution, and cumulative returns
    return (
        min_date, max_date, default_start, max_date,  # pnl-chart-date-range
        min_date, max_date, default_start, max_date,  # pnl-dist-date-range
        min_date, max_date, default_start, max_date   # cumret-date-range
    )


@app.callback(
    [
        Output('fsi-date-range', 'min_date_allowed'),
        Output('fsi-date-range', 'max_date_allowed'),
        Output('fsi-date-range', 'start_date'),
        Output('fsi-date-range', 'end_date'),
    ],
    Input('fsi-store', 'data')
)
def set_fsi_date_limits(data):
    if data is None:
        return None, None, None, None
    grouped_contribs = pd.read_json(io.StringIO(data["grouped_contribs"]), orient="split")
    min_date = grouped_contribs.index.min().date()
    max_date = grouped_contribs.index.max().date()
    return min_date, max_date, min_date, max_date

# --- 4. Download as Image (all charts) ---
@app.callback(
    Output("download-image", "data"),
    [Input("dl-fig1", "n_clicks"),
     Input("dl-fig2", "n_clicks"),
     Input("dl-pnl", "n_clicks")],
    [State('fig1', 'figure'), State('fig2', 'figure'), State('fig-pnl', 'figure')],
    prevent_initial_call=True
)
def download_figure(dl1, dl2, dl3, fig1, fig2, fig_pnl):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn_id == "dl-fig1":
        fig = go.Figure(fig1)
    elif btn_id == "dl-fig2":
        fig = go.Figure(fig2)
    else:
        fig = go.Figure(fig_pnl)
    # Export as PNG
    return dcc.send_bytes(fig.to_image(format="png"), filename=f"{btn_id}.png")

# --- Run the distribution plots ---

@app.callback(
    [Output('dist-pnl-range', 'figure'),
     Output('dist-pnl-full', 'figure'),
     Output('pnl-stats-table', 'children')],
    [
        Input('upload-pnl', 'contents'),
        Input('pnl-dist-date-range', 'start_date'),
        Input('pnl-dist-date-range', 'end_date')
    ],
    State('upload-pnl', 'filename')
)

def update_pnl_distributions(upload_contents, start_date, end_date, upload_filename):
    if not upload_contents:
        return go.Figure(), go.Figure(), html.Div("Upload a PnL file to see distributions and stats.")

    content_type, content_string = upload_contents.split(',')
    decoded = base64.b64decode(content_string)

    # Read file
    if upload_filename.lower().endswith('.csv'):
        pnl_df = pd.read_csv(io.BytesIO(decoded))
    else:
        pnl_df = pd.read_excel(io.BytesIO(decoded))

    # Normalize and strip column names
    pnl_df.columns = [c.strip() for c in pnl_df.columns]
    lower_map = {c.lower(): c for c in pnl_df.columns}

    # Same flexible column detection as the main PnL callback
    date_candidates = ['date', 'datetime', 'timestamp']
    pnl_candidates  = ['p/l', 'p&l', 'pnl', 'pl', 'return', 'ret', 'pnl%', 'p/l %', 'p&l %']

    date_col = next((lower_map[c] for c in date_candidates if c in lower_map), None)
    pnl_col  = next((lower_map[c] for c in pnl_candidates  if c in lower_map), None)

    if date_col is None or pnl_col is None:
        msg = html.Div("File must contain a Date column and a PnL column "
                       "(accepted names: Date/Datetime/Timestamp + P/L, P&L, PnL, Return, etc.).")
        return go.Figure(), go.Figure(), msg

    # Parse dates
    pnl_df['Date'] = pd.to_datetime(pnl_df[date_col], errors='coerce')
    pnl_df = pnl_df.dropna(subset=['Date']).sort_values('Date')

    # Parse PnL values, handle percent strings and scaling
    pnl_series_raw = pnl_df[pnl_col]
    if pnl_series_raw.dtype == object:
        pnl_series_clean = pnl_series_raw.astype(str).str.replace('%', '', regex=False).str.replace(',', '')
        pnl_series = pd.to_numeric(pnl_series_clean, errors='coerce')
        if pnl_series.dropna().abs().max() > 1.0:
            pnl_series = pnl_series / 100.0
    else:
        pnl_series = pnl_series_raw.astype(float)

    # Put into a clean frame
    pnl_df = pnl_df.assign(**{'P/L': pnl_series}).set_index('Date').sort_index()

    # Filter by distribution date range (independent from chart range)
    if start_date and end_date:
        mask = (pnl_df.index >= pd.to_datetime(start_date)) & \
               (pnl_df.index <= pd.to_datetime(end_date))
        pnl_df = pnl_df.loc[mask]

    if pnl_df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No PnL data in selected range.")
        return empty_fig, empty_fig, html.Div("No PnL data in the selected range.")

    # Series for plotting
    pnl_series_window = pnl_df['P/L']

    # Title based on window
    date0 = pnl_df.index.min().strftime("%b-%Y")
    date1 = pnl_df.index.max().strftime("%b-%Y")
    period_title = f"{date0} to {date1}"

    fig_range = plot_distribution_plotly(pnl_series_window, period_title, pnl_range=(-0.03, 0.03))
    fig_full  = plot_distribution_plotly(pnl_series_window, period_title)

    # 👉 Dynamic stats table based on THIS window
    stats_table = build_pnl_stats_table(pnl_series_window)

    return fig_range, fig_full, stats_table


@app.callback(
    Output('fig-cumret', 'figure'),
    Input('upload-pnl', 'contents'),
    Input('cumret-date-range', 'start_date'),
    Input('cumret-date-range', 'end_date'),
    State('upload-pnl', 'filename')
)
def update_cumret_chart(upload_contents, start_date, end_date, upload_filename):
    # If no PnL uploaded or no benchmarks, return an empty chart
    if (not upload_contents) or (upload_filename is None):
        empty_series = pd.Series(dtype=float)
        return make_cumret_figure(
            neptune_returns=empty_series,
            benchmark_returns=benchmark_returns,
            start_date=start_date,
            end_date=end_date
        )

    # Decode uploaded file
    content_type, content_string = upload_contents.split(',')
    decoded = base64.b64decode(content_string)

    # Read file
    if upload_filename.lower().endswith('.csv'):
        pnl_df = pd.read_csv(io.BytesIO(decoded))
    else:
        pnl_df = pd.read_excel(io.BytesIO(decoded))

    # Normalize and strip column names
    pnl_df.columns = [c.strip() for c in pnl_df.columns]
    lower_map = {c.lower(): c for c in pnl_df.columns}

    # Same flexible detection as in update_pnl / distributions
    date_candidates = ['date', 'datetime', 'timestamp']
    pnl_candidates  = ['p/l', 'p&l', 'pnl', 'pl', 'return', 'ret', 'pnl%', 'p/l %', 'p&l %']

    date_col = next((lower_map[c] for c in date_candidates if c in lower_map), None)
    pnl_col  = next((lower_map[c] for c in pnl_candidates  if c in lower_map), None)

    if date_col is None or pnl_col is None:
        empty_series = pd.Series(dtype=float)
        return make_cumret_figure(
            neptune_returns=empty_series,
            benchmark_returns=benchmark_returns,
            start_date=start_date,
            end_date=end_date
        )

    # Parse dates
    pnl_df['Date'] = pd.to_datetime(pnl_df[date_col], errors='coerce')
    pnl_df = pnl_df.dropna(subset=['Date']).sort_values('Date')

    # Parse PnL values, handle percent strings and scaling
    pnl_series_raw = pnl_df[pnl_col]
    if pnl_series_raw.dtype == object:
        pnl_series_clean = pnl_series_raw.astype(str).str.replace('%', '', regex=False).str.replace(',', '')
        pnl_series = pd.to_numeric(pnl_series_clean, errors='coerce')
        if pnl_series.dropna().abs().max() > 1.0:
            pnl_series = pnl_series / 100.0
    else:
        pnl_series = pnl_series_raw.astype(float)

    # NEPTUNE returns series with Date index
    neptune_returns = pd.Series(pnl_series.values, index=pnl_df['Date'])
    neptune_returns = neptune_returns.dropna()
    neptune_returns.index.name = 'Date'
    neptune_returns.name = 'NEPTUNE'

    # Build figure (make_cumret_figure should rebase at start_date)
    return make_cumret_figure(
        neptune_returns=neptune_returns,
        benchmark_returns=benchmark_returns,
        start_date=start_date,
        end_date=end_date,
    )



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
