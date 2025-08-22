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


# --- Your framework imports ---
from main import load_configuration, merge_data
from fsi_estimation import estimate_fsi_recursive_rolling_with_stability, compute_variable_contributions
from plotting import (
    plot_group_contributions_with_regime,
    plot_grouped_contributions,
    plot_pnl_with_regime_ribbons,
    plot_distribution_plotly,
)
from utils import (
    aggregate_contributions_by_group,
    get_current_regime, run_hmm, predict_regime_probability, compute_transition_matrix, classify_risk_regime_hybrid, average_time_in_regime, classify_adaptive_regime_hybrid_fallback
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
                    html.Label("Show Y-Axis Ticks:"),
                    dcc.Checklist(
                        id='fsi-yaxis-ticks',
                        options=[{'label': 'Show Y-Ticks', 'value': 'show'}],
                        value=['show'],
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

                # --- Improved PnL Chart Section ---
                html.Div([
                    html.H2([
                        "PnL Chart with Regime Ribbons",
                        info_icon("Upload your PnL file. Regimes are highlighted along the PnL curve.")
                    ]),
                    dcc.Graph(id='fig-pnl', style={"margin-bottom": "8px"}),

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
                            children=html.Button('Upload PnL File', className="download-btn", style={"background": "#666", "color": "#fff"}),
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

                ], style={'margin-bottom': '30px'}),
            ], style={'width': '95%', 'margin': 'auto'})
        ]
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
        # dcc.Graph(id='regime-transition-matrix'),
        html.Div(
            dcc.Graph(id='regime-transition-matrix'),
            style={"textAlign": "center", "maxWidth": "440px", "margin": "0 auto"}
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
        html.Label("Select PnL date range:"),
        dcc.DatePickerRange(
            id='pnl-date-range',
            min_date_allowed=None,
            max_date_allowed=None,
            start_date=None,
            end_date=None,
            display_format="YYYY-MM-DD",
            style={"marginBottom": "12px"}
        ),
        html.Div([
            dcc.Graph(id='dist-pnl-range', style={"flex": "1", "minWidth": "380px", "height": "360px"}),
            dcc.Graph(id='dist-pnl-full', style={"flex": "1", "minWidth": "380px", "height": "360px"})
        ], style={"display": "flex", "flexDirection": "row", "gap": "30px"})
    ], style={"margin": "30px 0"}),

    dcc.Download(id="download-image"),
], style={
    'font-family': 'Segoe UI, Arial, sans-serif',
    'background-color': '#f7f8fa',
    "padding-bottom": "35px"
})

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

    # cache_key = "fsi_analysis_latest"
    cache_key = "fsi_analysis_latest_hybrid_v2"
    msg = "⏳ Analysis running, please wait..."
    timestamp_label = ""
    result = cache.get(cache_key)
    if result is not None:
        msg = f"✅ Served from cache (last computed at {result.get('timestamp', 'unknown')})"
        timestamp_label = f"Last update: {result.get('timestamp', 'unknown')}"
        return result, msg, False, timestamp_label

    config = load_configuration()
    df = merge_data(config)
    if df is None:
        return dash.no_update, "❌ Data loading failed", False, ""

    fsi_series, omega_history, _, _ = estimate_fsi_recursive_rolling_with_stability(
        df,
        window_size=int(config['fsi']['window_size']),
        n_iter=int(config['fsi']['n_iter']),
        stability_threshold=float(config['fsi']['stability_threshold'])
    )

    # --- Enforce a consistent "stress is positive" orientation ---
    anchor_vars_pref = ['VIX_dev_250', 'MOVE_dev_250', 'HY_OAS_dev_250', 'IG_OAS_dev_250']
    anchors = [c for c in anchor_vars_pref if c in df.columns]

    if anchors:
        # Option A: use weights sign on anchors at the most recent date
        anchor_sign = np.sign(omega_history[anchors].iloc[-1].mean())
        if anchor_sign < 0:
            fsi_series *= -1
            omega_history *= -1
    else:
        # Option B (fallback): use correlation with an easy stress proxy if present
        proxy_candidates = [c for c in ['VIX_dev_250', 'HY_OAS_dev_250'] if c in df.columns]
        if proxy_candidates:
            proxy = df[proxy_candidates].mean(axis=1)
            aligned = pd.concat([fsi_series, proxy], axis=1).dropna()
            if not aligned.empty and aligned.corr().iloc[0,1] < 0:  # corr(FSI, proxy) < 0 ⇒ flip
                fsi_series *= -1
                omega_history *= -1
    # --- end orientation ---

    latest_omega = omega_history.iloc[-1]
    variable_contribs = compute_variable_contributions(df.loc[fsi_series.index], latest_omega)
    group_map = {
        "Volatility": [
            "VIX_dev_250", "MOVE_dev_250", "OVX_dev_250",
            "VIX3M_dev_250", "VIX_VIX3M_spread_dev_250"  # if engineered
        ],
        "Rates": [
            "2Y_rate_250", "10Y_3M_slope_dev_250", "10Y_rate_250"
        ],
        "Funding": [
            "3M_TBill_stress_250", "EFFR_stress_250" # include USD only if DXY fetched , "EFFR_VOLUME_250"
        ],
        "Credit": [
            "IG_OAS_dev_250", "HY_OAS_dev_250", "BBB_OAS_dev_250", "HY_IG_spread_250"
        ],
        "FX/Safe_Haven": [
            "Gold_dev_250", "USDJPY_dev_250", "USD_stress_250" 
        ],
    }
    grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)
    regimes_full = classify_adaptive_regime_hybrid_fallback(variable_contribs['FSI'], quantile_window=1260)
    # regimes_full = classify_risk_regime_hybrid(variable_contribs['FSI'])

    df_aligned = df.loc[fsi_series.index].copy()
    df_aligned["Regime"] = regimes_full.astype(str)

    print("First 10 regimes:", df_aligned["Regime"].head(10).tolist())
    print("Regime counts:", df_aligned["Regime"].value_counts())

    result = {
        "fsi_series": fsi_series.to_json(date_format="iso", orient="split"),
        "variable_contribs": variable_contribs.to_json(date_format="iso", orient="split"),
        "grouped_contribs": grouped_contribs.to_json(date_format="iso", orient="split"),
        "df": df_aligned.to_json(date_format="iso", orient="split"),
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    cache.set(cache_key, result, expire=3600)
    msg = f"✅ Analysis completed and cached at {result['timestamp']}"
    timestamp_label = f"Last update: {result['timestamp']}"
    return result, msg, False, timestamp_label

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
    ],
    [
        Input('fsi-store', 'data'),
        Input('fsi-date-range', 'start_date'),
        Input('fsi-date-range', 'end_date'),
        Input('fsi-yaxis-ticks', 'value')
    ]
)

def update_all_from_store(data, start_date, end_date, ytick_opts):
    if data is None:
        raise dash.exceptions.PreventUpdate

    variable_contribs = pd.read_json(io.StringIO(data["variable_contribs"]), orient="split")
    grouped_contribs  = pd.read_json(io.StringIO(data["grouped_contribs"]), orient="split")
    df_all            = pd.read_json(io.StringIO(data["df"]), orient="split")  # contains 'Regime'

    # --- USE PRECOMPUTED REGIMES (STATIC) ---
    regimes_full = df_all["Regime"].astype(str)

    # --- FILTER by selected date ---
    df_filtered = df_all.copy()
    if start_date:
        sd = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered.index >= sd]
    if end_date:
        ed = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered.index <= ed]

    # Make all inputs share the same index (NO label leakage)
    idx = df_filtered.index
    variable_contribs = variable_contribs.reindex(idx)  # numeric data can be NaN, it's fine
    grouped_contribs  = grouped_contribs.reindex(idx)

    # Use the already-computed labels, just realigned; DO NOT fill across slice edges
    regimes_filtered = regimes_full.reindex(idx)

    # --- Pass regimes to plotting (so ribbons don’t change) ---
    fig1 = plot_group_contributions_with_regime(variable_contribs, regimes=regimes_filtered)
    fig2 = plot_grouped_contributions(grouped_contribs, regimes=regimes_filtered)

    # --- Y-Axis Tick Visibility ---
    show_ticks = 'show' in (ytick_opts or [])
    fig1.update_yaxes(showticklabels=show_ticks)
    fig2.update_yaxes(showticklabels=show_ticks)

    curr_regime = get_current_regime(df_all)
    curr_regime_html = regime_color_text(curr_regime)

    # --- Improved: Map HMM state to regime color ---
    hmm_state, _, hmm_states_series = run_hmm(
        df_all, n_states=4,
        columns=[c for c in df_all.columns if 'FSI' in c or 'dev' in c or 'stress' in c or 'OAS' in c]
    )

    def hmm_state_to_regime(state):
        mapping = {0: "Green", 1: "Yellow", 2: "Amber", 3: "Red"}
        return mapping.get(state, f"Unknown ({state})")

    hmm_regime = hmm_state_to_regime(hmm_state)
    hmm_regime_html = regime_color_text(hmm_regime)

    def make_prob_gauge(prob, label):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
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
        fig.update_layout(
            margin=dict(l=10, r=10, t=45, b=14),
            paper_bgcolor="#f7f8fa",
            height=210
        )
        return fig

    prob_logit, _, _, _, score_logit = predict_regime_probability(df_all, model_type='logit', lookahead=20)
    prob_xgb, _, _, _, score_xgb = predict_regime_probability(df_all, model_type='xgboost', lookahead=20)
    fig_prob_logit = make_prob_gauge(prob_logit, f"Logit P(Red) (AUC: {score_logit:.2f})")
    fig_prob_xgb = make_prob_gauge(prob_xgb, f"XGBoost P(Red) (AUC: {score_xgb:.2f})")

    regime_series = df_all['Regime'].astype(str).fillna("NA").reset_index(drop=True)

    valid_idx = regime_series != "NA"
    regime_series = regime_series[valid_idx]
    if regime_series.nunique() < 2:
        fig_matrix = go.Figure()
        fig_matrix.update_layout(
            title="Only one regime found in data (no regime changes).",
            plot_bgcolor="#f7f8fa", paper_bgcolor="#f7f8fa",
            xaxis_visible=False, yaxis_visible=False
        )
    else:
        matrix = compute_transition_matrix(regime_series)
        regimes = list(REGIME_COLORS.keys())
        matrix = matrix.reindex(index=regimes, columns=regimes, fill_value=0)
        off_diag = matrix.values.copy()
        np.fill_diagonal(off_diag, 0)
        off_diag_sum = off_diag.sum()
        if off_diag_sum < 1e-8:
            fig_matrix = go.Figure()
            fig_matrix.update_layout(
                title="Only self-transitions detected (no regime changes in sample).",
                plot_bgcolor="#f7f8fa", paper_bgcolor="#f7f8fa",
                xaxis_visible=False, yaxis_visible=False
            )
        else:
            z = matrix.values
            x = list(matrix.columns)
            y = list(matrix.index)
            hovertext = [[f"From <b>{y[i]}</b> to <b>{x[j]}</b>: {z[i][j]:.2%}" for j in range(len(x))] for i in range(len(y))]
            fig_matrix = go.Figure(data=go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale='RdYlGn',
                reversescale=True,
                hoverinfo='text',
                text=hovertext,
                zmin=0, zmax=1,
                colorbar=dict(title="Prob.")
            ))
            fig_matrix.update_layout(
                title="<b>Regime Transition Matrix<br>(Rows: FROM, Cols: TO)</b><br>",
                xaxis_title="To Regime",
                yaxis_title="From Regime",
                margin=dict(l=25, r=25, t=45, b=40),
                font=dict(size=12),
                plot_bgcolor="#f7f8fa", paper_bgcolor="#f7f8fa"
            )

    avg_time = average_time_in_regime(regime_series)
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
                    }) for reg in ["Green", "Yellow", "Amber", "Red"]
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(f"{avg_time.get(reg, 0):.1f}", style={
                        "padding": "8px 18px",
                        "fontWeight": "500",
                        "fontSize": "1.09em"
                    }) for reg in ["Green", "Yellow", "Amber", "Red"]
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

    return fig1, fig2, curr_regime_html, hmm_regime_html, fig_prob_logit, fig_prob_xgb, fig_matrix, avg_time_table


# --- 3. PnL Upload Logic (now supports CSV and preview, error feedback) ---
@app.callback(
    [Output('fig-pnl', 'figure'),
     Output('upload-message', 'children'),
     Output('pnl-preview', 'children')],
    [Input('upload-pnl', 'contents')],
    [State('upload-pnl', 'filename'), State('fsi-store', 'data')]
)
def update_pnl(upload_contents, upload_filename, fsi_data):
    if fsi_data is None:
        return go.Figure(), "Please run analysis first.", ""

    variable_contribs = pd.read_json(io.StringIO(fsi_data["variable_contribs"]), orient="split")
    fsi_series        = pd.read_json(io.StringIO(fsi_data["fsi_series"]), typ="series", orient="split")
    df_all            = pd.read_json(io.StringIO(fsi_data["df"]), orient="split")  # <- has Regime
    regimes_full      = df_all["Regime"].astype(str)

    msg = ""
    pnl_df = None
    preview_table = None

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

                # Create preview table
                preview_table = dash_table.DataTable(
                    data=pnl_df.reset_index()[['Date', 'P/L']].head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in ['Date', 'P/L']],
                    style_table={'maxWidth': '450px'},
                    style_cell={'font-family': 'Segoe UI, Arial', 'textAlign': 'center'},
                    style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
                )
                msg = "PnL file loaded. Preview below."

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
        Output('pnl-date-range', 'min_date_allowed'),
        Output('pnl-date-range', 'max_date_allowed'),
        Output('pnl-date-range', 'start_date'),
        Output('pnl-date-range', 'end_date'),
    ],
    Input('upload-pnl', 'contents'),
    State('upload-pnl', 'filename')
)
def set_datepicker_limits(upload_contents, upload_filename):
    if not upload_contents:
        return None, None, None, None
    content_type, content_string = upload_contents.split(',')
    decoded = base64.b64decode(content_string)
    if upload_filename.lower().endswith('.csv'):
        pnl_df = pd.read_csv(io.BytesIO(decoded))
    else:
        pnl_df = pd.read_excel(io.BytesIO(decoded))
    pnl_df.columns = [c.strip() for c in pnl_df.columns]
    col_map = {c.lower(): c for c in pnl_df.columns}
    if not {'date', 'p/l'}.issubset(col_map):
        return None, None, None, None
    pnl_df['Date'] = pd.to_datetime(pnl_df[col_map['date']])
    min_date = pnl_df['Date'].min().date()
    max_date = pnl_df['Date'].max().date()
    return min_date, max_date, min_date, max_date

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
     Output('dist-pnl-full', 'figure')],
    [
        Input('upload-pnl', 'contents'),
        Input('pnl-date-range', 'start_date'),
        Input('pnl-date-range', 'end_date')
    ],
    State('upload-pnl', 'filename')
)
def update_pnl_distributions(upload_contents, start_date, end_date, upload_filename):
    if not upload_contents:
        return go.Figure(), go.Figure()
    import io, base64
    content_type, content_string = upload_contents.split(',')
    decoded = base64.b64decode(content_string)
    if upload_filename.lower().endswith('.csv'):
        pnl_df = pd.read_csv(io.BytesIO(decoded))
    else:
        pnl_df = pd.read_excel(io.BytesIO(decoded))
    pnl_df.columns = [c.strip() for c in pnl_df.columns]
    col_map = {c.lower(): c for c in pnl_df.columns}
    if not {'date', 'p/l'}.issubset(col_map):
        return go.Figure(), go.Figure()
    pnl_df['Date'] = pd.to_datetime(pnl_df[col_map['date']])
    pnl_df = pnl_df.sort_values('Date')
    # Filter by selected date range
    if start_date and end_date:
        mask = (pnl_df['Date'] >= pd.to_datetime(start_date)) & (pnl_df['Date'] <= pd.to_datetime(end_date))
        pnl_df = pnl_df.loc[mask]
    pnl_series = pnl_df[col_map['p/l']]
    if not pnl_df.empty:
        date0 = pnl_df['Date'].min().strftime("%b-%Y")
        date1 = pnl_df['Date'].max().strftime("%b-%Y")
        period_title = f"{date0} to {date1}"
    else:
        period_title = ""
    fig_range = plot_distribution_plotly(pnl_series, period_title, pnl_range=(-0.03, 0.03))
    fig_full = plot_distribution_plotly(pnl_series, period_title)
    return fig_range, fig_full


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
