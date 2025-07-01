# app.py

# import dash
# from dash import dcc, html, Input, Output, State, ctx
# import plotly.graph_objs as go
# import pandas as pd
# import base64
# import io
# import diskcache
# cache = diskcache.Cache("./cache-directory")

# # --- Your framework imports ---
# from main import load_configuration, merge_data
# from fsi_estimation import estimate_fsi_recursive_rolling_with_stability, compute_variable_contributions
# from plotting import (
#     plot_group_contributions_with_regime,
#     plot_grouped_contributions,
#     plot_pnl_with_regime_ribbons,
# )
# from utils import (
#     aggregate_contributions_by_group,
#     get_current_regime, run_hmm, predict_regime_probability, compute_transition_matrix,
# )

# app = dash.Dash(__name__)
# server = app.server

# app.layout = html.Div([
#     html.H1("Financial Stress Dashboard"),
#     html.Button("Run/Refresh Analysis", id="run-btn", n_clicks=0),
#     html.Span(id='run-message', style={'color': 'green', 'margin-left': '20px'}),
#     dcc.Store(id='fsi-store'),   # Hidden component for pipeline results
#     html.Div([
#         html.H2("Variable-Level FSI"),
#         dcc.Graph(id='fig1'),
#         html.H2("Group-Level FSI"),
#         dcc.Graph(id='fig2'),
#         html.H2("PnL Chart with Regime Ribbons"),
#         dcc.Graph(id='fig-pnl'),
#         html.Div([
#             html.Label("Upload PnL Excel File (.xlsx with 'Date' and 'P/L'):"), 
#             dcc.Upload(
#                 id='upload-pnl',
#                 children=html.Button('Upload PnL Excel'),
#                 accept='.xlsx',
#                 multiple=False,
#             ),
#             html.Span(id='upload-message', style={'color': 'red', 'margin-left': '20px'})
#         ], style={'margin-bottom': '30px'}),
#     ], style={'width': '90%', 'margin': 'auto'}),
#     html.Hr(),
#     html.Div([
#         html.H2("Forward-Looking & Regime Risk Metrics"),
#         html.Div([
#             html.Div([
#                 html.H4("Current Regime (Rule-Based):"),
#                 html.Div(id='current-regime', style={'font-size': '1.7em', 'font-weight': 'bold'})
#             ], style={'display': 'inline-block', 'width': '23%', 'vertical-align': 'top', 'text-align': 'center'}),
#             html.Div([
#                 html.H4("Current HMM Market State:"),
#                 html.Div(id='current-hmm', style={'font-size': '1.7em', 'font-weight': 'bold'})
#             ], style={'display': 'inline-block', 'width': '23%', 'vertical-align': 'top', 'text-align': 'center'}),
#             html.Div([
#                 html.H4("Probability of 'Red' Regime (Logit):"),
#                 dcc.Graph(id='prob-red-logit', config={'displayModeBar': False}, style={'height': '150px'})
#             ], style={'display': 'inline-block', 'width': '25%', 'vertical-align': 'top', 'text-align': 'center'}),
#             html.Div([
#                 html.H4("Probability of 'Red' Regime (XGBoost):"),
#                 dcc.Graph(id='prob-red-xgb', config={'displayModeBar': False}, style={'height': '150px'})
#             ], style={'display': 'inline-block', 'width': '25%', 'vertical-align': 'top', 'text-align': 'center'}),
#         ], style={'width': '100%', 'margin-bottom': '25px'}),
#         html.H4("Historical Regime Transition Matrix"),
#         dcc.Graph(id='regime-transition-matrix'),
#     ], style={'width': '90%', 'margin': 'auto'}),
# ], style={'font-family': 'Arial, sans-serif'})

# # --- 1. RUN/REFRESH BUTTON: Pipeline Callback ---
# @app.callback(
#     [Output('fsi-store', 'data'), Output('run-message', 'children')],
#     Input('run-btn', 'n_clicks'),
#     prevent_initial_call=True
# )
# def run_full_pipeline(n_clicks):
#     import time
#     config = load_configuration()
#     df = merge_data(config)
#     if df is None:
#         return dash.no_update, "❌ Data loading failed"

#     fsi_series, omega_history, _, _ = estimate_fsi_recursive_rolling_with_stability(
#         df,
#         window_size=int(config['fsi']['window_size']),
#         n_iter=int(config['fsi']['n_iter']),
#         stability_threshold=float(config['fsi']['stability_threshold'])
#     )
#     latest_omega = omega_history.iloc[-1]
#     variable_contribs = compute_variable_contributions(df.loc[fsi_series.index], latest_omega)
#     group_map = {
#         'Volatility': ['VIX_dev_250', 'MOVE_dev_250', 'OVX_dev_250', 'VXV_dev_250', 'VIX_VXV_spread_dev_250'],
#         'Rates': ['10Y_rate_250', '2Y_rate_250', '10Y_2Y_slope_dev_250', '10Y_3M_slope_dev_250', 'USDO_rate_dev_250'],
#         'Funding': ['USD_stress_250', '3M_TBill_stress_250', 'Fed_RRP_stress_250', 'FRED_RRP_stress_250'],
#         'Credit': ['IG_OAS_dev_250', 'HY_OAS_dev_250', 'HY_IG_spread_250'],
#         'Safe_Haven': ['Gold_dev_250']
#     }
#     grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)
#     # Store everything as JSON for Dash
#     result = {
#         "fsi_series": fsi_series.to_json(date_format="iso", orient="split"),
#         "variable_contribs": variable_contribs.to_json(date_format="iso", orient="split"),
#         "grouped_contribs": grouped_contribs.to_json(date_format="iso", orient="split"),
#         "df": df.to_json(date_format="iso", orient="split"),
#         # Add more as needed
#     }
#     return result, f"✅ Analysis completed at {time.strftime('%H:%M:%S')}"

# # --- 2. Update Main Charts/Stats When Data Is Available ---
# @app.callback(
#     [
#         Output('fig1', 'figure'),
#         Output('fig2', 'figure'),
#         Output('current-regime', 'children'),
#         Output('current-hmm', 'children'),
#         Output('prob-red-logit', 'figure'),
#         Output('prob-red-xgb', 'figure'),
#         Output('regime-transition-matrix', 'figure'),
#     ],
#     Input('fsi-store', 'data')
# )
# def update_all_from_store(data):
#     if data is None:
#         raise dash.exceptions.PreventUpdate
#     import pandas as pd
#     from plotting import plot_group_contributions_with_regime, plot_grouped_contributions
#     from utils import get_current_regime, run_hmm, predict_regime_probability, compute_transition_matrix

#     variable_contribs = pd.read_json(data["variable_contribs"], orient="split")
#     grouped_contribs = pd.read_json(data["grouped_contribs"], orient="split")
#     df = pd.read_json(data["df"], orient="split")

#     fig1 = plot_group_contributions_with_regime(variable_contribs)
#     fig2 = plot_grouped_contributions(grouped_contribs)

#     curr_regime = get_current_regime(df)
#     hmm_state, _, _ = run_hmm(df, n_states=4, columns=[c for c in df.columns if 'FSI' in c or 'dev' in c or 'stress' in c or 'OAS' in c])
#     hmm_state_str = f"State {hmm_state}"

#     prob_logit, _, _ = predict_regime_probability(df, model_type='logit', lookahead=20)
#     prob_xgb, _, _ = predict_regime_probability(df, model_type='xgboost', lookahead=20)

#     def make_prob_gauge(prob, label):
#         fig = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=prob * 100,
#             domain={'x': [0, 1], 'y': [0, 1]},
#             title={'text': label},
#             gauge={
#                 'axis': {'range': [0, 100]},
#                 'bar': {'color': "crimson" if prob > 0.6 else "gold" if prob > 0.3 else "limegreen"},
#                 'steps': [
#                     {'range': [0, 30], 'color': "lightgreen"},
#                     {'range': [30, 60], 'color': "yellow"},
#                     {'range': [60, 100], 'color': "pink"}
#                 ],
#             },
#             number={'suffix': "%"}
#         ))
#         fig.update_layout(margin=dict(l=15, r=15, t=35, b=15))
#         return fig

#     fig_prob_logit = make_prob_gauge(prob_logit, "Logit P(Red in 20d)")
#     fig_prob_xgb = make_prob_gauge(prob_xgb, "XGBoost P(Red in 20d)")

#     trans_matrix = compute_transition_matrix(df['Regime'])
#     regimes = ['Green', 'Yellow', 'Amber', 'Red']
#     trans_matrix = trans_matrix.reindex(index=regimes, columns=regimes, fill_value=0)
#     z = trans_matrix.values
#     x = list(trans_matrix.columns)
#     y = list(trans_matrix.index)
#     hovertext = [[f"From <b>{y[i]}</b> to <b>{x[j]}</b>: {z[i][j]:.2%}" for j in range(len(x))] for i in range(len(y))]
#     fig_matrix = go.Figure(data=go.Heatmap(
#         z=z,
#         x=x,
#         y=y,
#         colorscale='RdYlGn',
#         reversescale=True,
#         hoverinfo='text',
#         text=hovertext,
#         zmin=0, zmax=1,
#         colorbar=dict(title="Prob.")
#     ))
#     fig_matrix.update_layout(
#         title="Regime Transition Matrix<br>(Rows: FROM, Cols: TO)",
#         xaxis_title="To Regime",
#         yaxis_title="From Regime",
#         margin=dict(l=40, r=20, t=40, b=40)
#     )

#     return fig1, fig2, curr_regime, hmm_state_str, fig_prob_logit, fig_prob_xgb, fig_matrix

# # --- 3. PnL Upload Logic: This can stay mostly as in your original, but use the current variable_contribs and fsi_series as needed ---
# @app.callback(
#     [Output('fig-pnl', 'figure'), Output('upload-message', 'children')],
#     [Input('upload-pnl', 'contents')],
#     [State('upload-pnl', 'filename'), State('fsi-store', 'data')]
# )
# def update_pnl(upload_contents, upload_filename, fsi_data):
#     if fsi_data is None:
#         return go.Figure(), "Please run analysis first."
#     variable_contribs = pd.read_json(fsi_data["variable_contribs"], orient="split")
#     fsi_series = pd.read_json(fsi_data["fsi_series"], orient="split")
#     msg = ""
#     pnl_df = None
#     if upload_contents is not None:
#         try:
#             content_type, content_string = upload_contents.split(',')
#             decoded = base64.b64decode(content_string)
#             pnl_df = pd.read_excel(io.BytesIO(decoded))
#             pnl_df.columns = [c.strip() for c in pnl_df.columns]
#             if not {'Date', 'P/L'}.issubset(set(pnl_df.columns)):
#                 msg = "File must contain 'Date' and 'P/L' columns."
#                 pnl_df = None
#             else:
#                 pnl_df['Date'] = pd.to_datetime(pnl_df['Date'])
#                 pnl_df = pnl_df.set_index('Date')
#                 pnl_df = pnl_df.sort_index()
#         except Exception as e:
#             msg = f"Error reading Excel: {e}"
#             pnl_df = None

#     if pnl_df is not None:
#         fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series)
#     else:
#         fig_pnl = go.Figure()
#         fig_pnl.update_layout(title="PnL Chart (Upload file to see data)")
#     return fig_pnl, msg

# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 8080))
#     app.run(debug=True, host="0.0.0.0", port=port)















# app.py

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import pandas as pd
import base64
import io
import diskcache
import time

# --- Your framework imports ---
from main import load_configuration, merge_data
from fsi_estimation import estimate_fsi_recursive_rolling_with_stability, compute_variable_contributions
from plotting import (
    plot_group_contributions_with_regime,
    plot_grouped_contributions,
    plot_pnl_with_regime_ribbons,
)
from utils import (
    aggregate_contributions_by_group,
    get_current_regime, run_hmm, predict_regime_probability, compute_transition_matrix, classify_risk_regime_hybrid
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
    # --- Custom CSS for Pro Styling ---
    html.Style('''
        body, .app-container {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f4f6fa;
        }
        .metrics-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 18px;
            margin-bottom: 30px;
        }
        .card-metric {
            flex: 1 1 210px;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 12px #e5e9f2;
            padding: 18px 12px 10px 16px;
            min-width: 180px;
            margin-bottom: 0;
            transition: box-shadow 0.18s;
            border: 1px solid #e3e8ee;
        }
        .card-metric:hover {
            box-shadow: 0 4px 22px #dde2ec;
        }
        .metric-title {
            font-size: 1.08em;
            font-weight: 500;
            color: #30507d;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.45em;
            font-weight: bold;
            margin-top: 6px;
        }
        .metric-value.green { color: #15b67a; }
        .metric-value.yellow { color: #e3b112; }
        .metric-value.amber { color: #eb830d; }
        .metric-value.red { color: #e53935; }
        .regime-tag.green { color: #fff; background: #15b67a; border-radius: 6px; padding: 2px 10px; }
        .regime-tag.yellow { color: #fff; background: #e3b112; border-radius: 6px; padding: 2px 10px; }
        .regime-tag.amber { color: #fff; background: #eb830d; border-radius: 6px; padding: 2px 10px; }
        .regime-tag.red { color: #fff; background: #e53935; border-radius: 6px; padding: 2px 10px; }
        .regime-tag { margin-left: 10px; font-size: 1.05em; font-weight: 600; }
        .timestamp-label {
            font-size: 1.02em;
            color: #9da5b4;
            margin-top: 2px;
            margin-left: 10px;
        }
        .chart-container {
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 0 10px #e5e5e5;
            padding: 18px 12px 10px 14px;
            margin-bottom: 22px;
        }
        .dash-graph {
            min-height: 330px;
        }
        .dash-table-container {
            background: #f7f8fa;
            padding: 6px 10px;
            border-radius: 7px;
            box-shadow: 0 0 4px #ebedf1;
            margin-bottom: 18px;
        }
        .download-btn {
            background: #396aff;
            color: #fff;
            border-radius: 7px;
            padding: 8px 20px;
            font-size: 1em;
            font-weight: 600;
            border: none;
            margin-right: 10px;
            transition: background 0.15s;
            cursor: pointer;
        }
        .download-btn:hover {
            background: #2247b9;
        }
        .dash-uploader {
            margin: 8px 0 0 0;
            border: 1.5px dashed #abc3e9;
            border-radius: 7px;
            background: #f4f8fd;
            padding: 12px;
            text-align: center;
        }
        .dash-uploader input[type="file"] {
            display: none;
        }
        .dash-uploader .upload-text {
            color: #30507d;
            font-size: 1.04em;
            margin-bottom: 2px;
        }
        .dash-tooltip {
            background: #222c38;
            color: #fff;
            border-radius: 7px;
            font-size: 0.98em;
            padding: 6px 10px;
            box-shadow: 0 0 8px #bbb;
        }
        .dash-spinner {
            margin-top: 24px;
        }
        @media (max-width: 950px) {
            .metrics-row {
                flex-direction: column;
                gap: 12px;
            }
            .card-metric {
                width: 98% !important;
            }
        }
        @media (max-width: 600px) {
            .metrics-row {
                flex-direction: column;
                gap: 6px;
            }
            .card-metric {
                width: 98% !important;
                min-width: 130px;
                font-size: 0.98em;
            }
        }
    '''),

    # --- Header, Timestamp, Controls ---
    html.H1("Financial Stress Dashboard", style={"margin-bottom": "5px"}),
    html.Div(id='timestamp-label', className="timestamp-label", style={'font-size': '1em', "margin-bottom": "12px"}),
    html.Button("Run/Refresh Analysis", id="run-btn", n_clicks=0, disabled=False, style={"margin-bottom": "7px"}),
    html.Span(id='run-message', style={'color': 'green', 'margin-left': '20px', "font-size": "1em"}),
    dcc.Store(id='fsi-store'),

    # --- Main Chart Panels ---
    dcc.Loading(
        id="loading-fsi",
        type="circle",
        children=[
            html.Div([
                html.Div([
                    html.H2([
                        "Variable-Level FSI", 
                        info_icon("Shows each variable’s weighted contribution to the overall Financial Stress Index.")
                    ]),
                    dcc.Graph(id='fig1'),
                    html.Button("Download as Image", id="dl-fig1", n_clicks=0, className="download-btn")
                ], style={"margin-bottom": "10px"}),
                html.Div([
                    html.H2([
                        "Group-Level FSI", 
                        info_icon("Aggregated by risk group: Volatility, Rates, Credit, etc.")
                    ]),
                    dcc.Graph(id='fig2'),
                    html.Button("Download as Image", id="dl-fig2", n_clicks=0, className="download-btn")
                ]),
                html.Div([
                    html.H2([
                        "PnL Chart with Regime Ribbons", 
                        info_icon("Upload your PnL file. Regimes are highlighted along the PnL curve.")
                    ]),
                    dcc.Graph(id='fig-pnl'),
                    html.Button("Download as Image", id="dl-pnl", n_clicks=0, className="download-btn")
                ]),
                html.Div([
                    html.Label([
                        "Upload PnL file (.xlsx/.csv):", 
                        info_icon("Date and P/L columns required (case-insensitive).")
                    ]),
                    dcc.Upload(
                        id='upload-pnl',
                        children=html.Button('Upload PnL File'),
                        accept='.xlsx,.csv',
                        multiple=False,
                        className="dash-uploader"
                    ),
                    html.Div(id="pnl-preview", style={"margin": "7px 0 7px 0", "font-size": "0.95em"}),
                    html.Span(id='upload-message', style={'color': 'red', 'margin-left': '20px'})
                ], style={'margin-bottom': '30px'})
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
            html.Div([
                html.H4([
                    "Current Regime (Rule-Based):", 
                    info_icon("Classified by FSI and thresholds.")
                ]),
                html.Div(id='current-regime', style={'font-size': '1.7em', 'font-weight': 'bold', "margin-bottom": "8px"})
            ], className="card-metric"),
            html.Div([
                html.H4([
                    "Current HMM Market State:", 
                    info_icon("Market regime inferred by a Hidden Markov Model.")
                ]),
                html.Div(id='current-hmm', style={'font-size': '1.7em', 'font-weight': 'bold', "margin-bottom": "8px", "color": "#2d3436"})
            ], className="card-metric"),
            html.Div([
                html.H4([
                    "Probability of 'Red' Regime (Logit):", 
                    info_icon("20-day ahead probability, logistic regression.")
                ]),
                dcc.Graph(id='prob-red-logit', config={'displayModeBar': False}, style={'height': '140px', 'width': '98%'})
            ], className="card-metric"),
            html.Div([
                html.H4([
                    "Probability of 'Red' Regime (XGBoost):", 
                    info_icon("20-day ahead probability, XGBoost.")
                ]),
                dcc.Graph(id='prob-red-xgb', config={'displayModeBar': False}, style={'height': '140px', 'width': '98%'})
            ], className="card-metric"),
        ], className="metrics-row"),
        html.H4([
            "Historical Regime Transition Matrix", 
            info_icon("Rows: FROM regime; Cols: TO regime. Shows likelihood of switching between risk regimes.")
        ]),
        dcc.Graph(id='regime-transition-matrix'),
    ], style={'width': '95%', 'margin': 'auto'}),

    dcc.Download(id="download-image"),
], style={
    'font-family': 'Segoe UI, Arial, sans-serif',
    'background-color': '#f7f8fa',
    "padding-bottom": "35px"
})


# --- 1. RUN/REFRESH BUTTON: Pipeline Callback with Caching and Button Disable ---
@app.callback(
    [Output('fsi-store', 'data'), Output('run-message', 'children'), Output('run-btn', 'disabled'), Output('timestamp-label', 'children')],
    Input('run-btn', 'n_clicks'),
    prevent_initial_call=True
)
def run_full_pipeline(n_clicks):
    import time
    from utils import classify_risk_regime_hybrid

    cache_key = "fsi_analysis_latest"
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
    latest_omega = omega_history.iloc[-1]
    variable_contribs = compute_variable_contributions(df.loc[fsi_series.index], latest_omega)
    group_map = {
        'Volatility': ['VIX_dev_250', 'MOVE_dev_250', 'OVX_dev_250', 'VXV_dev_250', 'VIX_VXV_spread_dev_250'],
        'Rates': ['10Y_rate_250', '2Y_rate_250', '10Y_2Y_slope_dev_250', '10Y_3M_slope_dev_250', 'USDO_rate_dev_250'],
        'Funding': ['USD_stress_250', '3M_TBill_stress_250', 'Fed_RRP_stress_250', 'FRED_RRP_stress_250'],
        'Credit': ['IG_OAS_dev_250', 'HY_OAS_dev_250', 'HY_IG_spread_250'],
        'Safe_Haven': ['Gold_dev_250']
    }
    grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)
    regimes = classify_risk_regime_hybrid(fsi_series)
    df_aligned = df.loc[fsi_series.index].copy()
    df_aligned["Regime"] = regimes.values

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
    ],
    Input('fsi-store', 'data')
)
def update_all_from_store(data):
    if data is None:
        raise dash.exceptions.PreventUpdate
    variable_contribs = pd.read_json(io.StringIO(data["variable_contribs"]), orient="split")
    grouped_contribs = pd.read_json(io.StringIO(data["grouped_contribs"]), orient="split")
    df = pd.read_json(io.StringIO(data["df"]), orient="split")
    fig1 = plot_group_contributions_with_regime(variable_contribs)
    fig2 = plot_grouped_contributions(grouped_contribs)
    curr_regime = get_current_regime(df)
    curr_regime_html = regime_color_text(curr_regime)
    hmm_state, _, _ = run_hmm(df, n_states=4, columns=[c for c in df.columns if 'FSI' in c or 'dev' in c or 'stress' in c or 'OAS' in c])
    hmm_state_str = html.Span(f"State {hmm_state}", style={'color': "#636e72", "fontWeight": "bold"})
    prob_logit, _, _ = predict_regime_probability(df, model_type='logit', lookahead=20)
    prob_xgb, _, _ = predict_regime_probability(df, model_type='xgboost', lookahead=20)
    def make_prob_gauge(prob, label):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': label, "font": {"size": 15}},
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
        fig.update_layout(margin=dict(l=8, r=8, t=28, b=12), paper_bgcolor="#f7f8fa")
        return fig
    fig_prob_logit = make_prob_gauge(prob_logit, "Logit P(Red in 20d)")
    fig_prob_xgb = make_prob_gauge(prob_xgb, "XGBoost P(Red in 20d)")
    trans_matrix = compute_transition_matrix(df['Regime'])
    regimes = list(REGIME_COLORS.keys())
    trans_matrix = trans_matrix.reindex(index=regimes, columns=regimes, fill_value=0)
    z = trans_matrix.values
    x = list(trans_matrix.columns)
    y = list(trans_matrix.index)
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
        title="Regime Transition Matrix<br>(Rows: FROM, Cols: TO)",
        xaxis_title="To Regime",
        yaxis_title="From Regime",
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="#f7f8fa", paper_bgcolor="#f7f8fa"
    )
    return fig1, fig2, curr_regime_html, hmm_state_str, fig_prob_logit, fig_prob_xgb, fig_matrix

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
    fsi_series = pd.read_json(io.StringIO(fsi_data["fsi_series"]), typ="series", orient="split")
    msg = ""
    pnl_df = None
    preview_table = None
    if upload_contents is not None:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            if upload_filename.lower().endswith('.csv'):
                pnl_df = pd.read_csv(io.BytesIO(decoded))
            else:
                pnl_df = pd.read_excel(io.BytesIO(decoded))
            pnl_df.columns = [c.strip() for c in pnl_df.columns]
            # Allow case-insensitive match for date/pl
            col_map = {c.lower(): c for c in pnl_df.columns}
            if not {'date', 'p/l'}.issubset(set(col_map)):
                msg = "File must contain 'Date' and 'P/L' columns."
                pnl_df = None
            else:
                pnl_df['Date'] = pd.to_datetime(pnl_df[col_map['date']])
                pnl_df['P/L'] = pnl_df[col_map['p/l']]
                pnl_df = pnl_df.set_index('Date')
                pnl_df = pnl_df.sort_index()
                preview_table = dash_table.DataTable(
                    data=pnl_df.reset_index().head(5).to_dict('records'),
                    columns=[{"name": i, "id": i} for i in pnl_df.reset_index().columns],
                    style_table={'maxWidth': '450px'},
                    style_cell={'font-family': 'Segoe UI, Arial', 'textAlign': 'center'},
                    style_header={'backgroundColor': '#f2f2f2', 'fontWeight': 'bold'},
                )
                msg = "PnL file loaded. Preview below."
        except Exception as e:
            msg = f"Error reading file: {e}"
            pnl_df = None

    if pnl_df is not None:
        fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series)
    else:
        fig_pnl = go.Figure()
        fig_pnl.update_layout(title="PnL Chart (Upload file to see data)")
    return fig_pnl, msg, preview_table

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

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
