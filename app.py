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
from dash import dcc, html, Input, Output, State
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

cache = diskcache.Cache("./cache-directory")

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Financial Stress Dashboard"),
    html.Button("Run/Refresh Analysis", id="run-btn", n_clicks=0, disabled=False),
    html.Span(id='run-message', style={'color': 'green', 'margin-left': '20px'}),
    dcc.Store(id='fsi-store'),
    dcc.Loading(
        id="loading-fsi",
        type="circle",
        children=[
            html.Div([
                html.H2("Variable-Level FSI"),
                dcc.Graph(id='fig1'),
                html.H2("Group-Level FSI"),
                dcc.Graph(id='fig2'),
                html.H2("PnL Chart with Regime Ribbons"),
                dcc.Graph(id='fig-pnl'),
                html.Div([
                    html.Label("Upload PnL Excel File (.xlsx with 'Date' and 'P/L'):"),
                    dcc.Upload(
                        id='upload-pnl',
                        children=html.Button('Upload PnL Excel'),
                        accept='.xlsx',
                        multiple=False,
                    ),
                    html.Span(id='upload-message', style={'color': 'red', 'margin-left': '20px'})
                ], style={'margin-bottom': '30px'}),
            ], style={'width': '90%', 'margin': 'auto'})
        ]
    ),
    html.Hr(),
    html.Div([
        html.H2("Forward-Looking & Regime Risk Metrics"),
        html.Div([
            html.Div([
                html.H4("Current Regime (Rule-Based):"),
                html.Div(id='current-regime', style={'font-size': '1.7em', 'font-weight': 'bold'})
            ], style={'display': 'inline-block', 'width': '23%', 'vertical-align': 'top', 'text-align': 'center'}),
            html.Div([
                html.H4("Current HMM Market State:"),
                html.Div(id='current-hmm', style={'font-size': '1.7em', 'font-weight': 'bold'})
            ], style={'display': 'inline-block', 'width': '23%', 'vertical-align': 'top', 'text-align': 'center'}),
            html.Div([
                html.H4("Probability of 'Red' Regime (Logit):"),
                dcc.Graph(id='prob-red-logit', config={'displayModeBar': False}, style={'height': '150px'})
            ], style={'display': 'inline-block', 'width': '25%', 'vertical-align': 'top', 'text-align': 'center'}),
            html.Div([
                html.H4("Probability of 'Red' Regime (XGBoost):"),
                dcc.Graph(id='prob-red-xgb', config={'displayModeBar': False}, style={'height': '150px'})
            ], style={'display': 'inline-block', 'width': '25%', 'vertical-align': 'top', 'text-align': 'center'}),
        ], style={'width': '100%', 'margin-bottom': '25px'}),
        html.H4("Historical Regime Transition Matrix"),
        dcc.Graph(id='regime-transition-matrix'),
    ], style={'width': '90%', 'margin': 'auto'}),
], style={'font-family': 'Arial, sans-serif'})

# --- 1. RUN/REFRESH BUTTON: Pipeline Callback with Caching and Button Disable ---
@app.callback(
    [Output('fsi-store', 'data'), Output('run-message', 'children'), Output('run-btn', 'disabled')],
    Input('run-btn', 'n_clicks'),
    prevent_initial_call=True
)

def run_full_pipeline(n_clicks):
    import time
    from utils import classify_risk_regime_hybrid

    cache_key = "fsi_analysis_latest"
    # Disable button immediately
    msg = "⏳ Analysis running, please wait..."

    # Try cache first
    result = cache.get(cache_key)
    if result is not None:
        msg = f"✅ Served from cache (last computed at {result.get('timestamp', 'unknown')})"
        # Re-enable button
        return result, msg, False

    # If not cached, run full pipeline
    config = load_configuration()
    df = merge_data(config)
    if df is None:
        return dash.no_update, "❌ Data loading failed", False

    # Estimate FSI, get rolling weights
    fsi_series, omega_history, _, _ = estimate_fsi_recursive_rolling_with_stability(
        df,
        window_size=int(config['fsi']['window_size']),
        n_iter=int(config['fsi']['n_iter']),
        stability_threshold=float(config['fsi']['stability_threshold'])
    )
    latest_omega = omega_history.iloc[-1]
    variable_contribs = compute_variable_contributions(df.loc[fsi_series.index], latest_omega)

    # Define group mapping
    group_map = {
        'Volatility': ['VIX_dev_250', 'MOVE_dev_250', 'OVX_dev_250', 'VXV_dev_250', 'VIX_VXV_spread_dev_250'],
        'Rates': ['10Y_rate_250', '2Y_rate_250', '10Y_2Y_slope_dev_250', '10Y_3M_slope_dev_250', 'USDO_rate_dev_250'],
        'Funding': ['USD_stress_250', '3M_TBill_stress_250', 'Fed_RRP_stress_250', 'FRED_RRP_stress_250'],
        'Credit': ['IG_OAS_dev_250', 'HY_OAS_dev_250', 'HY_IG_spread_250'],
        'Safe_Haven': ['Gold_dev_250']
    }
    grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)

    # === THE FIX: Add regime classification to dataframe ===
    regimes = classify_risk_regime_hybrid(fsi_series)
    df_aligned = df.loc[fsi_series.index].copy()
    df_aligned["Regime"] = regimes.values  # Ensure "Regime" column is present

    # Prepare results dict for dcc.Store
    result = {
        "fsi_series": fsi_series.to_json(date_format="iso", orient="split"),
        "variable_contribs": variable_contribs.to_json(date_format="iso", orient="split"),
        "grouped_contribs": grouped_contribs.to_json(date_format="iso", orient="split"),
        "df": df_aligned.to_json(date_format="iso", orient="split"),  # This df has the Regime column
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    cache.set(cache_key, result, expire=3600)
    msg = f"✅ Analysis completed and cached at {result['timestamp']}"
    return result, msg, False

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
    variable_contribs = pd.read_json(data["variable_contribs"], orient="split")
    grouped_contribs = pd.read_json(data["grouped_contribs"], orient="split")
    df = pd.read_json(data["df"], orient="split")
    fig1 = plot_group_contributions_with_regime(variable_contribs)
    fig2 = plot_grouped_contributions(grouped_contribs)
    curr_regime = get_current_regime(df)
    hmm_state, _, _ = run_hmm(df, n_states=4, columns=[c for c in df.columns if 'FSI' in c or 'dev' in c or 'stress' in c or 'OAS' in c])
    hmm_state_str = f"State {hmm_state}"
    prob_logit, _, _ = predict_regime_probability(df, model_type='logit', lookahead=20)
    prob_xgb, _, _ = predict_regime_probability(df, model_type='xgboost', lookahead=20)
    def make_prob_gauge(prob, label):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': label},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "crimson" if prob > 0.6 else "gold" if prob > 0.3 else "limegreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "pink"}
                ],
            },
            number={'suffix': "%"}
        ))
        fig.update_layout(margin=dict(l=15, r=15, t=35, b=15))
        return fig
    fig_prob_logit = make_prob_gauge(prob_logit, "Logit P(Red in 20d)")
    fig_prob_xgb = make_prob_gauge(prob_xgb, "XGBoost P(Red in 20d)")
    trans_matrix = compute_transition_matrix(df['Regime'])
    regimes = ['Green', 'Yellow', 'Amber', 'Red']
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
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig1, fig2, curr_regime, hmm_state_str, fig_prob_logit, fig_prob_xgb, fig_matrix

# --- 3. PnL Upload Logic (unchanged) ---
@app.callback(
    [Output('fig-pnl', 'figure'), Output('upload-message', 'children')],
    [Input('upload-pnl', 'contents')],
    [State('upload-pnl', 'filename'), State('fsi-store', 'data')]
)
def update_pnl(upload_contents, upload_filename, fsi_data):
    if fsi_data is None:
        return go.Figure(), "Please run analysis first."
    variable_contribs = pd.read_json(fsi_data["variable_contribs"], orient="split")
    fsi_series = pd.read_json(fsi_data["fsi_series"], orient="split")
    msg = ""
    pnl_df = None
    if upload_contents is not None:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            pnl_df = pd.read_excel(io.BytesIO(decoded))
            pnl_df.columns = [c.strip() for c in pnl_df.columns]
            if not {'Date', 'P/L'}.issubset(set(pnl_df.columns)):
                msg = "File must contain 'Date' and 'P/L' columns."
                pnl_df = None
            else:
                pnl_df['Date'] = pd.to_datetime(pnl_df['Date'])
                pnl_df = pnl_df.set_index('Date')
                pnl_df = pnl_df.sort_index()
        except Exception as e:
            msg = f"Error reading Excel: {e}"
            pnl_df = None
    if pnl_df is not None:
        fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series)
    else:
        fig_pnl = go.Figure()
        fig_pnl.update_layout(title="PnL Chart (Upload file to see data)")
    return fig_pnl, msg

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
