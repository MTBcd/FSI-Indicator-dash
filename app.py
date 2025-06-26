# app.py

import dash
from dash import dcc, html, Input, Output, State
from dash import dash_table
import plotly.graph_objs as go
import pandas as pd
import base64
import io

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
    get_current_regime, run_hmm, predict_regime_probability, compute_transition_matrix,
)

# ---- Load and preprocess FSI data ONCE at app startup ----
config = load_configuration()
df = merge_data(config)

fsi_series, omega_history, _, _ = estimate_fsi_recursive_rolling_with_stability(
    df,
    window_size=int(config['fsi']['window_size']),
    n_iter=int(config['fsi']['n_iter']),
    stability_threshold=float(config['fsi']['stability_threshold'])
)
latest_omega = omega_history.iloc[-1]
variable_contribs = compute_variable_contributions(df.loc[fsi_series.index], latest_omega)

group_map = {
    'Volatility': ['VIX_dev', 'VXV_dev', 'OVX_dev', 'GVZ_dev'],
    'Rates': ['10Y_rate', '1Y_rate', 'Yield_slope', 'USDO_rate_dev'],
    'Funding': ['USD_stress', '3M_TBill_stress', 'Fed_RRP_stress'],
    'Credit': ['Credit_spread', 'Corp_OAS_dev', 'HY_OAS_dev'],
}
grouped_contribs = aggregate_contributions_by_group(variable_contribs, group_map)

# ---- Dash App Layout ----
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Financial Stress Dashboard"),
    html.Div([
        html.Div([
            html.Label("Upload PnL Excel File (.xlsx with 'Date' and 'P/L'):"),
            dcc.Upload(
                id='upload-pnl',
                children=html.Button('Upload PnL Excel'),
                accept='.xlsx',
                multiple=False,
            ),
            html.Button('Refresh', id='refresh-btn', n_clicks=0, style={'margin-left': '20px'}),
            html.Span(id='upload-message', style={'color': 'red', 'margin-left': '20px'})
        ], style={'margin-bottom': '30px'}),
        html.Div([
            html.H2("Variable-Level FSI"),
            dcc.Graph(id='fig1', figure=plot_group_contributions_with_regime(variable_contribs)),
            html.H2("Group-Level FSI"),
            dcc.Graph(id='fig2', figure=plot_grouped_contributions(grouped_contribs)),
            html.H2("PnL Chart with Regime Ribbons"),
            dcc.Graph(id='fig-pnl'),
        ])
    ], style={'width': '90%', 'margin': 'auto'}),
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

# ---- Callback for uploading PnL and running metrics ----

@app.callback(
    [
        Output('fig-pnl', 'figure'),
        Output('upload-message', 'children'),
        Output('current-regime', 'children'),
        Output('current-hmm', 'children'),
        Output('prob-red-logit', 'figure'),
        Output('prob-red-xgb', 'figure'),
        Output('regime-transition-matrix', 'figure'),
    ],
    [
        Input('refresh-btn', 'n_clicks'),
        Input('upload-pnl', 'contents')
    ],
    [
        State('upload-pnl', 'filename')
    ]
)
def update_all(refresh_n, upload_contents, upload_filename):
    # --- Use PnL upload if available, else use None for empty chart ---
    pnl_df = None
    msg = ""
    if upload_contents is not None:
        try:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            pnl_df = pd.read_excel(io.BytesIO(decoded))
            # Format columns
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

    # --- PnL plot (only if upload succeeded) ---
    if pnl_df is not None:
        fig_pnl = plot_pnl_with_regime_ribbons(pnl_df, variable_contribs, fsi_series)
    else:
        # Empty/default chart
        fig_pnl = go.Figure()
        fig_pnl.update_layout(title="PnL Chart (Upload file to see data)")

    # --- Risk Metrics ---
    # Use regime on latest available data
    curr_regime = get_current_regime(df)
    hmm_state, _, _ = run_hmm(df, n_states=4, columns=[c for c in df.columns if 'FSI' in c or 'dev' in c or 'stress' in c or 'OAS' in c])
    hmm_state_str = f"State {hmm_state}"

    # Regime probability prediction
    prob_logit, _, _ = predict_regime_probability(df, model_type='logit', lookahead=20)
    prob_xgb, _, _ = predict_regime_probability(df, model_type='xgboost', lookahead=20)

    # --- Probability as Gauge/Bar ---
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

    # --- Transition Matrix ---
    trans_matrix = compute_transition_matrix(df['Regime'])
    # Fill all possible regimes for consistency
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

    return (
        fig_pnl, msg, curr_regime, hmm_state_str, fig_prob_logit, fig_prob_xgb, fig_matrix
    )

# ---- Run app ----
if __name__ == '__main__':
    app.run_server(debug=True)
