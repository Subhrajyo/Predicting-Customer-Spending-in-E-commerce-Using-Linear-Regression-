import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Load the customer data
customers = pd.read_csv('C:/Users/HP/Downloads/Ecommerce Customers.csv')

# Create a linear regression model
lm = LinearRegression()
lm.fit(customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']], customers['Yearly Amount Spent'])

app = dash.Dash(__name__)

app.layout = html.Div([
    # Header
    html.Div([
        html.H1('Ecommerce Customer Prediction', style={'textAlign': 'center', 'color': '#007bff', 'fontSize': '36px', 'marginBottom': '20px'}),
        html.P('Predict the yearly amount spent by ecommerce customers based on their session length, time on app, time on website, and length of membership.', style={'textAlign': 'center', 'color': '#666', 'fontSize': '18px', 'marginBottom': '20px'})
    ], style={'backgroundColor': '#f7f7f7', 'padding': '20px', 'borderBottom': '1px solid #ccc'}),

    # Main Content
    html.Div([
        # Make a Prediction
        html.Div([
            html.H2('Make a Prediction', style={'textAlign': 'center', 'color': '#007bff', 'fontSize': '24px', 'marginBottom': '10px'}),
            html.Label('Avg. Session Length:', style={'marginRight': '10px'}),
            dcc.Input(id='avg-session-length', type='number', style={'width': '20%', 'height': '30px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
            html.Label('Time on App:', style={'marginRight': '10px'}),
            dcc.Input(id='time-on-app', type='number', style={'width': '20%', 'height': '30px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
            html.Label('Time on Website:', style={'marginRight': '10px'}),
            dcc.Input(id='time-on-website', type='number', style={'width': '20%', 'height': '30px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
            html.Label('Length of Membership:', style={'marginRight': '10px'}),
            dcc.Input(id='length-of-membership', type='number', style={'width': '20%', 'height': '30px', 'padding': '10px', 'borderRadius': '5px', 'border': '1px solid #ccc'}),
            html.Button('Make Prediction', id='make-prediction-button', style={'backgroundColor': '#007bff', 'color': 'white', 'padding': '10px 20px', 'borderRadius': '5px', 'border': 'none', 'cursor': 'pointer'})
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'alignItems': 'center', 'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)'}),

        # Prediction Result
        html.Div([
            html.H2('Prediction Result', style={'textAlign': 'center', 'color': '#007bff', 'fontSize': '24px', 'marginBottom': '10px'}),
            html.Div(id='prediction-result', style={'fontSize': '24px', 'fontWeight': 'bold', 'color': '#007bff'})
        ], style={'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)'}),

        # Error Metrics
        html.Div([
            html.H2('Error Metrics', style={'textAlign': 'center', 'color': '#007bff', 'fontSize': '24px', 'marginBottom': '10px'}),
            html.Div(id='error-metrics', style={'fontSize': '18px', 'color': '#666'})
        ], style={'padding': '20px', 'border': '1px solid #ccc', 'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)'}),

    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'alignItems': 'center', 'padding': '20px'}),

    # Footer
    html.Div([
        html.P('Copyright 2023 Ecommerce Customer Prediction. All rights reserved.', style={'textAlign': 'center', 'color': '#666', 'fontSize': '14px', 'marginBottom': '20px'})
    ], style={'backgroundColor': '#f7f7f7', 'padding': '20px', 'borderTop': '1px solid #ccc'})
])

@app.callback(
    Output('prediction-result', 'children'),
    [Input('make-prediction-button', 'n_clicks')],
    [State('avg-session-length', 'value'),
     State('time-on-app', 'value'),
     State('time-on-website', 'value'),
     State('length-of-membership', 'value')]
)
def make_prediction(n_clicks, avg_session_length, time_on_app, time_on_website, length_of_membership):
    X = pd.DataFrame({'Avg. Session Length': [avg_session_length], 'Time on App': [time_on_app], 'Time on Website': [time_on_website], 'Length of Membership': [length_of_membership]})
    y_pred = lm.predict(X)
    return f'Predicted Yearly Amount Spent: ${y_pred[0]:.2f}'

@app.callback(
    Output('error-metrics', 'children'),
    [Input('make-prediction-button', 'n_clicks')]
)
def update_error_metrics(n_clicks):
    y_test = customers['Yearly Amount Spent']
    predictions = lm.predict(customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']])
    mean_absolute_error_val = mean_absolute_error(y_test, predictions)
    mean_squared_error_val = mean_squared_error(y_test, predictions)
    root_mean_squared_error_val = math.sqrt(mean_squared_error_val)
    return f'Mean Absolute Error: ${mean_absolute_error_val:.2f}<br>Mean Squared Error: ${mean_squared_error_val:.2f}<br>Root Mean Squared Error: ${root_mean_squared_error_val:.2f}'

if __name__ == '__main__':
    app.run_server()