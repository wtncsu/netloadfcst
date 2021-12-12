import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import numpy as np
import pandas as pd
import base64
import io
from app import app

markdown_text = '''
### S2: 1-Year Data of 20 Houses in the Chicago Area

The data are collected from the SunDance dataset in the [UMass Trace Repository](http://traces.cs.umass.edu).
The SunDance dataset includes hourly energy data (net meter and solar generation) and weather data
(weather condition data from public weather stations and APIs) for 100 solar sites in North America from 01/01/2015 to 12/31/2015.
We consider the aggregated net load as our target. Data from 01/01/2015 to 10/31/2015 are used for training,
and the remaining data are used for testing.
'''

df = pd.read_csv('data/s2/data.csv')
df['Year'] = 2015
df['DateTime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
df = df.set_index('DateTime')
df.rename(columns={'Demand': 'Load'}, inplace=True)
df_daily = df[['Load']].resample('D').mean()

layout = html.Div([
    dcc.Markdown(markdown_text),
    dcc.Markdown('**Time Series of Load**'),
    dcc.RadioItems(
        id='s2-load-ts-res',
        inputStyle={'margin-left': '1rem', 'margin-right': '0.5rem'},
        options=[{'label': i, 'value': i} for i in ['Hourly', 'Daily']],
        style={'margin-left': '-1rem'}
    ),
    dcc.Graph(id='s2-load-ts'),
    dcc.Markdown('**Summary Statistics of Load**'),
    dcc.RadioItems(
        id='s2-load-stat-by',
        inputStyle={'margin-left': '1rem', 'margin-right': '0.5rem'},
        options=[{'label': i, 'value': i} for i in ['By hour', 'By month']],
        style={'margin-left': '-1rem'}
    ),
    dcc.Graph(id='s2-load-stat'),
    dcc.Markdown('**Correlation Analysis**'),
    dcc.RadioItems(
        id='s2-load-temp-by',
        inputStyle={'margin-left': '1rem', 'margin-right': '0.5rem'},
        options=[{'label': i, 'value': i} for i in ['By hour', 'By month']],
        style={'margin-left': '-1rem'}
    ),
    dcc.Graph(id='s2-load-temp'),
    dcc.Markdown('**Evaluation of Forecasts**'),
    dcc.Markdown('Select the range of the test set, and download the test set or upload your forecasts.'),
    dcc.DatePickerRange(
        id='s2-eval-test',
        min_date_allowed='2015-01-01',
        max_date_allowed='2015-12-31',
        start_date='2015-11-01',
        end_date='2015-12-31'
    ),
    html.Div(
        dbc.Button('Download', id='s2-download'),
        style={'display': 'inline-block', 'margin-left': '20px'}
    ),
    dcc.Download(id='s2-download-csv'),
    html.Div(
        dcc.Upload(
            'Drag and Drop or Select a File',
            id='s2-upload',
            style={
                'width': '200%',
                'height': '50px',
                'lineHeight': '50px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            }
        ),
        style={'display': 'inline-block', 'margin-left': '20px'}
    ),
    dcc.Graph(id='s2-eval')
])

@app.callback(Output('s2-load-ts', 'figure'), Input('s2-load-ts-res', 'value'), prevent_initial_call=True)
def update_s2_load_ts(res):
    if res == 'Hourly':
        fig = px.line(df, x=df.index, y='Load', labels={'Load': 'Load (kW)'})
    else:
        fig = px.line(df_daily, x=df_daily.index, y='Load', labels={'Load': 'Load (kW)'})
    fig.update_layout(margin={'t': 50})
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1 month', step='month', stepmode='backward'),
                dict(count=3, label='3 months', step='month', stepmode='backward'),
                dict(count=6, label='6 months', step='month', stepmode='backward'),
                dict(step='all')
            ])
        )
    )
    return fig

@app.callback(Output('s2-load-stat', 'figure'), Input('s2-load-stat-by', 'value'), prevent_initial_call=True)
def update_s2_load_stat(res):
    if res == 'By hour':
        fig = px.box(df, x='Hour', y='Load', labels={'Load': 'Load (kW)'})
    else:
        fig = px.box(df, x='Month', y='Load', labels={'Load': 'Load (kW)'})
    fig.update_layout(margin={'t': 30})
    return fig

@app.callback(Output('s2-load-temp', 'figure'), Input('s2-load-temp-by', 'value'), prevent_initial_call=True)
def update_s2_load_temp(res):
    if res == 'By hour':
        fig = px.scatter(
            df, x='Temperature', y='Load', facet_col='Hour', facet_col_wrap=6, facet_row_spacing=0.03, height=900,
            labels={'Temperature': 'Temperature (\u00b0C)', 'Load': 'Load (kW)'},
            category_orders={'Hour': list(range(24))}
        )
    else:
        fig = px.scatter(
            df, x='Temperature', y='Load', facet_col='Month', facet_col_wrap=6, height=500,
            labels={'Temperature': 'Temperature (\u00b0C)', 'Load': 'Load (kW)'}
        )
    fig.update_layout(margin={'t': 30})
    return fig

@app.callback(
    Output('s2-download-csv', 'data'),
    Input('s2-download', 'n_clicks'),
    State('s2-eval-test', 'start_date'),
    State('s2-eval-test', 'end_date'),
    prevent_initial_call=True
)
def update_s2_download(n_clicks, start_date, end_date):
    df_test = df[start_date:end_date]
    return dcc.send_data_frame(df_test.to_csv, 'test.csv')

@app.callback(
    Output('s2-eval', 'figure'),
    Input('s2-upload', 'contents'),
    State('s2-eval-test', 'start_date'),
    State('s2-eval-test', 'end_date'),
    prevent_initial_call=True
)
def update_s2_upload(contents, start_date, end_date):
    df_test = df[start_date:end_date].copy()
    if contents is None:
        fig = px.line(df_test, x=df_test.index, y='Load', labels={'Load': 'Load (kW)'})
    else:
        df_test['set'] = 'Actual'
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df_fcst = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df_fcst = df_fcst.set_index('DateTime')
        df_fcst['set'] = 'Forecasted'
        mape = np.mean(np.absolute((df_test['Load'].to_numpy() - df_fcst['Load'].to_numpy()) / df_test['Load'].max()))
        title = 'Mean absolute percentage error (MAPE): {:.2%}'.format(mape)
        df_eval = pd.concat([df_test, df_fcst])
        print(df_eval.shape)
        fig = px.line(df_eval, x=df_eval.index, y='Load', color='set', labels={'Load': 'Load (kW)'}, title = title)
        fig.update_layout(legend_title_text=None)
    return fig
