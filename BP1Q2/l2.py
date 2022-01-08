import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from app import app
import base64
import io, os
import numpy as np

markdown_text = '''
### L2: 5-Year Data of 8 Areas in ERCOT

The data are collected from the [ERCOT] (http://www.ercot.com/gridinfo/load/load_hist/).
ERCOT does not provide solar data, the solar data are also collected from [PVWatts] (https://pvwatts.nrel.gov/). 
The temperature data are collected from [NOAA] (https://www.noaa.gov/). 
The 8 selected areas are COAST, EAST, FWEST, NCENT, NORTH, SCENT, SOUTH, and WEST.
Data from 01/01/2015 to 12/31/2018 are used for training, and data from 01/01/2019 to 12/31/2019 are used for testing.
'''

DATA_PATH = os.path.join( "data/l2")
df = pd.DataFrame(columns=[ "Location", "Year", "Month", "Day", "Weekday", "Hour","Demand", "Temperature"])
for root, dirs, files in os.walk(DATA_PATH):
    date = pd.read_csv(os.path.join(root, "Date.csv"))
    month_series = date["Month"]
    curr_year = 2015
    year_series = [0 for i in month_series]

    for i in range(len(month_series)-1):
        if month_series[i+1] >= month_series[i]:
            year_series[i] += curr_year
        else:
            year_series[i] += curr_year
            curr_year+=1

    year_series[-1] += curr_year

    for file in files:
        if file != "Date.csv" and file != "fcst.csv" and file != "test.csv":
            t = pd.read_csv(os.path.join(root, file))
            if "Demand" in t.columns:
                temp_df = pd.DataFrame(data = t["Demand"])
            else:
                temp_df = pd.DataFrame(data = t["Net"])
                temp_df["Demand"] = temp_df["Net"]
                temp_df.drop(columns=["Net"], inplace = True)
            
            temp_df["Location"] = file.split(".")[0]
            temp_df["Year"] = year_series
            temp_df["Month"] = month_series
            temp_df["Day"] = date["Day"]
            temp_df["Hour"] = date["Hour"]
            temp_df["Weekday"] = date["Weekday"]
            temp_df["Temperature"] = t["Temperature"]   
            
            df = pd.concat([df, temp_df])
                
df['DateTime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])                
df = df.set_index('DateTime')                
df.rename(columns={'Demand': 'Load'}, inplace=True)       
df_daily = df.groupby(['Location'])[['Load']].resample('D').mean().reset_index(level=['Location'])   
locations = df.Location.unique()

layout = html.Div([
    dcc.Markdown(markdown_text),
    dcc.Markdown('**Time Series of Load**'),
    dcc.RadioItems(
        id='l2-load-ts-res',
        inputStyle={'margin-left': '1rem', 'margin-right': '0.5rem'},
        options=[{'label': i, 'value': i} for i in ['Hourly', 'Daily']],
        style={'margin-left': '-1rem'}
    ),
    dcc.Graph(id='l2-load-ts'),
    dcc.Markdown('**Summary Statistics of Load**'),
    dcc.RadioItems(
        id='l2-load-stat-by',
        inputStyle={'margin-left': '1rem', 'margin-right': '0.5rem'},
        options=[{'label': i, 'value': i} for i in ['By hour', 'By month']],
        style={'margin-left': '-1rem'}
    ),
    dcc.Graph(id='l2-load-stat'),
    dcc.Markdown('**Correlation Analysis**'),
    dcc.RadioItems(
        id='l2-load-temp-by',
        inputStyle={'margin-left': '1rem', 'margin-right': '0.5rem'},
        options=[{'label': i, 'value': i} for i in ['By hour', 'By month']],
        style={'margin-left': '-1rem'}
    ),
    dcc.Graph(id='l2-load-temp'),
    dcc.Markdown('**Evaluation of Forecasts**'),
    dcc.Markdown('Select the range of the test set, and download the test set or upload your forecasts.'),
    dcc.DatePickerRange(
        id='l2-eval-test',
        min_date_allowed='2015-01-01',
        max_date_allowed='2019-12-31',
        start_date='2019-01-01',
        end_date='2019-12-31',
    ),
    html.Div(
        dbc.Button('Download', id='l2-download'),
        style={'display': 'inline-block',"margin-left": "20px"}
        ),
    dcc.Download(id='l2-download-csv'),
    html.Div(    
        dcc.Upload(
            id='l2-upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
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
        style={'display': 'inline-block',"margin-left": "20px"}
        ),
    dcc.Graph(id='l2-eval')
])

@app.callback(Output('l2-load-ts', 'figure'), Input('l2-load-ts-res', 'value'), prevent_initial_call=True)
def update_l2_load_ts(res):
    if res == 'Hourly':
        fig = px.line(df, x=df.index, y='Load', color='Location')
    else:
        fig = px.line(df_daily, x=df_daily.index, y='Load', color='Location')
    fig.update_layout(margin={'t': 50},xaxis_title="Datetime",yaxis_title = "Load (kW)")
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1 month', step='month', stepmode='backward'),
                dict(count=3, label='3 months', step='month', stepmode='backward'),
                dict(count=6, label='6 months', step='month', stepmode='backward'),
                dict(count=12, label='1 year', step='month', stepmode='backward'),
                dict(step='all')
            ])
        )
    )
    return fig

@app.callback(Output('l2-load-stat', 'figure'), Input('l2-load-stat-by', 'value'), prevent_initial_call=True)
def update_l2_load_stat(res):
    if res == 'By hour':     
        fig = px.box(df, x='Hour', y='Load', color='Location', labels={'Load': 'Load (kW)'})
    else:
        fig = px.box(df, x='Month', y='Load', color='Location', labels={'Load': 'Load (kW)'})
    fig.update_layout(margin={'t': 30}, boxmode='group',xaxis_title="Datetime",yaxis_title = "Load (kW)")
    return fig

@app.callback(Output('l2-load-temp', 'figure'), Input('l2-load-temp-by', 'value'), prevent_initial_call=True)
def update_l2_load_temp(res):
    if res == 'By hour':
        fig = px.scatter(
            df, x='Temperature', y='Load', color='Location',
            facet_col='Hour', facet_col_wrap=6, facet_row_spacing=0.03, height=900,
            labels={'Temperature': 'Temperature (\u00b0C)', 'Load': 'Load (kW)'},
            category_orders={'Hour': list(range(24))}
        )
    else:
        fig = px.scatter(
            df, x='Temperature', y='Load', color='Location',  facet_col='Month', facet_col_wrap=6, height=500,
            labels={'Temperature': 'Temperature (\u00b0C)', 'Load': 'Load (kW)'}
        )
    fig.update_layout(margin={'t': 30})
    return fig

@app.callback(
    Output('l2-download-csv', 'data'),
    Input('l2-download', 'n_clicks'),
    State('l2-eval-test', 'start_date'),
    State('l2-eval-test', 'end_date'),
    prevent_initial_call=True
)
def update_l2_download(n_clicks, start_date, end_date):
    df_test = df[start_date:end_date]
    return dcc.send_data_frame(df_test.to_csv, 'test.csv')

@app.callback(
    Output('l2-eval', 'figure'),
    Input('l2-upload', 'contents'),
    State('l2-eval-test', 'start_date'),
    State('l2-eval-test', 'end_date'),
    prevent_initial_call=True
)
def update_l2_upload(contents, start_date, end_date):
    df_test = df[start_date:end_date].copy()
    if contents is None:
        fig = px.line(df_test, x=df_test.index, y='Load', color='Location', labels={'Load': 'Load (kW)'}, height=500)
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
        fig = px.line(df_eval, x=df_eval.index, y='Load', color='Location',line_dash = 'set', labels={'Load': 'Load (kW)'}, title = title, height=500)
        fig.update_layout(legend_title_text=None)
    return fig
