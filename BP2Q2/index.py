import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
from app import app
import s1, s2, s3, s4, l1, l2, l3, l4

server = app.server

app.layout = html.Div([
    dcc.Location(id='url'),
    html.Div(
        [
            html.H2('Net-Load Forecasting', className='display-6'),
            html.Hr(),
            html.P('A Dash app for data visualization', className='lead'),
            dbc.Nav(
                [
                    dbc.NavLink('Home', href='/', active='exact'),
                    dbc.NavLink('Small Data Set 1', href='/s1', active='exact'),
                    dbc.NavLink('Small Data Set 2', href='/s2', active='exact'),
                    dbc.NavLink('Small Data Set 3', href='/s3', active='exact'),
                    dbc.NavLink('Small Data Set 4', href='/s4', active='exact'),
                    dbc.NavLink('Large Data Set 1', href='/l1', active='exact'),
                    dbc.NavLink('Large Data Set 2', href='/l2', active='exact'),
                    dbc.NavLink('Large Data Set 3', href='/l3', active='exact'),
                    dbc.NavLink('Large Data Set 4', href='/l4', active='exact')
                ],
                pills=True,
                vertical=True
            )
        ],
        style={
            'position': 'fixed',
            'top': 0,
            'left': 0,
            'bottom': 0,
            'width': '16rem',
            'padding': '2rem 1rem',
            'background-color': '#f8f9fa'
        }
    ),
    html.Div(
        id='page-content',
        style={
            'margin-left': '18rem',
            'margin-right': '2rem',
            'padding': '2rem 1rem'
        }
    )
])

markdown_text = '''
### Data Visualization Platform for Net-Load Forecasting

**Project Title**: Day-Ahead Probabilistic Forecasting of Net-Load and Demand Response Potentials with High Penetration of
Behind-the-Meter Solar-plus-Storage

**Agency**: U.S. Department of Energy

**Award Number**: DE-EE0009357

**Project Period**: 06/01/2021–05/31/2024

**Lead Organization**: NC State University

**Team Member Organizations**: Temple University, North Carolina Electric Membership Corporation (NCEMC), Dominion Energy

This data visualization platform meets the milestone 1.3.1, which implements the following features: graphical displays of data,
summary statistics, correlation analysis, and evaluation of forecasts. If you have any questions or comments,
please contact the Principal Investigator [Wenyuan Tang](https://people.engr.ncsu.edu/wtang8/).
'''

layout = html.Div([
    dcc.Markdown(markdown_text),
    html.Br(),
    html.Img(height='70', src='/assets/doe.png'),
    html.Img(height='70', src='/assets/ncsu.png', style={'margin-left': '2rem'}),
    html.Img(height='70', src='/assets/temple.svg', style={'margin-left': '2rem'}),
    html.Img(height='70', src='/assets/ncemc.png', style={'margin-left': '2rem'}),
    html.Img(height='70', src='/assets/dominion.svg', style={'margin-left': '2rem'})
])

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page_content(pathname):        
    if pathname == '/s1':
        return s1.layout
    elif pathname == '/s2':
        return s2.layout
    elif pathname == '/s3':
        return s3.layout
    elif pathname == '/s4':
        return s4.layout
    elif pathname == '/l1':
        return l1.layout
    elif pathname == '/l2':
        return l2.layout
    elif pathname == '/l3':
        return l3.layout
    elif pathname == '/l4':
        return l4.layout
    else:
        return layout

if __name__ == '__main__':
    app.run_server(debug=True)
