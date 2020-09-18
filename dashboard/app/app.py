import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import pandas as pd
import numpy as np
import time

df = pd.read_csv('../../data/cleaned_reviews.csv')
dr = pd.read_csv('../../data/date_range.csv')
lda_vis = 'lda.html'

rating_hist = px.histogram(df, x="rating", nbins=5, title = 'Histogram of Ratings')
rating_ot = px.line(dr, x=dr['dates'], y=dr['moving'])
rating_ot.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# EXTERNAL RESOURCES
external_stylesheets = [dbc.themes.YETI]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
        dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    dbc.Col(dbc.NavbarBrand("Fraud Dashboard", className="ml-2")),
                    align="center",
                    no_gutters=True,
                ),
                href="#",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
        ],
        color="dark",
        dark=True,
        ),
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                id='rating-hist',
                figure=rating_hist
                ), style=dict(width='49%')
            ),
            dbc.Col(
                dcc.Graph(
                id='rating-hist2',
                figure=rating_hist
                ), style=dict(width='49%')
            )
        ]),
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                id='rating-ot',
                figure=rating_ot
                ), style=dict(width='100%')
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Iframe(src=app.get_asset_url(lda_vis),style=dict(width="100%", height="900px", paddingLeft='5%', textAlign='center'))
            )
        )
])

if __name__ == '__main__':
    app.run_server(debug=True)