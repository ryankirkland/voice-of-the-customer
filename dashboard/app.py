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

df = pd.read_csv('../../data/cleaned.csv')

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
])

if __name__ == '__main__':
    app.run_server(debug=True)