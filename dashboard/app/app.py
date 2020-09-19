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
lda_vis = 'negative_rechargeble_bats_ldavis.html'

rating_hist = px.histogram(df, x="rating", nbins=5, title = 'Histogram of Ratings')
rating_ot = px.line(dr, x=dr['dates'], y=dr['moving'], title='Simple Moving Average Rating')
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

sentiment = px.scatter(df, 
                 x='Polarity', 
                 y='Subjectivity', 
                 color = 'Analysis',
                 size='Subjectivity')

#add a vertical line at x=0 for Netural Reviews
sentiment.update_layout(title='Sentiment Analysis of Review Content',
                  shapes=[dict(type= 'line',
                               yref= 'paper', y0= 0, y1= 1, 
                               xref= 'x', x0= 0, x1= 0)])

# EXTERNAL RESOURCES
external_stylesheets = [dbc.themes.YETI]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
        dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    dbc.Col(dbc.NavbarBrand("Product Sentiment Analysis Dashboard", className="ml-2")),
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
        dbc.Jumbotron([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1('Voice of the Customer', className='display-5'),
                        html.P(
                            'The following visualizations provide insight into customer '
                            'sentiment of the rechargeable batteries Amazon '
                            'subcategory.',
                            className='lead'
                        )
                    ], width=5),
                    dbc.Col([
                        html.Img(src='assets/amz_box.png')
                    ], width=5)
                ])
            ],
            fluid=False
            )
        ],
        fluid=True
        ),
        dbc.Row([
            dbc.Col(
                dcc.Graph(
                id='rating-hist',
                figure=rating_hist,
                style=dict(width='100%')
                ), width=5
            ),
            dbc.Col(
                dcc.Graph(
                id='sentiment',
                figure=sentiment
                ), width=7
            )
        ]),
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                id='rating-ot',
                figure=rating_ot
                ), style=dict(width='100%', backgroundColor='#FFF')
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Iframe(src=app.get_asset_url(lda_vis),style=dict(width="100%", height="900px", paddingLeft='10%', textAlign='center'))
            )
        )
    ],
    fluid=True,
    style=dict(padding=0)
)

if __name__ == '__main__':
    app.run_server(debug=True)