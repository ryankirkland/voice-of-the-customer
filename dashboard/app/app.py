import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_dangerously_set_inner_html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pyLDAvis.sklearn

import pandas as pd
import numpy as np
import json
import base64
import datetime
import io
import time
import src.helpers as h
import src.preprocess as p
from src.reviewmodel import ReviewLDA

df = pd.read_csv('../../data/cleaned_reviews.csv')
dr = pd.read_csv('../../data/date_range.csv')
lda_vis = 'negative_rechargeble_bats_ldavis.html'

test_df = pd.DataFrame({
    'yeet': [1, 2, 3,]
})

# EXTERNAL RESOURCES
external_stylesheets = [dbc.themes.YETI]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
        dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    dbc.Col(dbc.NavbarBrand("Amazon Reviews Sentiment Analysis - View the Repo", className="ml-2")),
                    align="center",
                    no_gutters=True,
                ),
                href="https://github.com/ryankirkland/voice-of-the-customer",
                target='_blank'
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
                            'sentiment of subsets of Amazon customer reviews. Drop a file below to see what customers are saying.',
                            className='lead'
                        )
                    ], width=5),
                    dbc.Col([
                        html.Img(src='assets/amz_box.png')
                    ], width=5)
                ]),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files to Get Started')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    # Allow multiple files to be uploaded
                    multiple=True
                ),
                html.Div(id='output-data-upload')
            ],
            fluid=False
            )
        ],
        fluid=True
        ),
        html.Div(id='output-container-button')
    ],
    fluid=True,
    style=dict(padding=0)
)

def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    reviews_df = h.cleaned_reviews_dataframe(df)
    reviews_df = h.get_sentiment(reviews_df)
    sma_df = h.get_moving_average(reviews_df)

    # reviews_df.to_csv('assets/reviews.csv')
    # sma_df.to_csv('assets/sma.csv')

    return html.Button('Get Sentiment', id='button')

    # return html.Div([
    #     html.Div(children=reviews_df.to_json(), id='reviews_df'),
    #     html.Div(children=sma_df.to_json(), id='sma_df')
    # ])

def generate_eda_figs():

    reviews_df = pd.read_csv('assets/reviews.csv')
    sma_df = pd.read_csv('assets/sma.csv')

    rating_hist = px.histogram(reviews_df, x="rating", nbins=5, title = 'Histogram of Ratings')
    rating_ot = px.line(sma_df, x=dr['dates'], y=dr['moving'], title='Simple Moving Average Rating')
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

    sentiment = px.scatter(reviews_df, 
                    x='Polarity', 
                    y='Subjectivity', 
                    color = 'Analysis',
                    size='Subjectivity')

    return html.Div([
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
    ])

    # #add a vertical line at x=0 for Netural Reviews
    # sentiment.update_layout(title='Sentiment Analysis of Review Content',
    #                 shapes=[dict(type= 'line',
    #                             yref= 'paper', y0= 0, y1= 1, 
    #                             xref= 'x', x0= 0, x1= 0)])
    
    # neg, pos = h.pos_neg_split(reviews_df)

    # neg_preprocessed = p.preprocess_corpus(neg['title_desc'])
    # neg_lda = ReviewLDA()
    # neg_lda.fit(neg_preprocessed)
    # neg_vis = pyLDAvis.sklearn.prepare(neg_lda.best_lda, neg_lda.dtm, neg_lda.tfidf, n_jobs=1)
    # neg_lda_vis = pyLDAvis.prepared_data_to_html(neg_vis)
    # # neg_html = open('assets/neg.html', 'w')
    # # neg_html.write(neg_lda_vis)
    # neg_lda_vis = 'neg.html'





    #     # dbc.Row(
    #     #     dbc.Col([
    #     #         html.Iframe(src=app.get_asset_url(neg_lda_vis))
    #     #     ])
    #     # ),

    #     html.Hr(),  # horizontal line
    # ])

@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output('output-container-button', 'children'),
            [Input('button', 'n_clicks')])
            #  Input('reviews_df', 'children'),
            #  Input('sma_df', 'children')])
def update_button(n_clicks):
    if n_clicks:
        return generate_eda_figs()

if __name__ == '__main__':
    app.run_server(debug=True)