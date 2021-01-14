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
import helpers as h
import preprocess as p
from reviewmodel import ReviewLDA

# EXTERNAL RESOURCES
external_stylesheets = [dbc.themes.YETI]

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

application = app.server

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

    return html.Div([
        html.Div(children=reviews_df.to_json(), id='reviews_df', style={'display': 'none'}),
        html.Div(children=sma_df.to_json(), id='sma_df', style={'display': 'none'}),
        html.Button('Get Sentiment', id='button')
    ])

def generate_eda_figs(reviews_df, sma_df):

    reviews_df = pd.read_json(reviews_df)
    sma_df = pd.read_json(sma_df)

    neg, pos = h.pos_neg_split(reviews_df)

    # Negative Reviews LDA
    neg_preprocessed = p.preprocess_corpus(neg['title_desc'])
    neg_lda = ReviewLDA()
    neg_lda.fit(neg_preprocessed)
    neg_review_weights = neg_lda.transform(neg_lda.dtm)

    neg_topics = h.display_topics(neg_lda, neg_lda.tfidf.get_feature_names(), 20)
    neg_topics_df = pd.DataFrame(neg_topics)
    neg_review_weights_df = pd.DataFrame(neg_review_weights).rename(columns={0: 'Topic 1', 1: 'Topic 2', 2: 'Topic 3'})

    neg_topic_reviews = pd.merge(neg_review_weights_df, neg.reset_index(), left_index=True, right_index=True)
    print(neg_topic_reviews)
    neg_topic1_reviews = neg_topic_reviews.sort_values('Topic 1', ascending=False)[0:5][['Topic 1','asin','desc']]

    # Positive Reviews LDA
    pos_preprocessed = p.preprocess_corpus(pos['title_desc'])
    pos_lda = ReviewLDA()
    pos_lda.fit(pos_preprocessed)

    rating_hist = px.histogram(reviews_df, x="rating", nbins=5, title = 'Histogram of Ratings')
    rating_ot = px.line(sma_df, x='date', y='moving', title='Simple Moving Average Rating')
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

    neg_topics_fig = go.Figure()
    neg_topics_fig.add_trace(
        go.Bar(x=list(neg_topics_df['Topic 1 words']),
                y=list(neg_topics_df['Topic 1 weights']),
                name="Topic 1"))

    neg_topics_fig.add_trace(
        go.Bar(x=list(neg_topics_df['Topic 2 words']),
                y=list(neg_topics_df['Topic 2 weights']),
                name="Topic 2"))

    neg_topics_fig.add_trace(
        go.Bar(x=list(neg_topics_df['Topic 3 words']),
                y=list(neg_topics_df['Topic 3 weights']),
                name="Topic 3"))

    neg_topics_fig.update_layout(
        updatemenus=[
            dict(
                active=3,
                buttons=list([
                    dict(label="Topic 1",
                        method="update",
                        args=[{"visible": [True, False, False]},
                            {"title": "Topic 1"}]),
                    dict(label="Topic 2",
                        method="update",
                        args=[{"visible": [False, True, False]},
                            {"title": "Topic 2"}]),
                    dict(label="Topic 3",
                        method="update",
                        args=[{"visible": [False, False, True]},
                            {"title": "Topic 3"}]),
                    dict(label="All Topics",
                        method="update",
                        args=[{"visible": [True, True, True]},
                            {"title": "All Topics"}])
                ]),
            )
        ])

    neg_topics_table_1 = dash_table.DataTable(
        id='neg-topic-1-reviews',
        columns=[{"name": i, "id": i} for i in neg_topic1_reviews.columns],
        data=neg_topic1_reviews.to_dict('records'),
        style_cell={
        'whiteSpace': 'normal',
        'height': 'auto',
        'width': '400px'
        },
        css=[{
        'selector': '.dash-spreadsheet td div',
        'rule': '''
            line-height: 15px;
            max-height: 30px; min-height: 30px; height: 30px;
            display: block;
            overflow-y: hidden;
        '''
        }],
    )

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
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id='neg-topics',
                    figure=neg_topics_fig
                ), style=dict(width='100%', backgroundColor='#FFF')
            )
        ),
        dbc.Row(
            dbc.Col(
                children=neg_topics_table_1,
                width={'size': 6, 'offset': 2},
                
            )
        )
    ])

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
            [Input('button', 'n_clicks'),
             Input('reviews_df', 'children'),
             Input('sma_df', 'children')])
def update_button(n_clicks, reviews_df, sma_df):
    if n_clicks:
        return generate_eda_figs(reviews_df, sma_df)

if __name__ == '__main__':
    app.run_server(debug=True)