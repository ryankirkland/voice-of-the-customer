import numpy as np
import pandas as pd
from textblob import TextBlob

def cleaned_reviews_dataframe(reviews_df):
    """
    Remove newline "\n" from titles and descriptions,
    as well as the "Unnamed: 0" column generated when
    loading DataFrame from CSV. This is the only cleaning
    required prior to NLP preprocessing.
    
    INPUT: Pandas DataFrame with 'title' and 'desc' column names
    
    OUTPUT: Cleaned DataFrame with combined 'title_desc' column
    """
    reviews_df['title'] = reviews_df['title'].str.replace('\n', '')
    reviews_df['desc'] = reviews_df['desc'].str.replace('\n','')
    reviews_df['title_desc'] = reviews_df['title'] + reviews_df['desc']
    if 'Unnamed: 0' in set(reviews_df.columns):
        reviews_df = reviews_df.drop('Unnamed: 0', axis=1)
    return reviews_df

def get_moving_average(df, window=30):
    """
    Create date range from earliest and latest dates in
    DataFrame to create a new DataFrame for time series
    visualization.
    
    INPUT: Pandas DataFrame with 'date' column name
    
    OUTPUT: New DataFrame with 'dates' as index and mean
    grouped data.
    """

    # Group daily reviews by the mean to get the average of a day's reviews
    date_group = df.groupby('date').mean()
    date_group = date_group.reset_index()
    date_group['date'] = pd.to_datetime(date_group['date'])
    date_group = date_group.sort_values('date')

    # Calculate the simple moving average over the specified window to monitor short-term impact to customer response
    date_group['moving'] = date_group.rating.rolling(window, min_periods=1).mean()
    dates = pd.date_range(date_group.index[0], date_group.index[-1])
    dates_df = pd.DataFrame(dates).rename(columns={0: 'dates'})
    date_df = dates_df.merge(date_group, how='outer', left_on='dates', right_on='date')

    # Forward fill the days that do not have reviews with the previous value
    date_df = date_df.fillna(method='ffill')
    date_df = date_df.drop('dates', axis=1)
    return date_df

#Create function to get subjectivity score
def subjectivity(text):
    return np.round(TextBlob(text).sentiment.subjectivity, 2)

#Create function to get the polarity score
def polarity(text): 
    return np.round(TextBlob(text).sentiment.polarity, 2)

#Create function to classify based on polarity score
def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

#Bring all functions together to run on df
def get_sentiment(df):
    """
    Processes the cleaned DataFrame of Amazon customer reviews
    using the TextBlob package to reclassify reviews as
    Positive, Negative, or Neutral based on written
    content instead of the actual star rating.

    Returns the DataFrame with 'Subjectivity',
    'Polarity', and 'Analysis' columns for vis.

    INPUT: Pandas DataFrame
    OUTPUT: Pandas DataFrame
    """
    df['Subjectivity'] = df['title_desc'].apply(subjectivity)
    df['Polarity'] = df['title_desc'].apply(polarity)
    df['Analysis'] = df['Polarity'].apply(getAnalysis)
    return df

def pos_neg_split(df):
    """
    Splits DataFrame into two separate positive and
    negative DataFrames for the creation of two
    separate models for LDAvis.

    INPUT: Sentiment-analyzed DataFrame

    OUTPUT: A positive DataFrame and negative DataFrame
    """
    neg = df[df['Analysis'] == 'Negative']
    pos = df[df['Analysis'] == 'Positive']
    return neg, pos

def display_topics(model, feature_names, n_top_words):
    '''
    INPUTS:
        model - the model we created
        feature_names - tells us what word each column in the matric represents
        n_top_words - number of top words to display

    OUTPUTS:
        a dataframe that contains the topics we created and the weights of each token
    '''
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx+1)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx+1)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-n_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)