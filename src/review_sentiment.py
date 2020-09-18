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

#Create function to get subjectivity score
def subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#Create function to get the polarity score
def polarity(text): 
    return TextBlob(text).sentiment.polarity

#Create function to classify based on polarity score
def getAnalysis(score):
    if score <0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

#Bring all functions together to run on df
def get_sentiment(df):
    df['Subjectivity'] = df['title_desc'].apply(subjectivity)
    df['Polarity'] = df['title_desc'].apply(polarity)
    df['Analysis'] = df['title_desc'].apply(getAnalysis)
    return df