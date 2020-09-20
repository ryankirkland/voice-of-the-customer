import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


nltk.download('stopwords')

stop_words = stopwords.words('english')

def cleaned_reviews_dataframe(reviews_df):
    reviews_df['title'] = reviews_df['title'].str.replace('\n', '')
    reviews_df['desc'] = reviews_df['desc'].str.replace('\n','')
    reviews_df['title_desc'] = reviews_df['title'] + reviews_df['desc']
    if 'Unnamed: 0' in set(reviews_df.columns):
        reviews_df = reviews_df.drop('Unnamed: 0', axis=1)
    return reviews_df

def get_review_dates(df):
    pass

def docs_to_raw(df, colname):
    return df[colname].tolist()

def vectorized_words(text_list, max_df=0.8, min_df=10, ngram_range=(1,3), stop_words=stop_words, extend_stopwords=False):
    pass