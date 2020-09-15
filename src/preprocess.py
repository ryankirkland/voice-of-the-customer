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

def remove_punc(my_string:str) -> str:
    '''Given a string, removes all punctuation and returned punctuation-less string'''
    return re.sub(f'[{string.punctuation}]', '', my_string)

def tokenize(str):
    '''
    Tokenize a str and return a tokenized list.
    '''
    return [word for word in word_tokenize(str)]

def lemmatize(doc):
    '''Takes in a doc and lemmatizes tokens in doc
    Parameters
    ----------
    doc: list of tokens
    
    Returns
    -------
    lemmatized tokens
    '''
    return [wordnet.lemmatize(tkn) for tkn in doc]

def rm_stop_words(doc, stops=set(stopwords.words('english'))):
    '''Takes in a doc and removes stop words
    Parameters
    ----------
    doc: list of tokens
    
    Returns
    -------
    Tokens with stop words removed
    '''
    return([w for w in doc if w not in stops])

def n_grams(input_tokens):
    # retain 1-gram tokens
    ret_list = list(input_tokens)
    
    for i in range(2,3):
        # add each n-grams to the list
        ret_list.extend(['-'.join(tgram) for tgram in ngrams(input_tokens, i)])
    return(ret_list)
    
    
def preprocess_corpus(content):
    '''
    Add docstring. Make flexible to allow for doing, or not doing, preprocessing functions. 
    Parameters
    ----------
    content (str): a collection of strings
    Returns
    -------
    A list of lists: each list contains a tokenized version of the original string
    '''
    preprocessed = []
    for i in range(len(content)):
        step_1 = remove_punc(content[i].lower())
        step_2 = tokenize(step_1)
        step_3 = lemmatize(step_2)
        step_4 = rm_stop_words(step_3)
        step_5 = n_grams(step_4)
        preprocessed.append(step_5)
    return preprocessed