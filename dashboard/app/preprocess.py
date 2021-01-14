import re
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem.wordnet import WordNetLemmatizer

wordnet = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def remove_punc(word:str) -> str:
    '''
    Given a string, removes all punctuation
     and returns punctuation-less string
     '''
    return re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", word)

def tokenize(s, n_grams=2):
    '''
    Tokenize a str and return a tokenized list.
    '''
    tokens = []
    for n in range(1, n_grams+1):
        n_grams = ngrams(nltk.word_tokenize(s), n)
        tokens.extend([' '.join(grams) for grams in n_grams])
    return tokens

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
    bag_of_words = word_tokenize(doc)
    new_bag = [w for w in bag_of_words if w not in stops]
    return ' '.join(new_bag)

def preprocess_corpus(content, n_grams=2):
    '''
    Takes list of strings or Pandas Series. 
    Parameters
    ----------
    content (str): a collection of strings
    Returns
    -------
    A list of lists: each list contains a tokenized version of the original string
    '''

    content_list = content.tolist()

    preprocessed = []
    for i in range(len(content_list)):
        print('removing punctuation')
        step_1 = remove_punc(content_list[i].lower())
        print(step_1)
        print('removing stop words')
        step_2 = rm_stop_words(step_1)
        print(f'Step 2: {step_2}')
        print('tokenizing')
        step_3 = tokenize(step_2, n_grams)
        print(f'Step 3: {step_3}')
        print('lemmatizing')
        step_4 = lemmatize(step_3)
        preprocessed.append(step_4)
        preprocessed_corpus = [' '.join(p) for p in preprocessed]
    return preprocessed_corpus