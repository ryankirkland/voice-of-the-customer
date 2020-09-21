def docs_to_raw(df, colname):
    return df[colname].tolist()

def vectorized_words(text_list, max_df=0.8, min_df=10, ngram_range=(1,3), stop_words=stop_words, extend_stopwords=False):
    pass

class ReviewLDA():

    def __init__(self):
        self.lda = LatentDirichletAllocation()

    def fit(self, X, validate=False):
        """
        Fit list of tokens to TF-IDF vectorizer model to then fit LDA
        model. If 'validate' is set to true, fits GridSearchCV to find
        optimal LDA model.

        INPUT: X is a list of preprocessed tokens derived form the corpus.

        OUTPUT: LDA model
        """
        