from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation


class ReviewLDA():

    def __init__(self, n_components=5, learning_decay=0.7):
        self.lda = LatentDirichletAllocation(n_components=n_components, learning_decay=learning_decay)

    def fit(self, X, validate=False, n_components=[3, 5, 10], learning_decay=[0.5, 0.7, 0.9]):
        """
        Fit list of tokens to TF-IDF vectorizer model to then fit LDA
        model. If 'validate' is set to true, fits GridSearchCV to find
        optimal LDA model.

        INPUT: X: A list of preprocessed tokens derived form the corpus.

        OUTPUT: LDA model
        """
        # Fit to TF-IDF
        self.tfidf = TfidfVectorizer()
        self.dtm = self.tfidf.fit_transform(X)

        # Begin validation
        if validate:
            search_params = {
                'n_components': n_components,
                'learning_decay': learning_decay
            }
            self.model = GridSearchCV(self.lda, search_params)
            self.model.fit(self.dtm)
            self.best_lda = self.model.best_estimator_
            self.cv_results = self.model.cv_results_
            self.best_params = self.model.best_params_
            self.best_score = self.model.best_score_

        # End validation
        else:
            self.best_lda = self.lda.fit(self.dtm)
        
        self.perplexity = self.best_lda.perplexity(self.dtm)
        self.log_likelihood = self.best_lda.score(self.dtm)
        self.components_ = self.best_lda.components_

        return self.best_lda

    def transform(self, review):
        """
        Convert reviews to list where values correspond to the probability
        the review belongs to each of n topics.

        INPUT: review: A tfidf vectorized review.

        OUTPUT: Probability review belongs to each of n topics
        """
        return self.best_lda.transform(review)