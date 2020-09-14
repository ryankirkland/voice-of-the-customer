import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

class ReviewClassifier:
    """
    Creates a Random Forest Classifier trained on positive and negative
    Amazon reviews that can then predict the probability that neutral, 
    3-star reviews are positive or negative.

    Fits by converting review text into TF-IDF vectorized features.
    """

    def __init__(self, n_estimators=100, max_depth=None):
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X_train, y_train):
        """
        INPUT: Expects X_train to be a TfIdf-vectorized feature set generated from Amazon review content and y_train to be the associated positve or negative label.
        """
        self.clf.fit(X_train, y_train)

    def predict_proba(self, X):
        preds = self.clf.predict_proba(X)
        return 