import sys
from sklearn.feature_extraction.text import CountVectorizer

class BagOfWords:
    def __init__(self):
        self.vectorizer = None

    def bag_of_words(self):
        self.vectorizer = CountVectorizer(lowercase=True, min_df=3, max_df=0.6)

        # return vectorizer


    def fit_transform(self, trn_data):
        # fit_transform training data
        data = self.vectorizer.fit_transform(trn_data)
        return data.toarray()


    def transform(self, dev_data):
        data = self.vectorizer.transform(dev_data)
        return data.toarray()
