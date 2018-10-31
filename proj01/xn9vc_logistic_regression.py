from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR
from xn9vc_commons import Commons
import sys

class LogisticRegression:
    def __init__(self):
        self.classifier = None
        self.vectorizer = None


    def logistics_regression(self, choice):

        # fit transfrom and get the vectorized data

        # logistics regresison 3.1 - default value
        if choice == 0:
            self.classifier = LR()
            self.vectorizer = CountVectorizer()

        # logisitcs regression 3.2 - ngram
        elif choice == 1:
            self.classifier = LR()
            self.vectorizer = CountVectorizer(lowercase=True, ngram_range=(1,2))

        # logisitcs regression 3.3 - L2 regularization - tune lambda
        elif choice == 2:
            lambdas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
            self.classifier = LR()
            self.vectorizer = CountVectorizer(lowercase=True, max_df=0.5)

        else:
            sys.stderr.write("Please use valid choice for CountVectorizer!")

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def score(self, X, y):
        self.classifier.score(X, y)


lr = LogisticRegression()
commons = Commons()
# Training data
trn_data = open('./trn.data').read().strip().split('\n')
trn_label = open('./trn.label').read().strip().split('\n')
trn_label = commons.str2int(trn_label)
# Dev data
dev_data = open('./dev.data').read().strip().split('\n')
dev_label = open('./dev.label').read().strip().split('\n')
dev_label = commons.str2int(dev_label)

lr.logistics_regression(choice=0)
lr.fit()