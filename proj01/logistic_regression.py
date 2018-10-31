from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LR

class LogisticRegression:
    def __init__(self):
        ## will be updated
        self.classifier = None
        self.vectorizer = None


    def logistics_regression(self, choice, parameter=None):
        # logistics regresison 3.1 - default value
        if choice == 1:
            self.classifier = LR()
            self.vectorizer = CountVectorizer()

        # logisitcs regression 3.2 - ngram
        elif choice == 2:
            self.classifier = LR()
            self.vectorizer = CountVectorizer(ngram_range=(1, 2))

        # logistic regression 3.3 - L2 Regularization
        elif choice == 3:
            # only need to update the LR
            self.classifier = LR(C=parameter)
            self.vectorizer = CountVectorizer(ngram_range=(1, 2))

        # logistic regression 3.4 - L1 Regularization
        elif choice == 4:
            # only need to update the LR
            self.classifier = LR(C=parameter, penalty='l1')
            self.vectorizer = CountVectorizer(ngram_range=(1, 2))

    def fit(self, X, y):
        # fit train data
        vec = self.vectorizer.fit_transform(X)

        self.classifier.fit(vec, y)



    def score(self, X, y):
        # score dev data or test data
        vec = self.vectorizer.transform(X)
        score = self.classifier.score(vec, y)
        return score


    def predict_result(self, X, y):
        vec = self.vectorizer.transform(X)
        return self.classifier.predict(vec)