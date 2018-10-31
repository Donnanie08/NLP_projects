import sys
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class BagOfWords:
    def __init__(self):
        self.vectorizer = None


    def bag_of_words(self, choice=1):

        # fit transfrom and get the vectorized data

        # logistics regresison 3.1 - default value
        if choice == 0:
            self.vectorizer = CountVectorizer()
        elif choice == 1:
            self.vectorizer = CountVectorizer(lowercase=True, min_df=2)
        elif choice == 2:
            self.vectorizer = CountVectorizer(lowercase=True, max_df=0.5)
        elif choice == 3:
            self.vectorizer = CountVectorizer(lowercase=True, min_df=3, max_df=0.6)
        elif choice == 4:
            self.vectorizer = CountVectorizer(lowercase=True, stop_words='english', min_df=2)

        # logisitcs regression 3.2
        elif choice == 5:
            self.vectorizer = CountVectorizer(lowercase=True, ngram_range=(1,2))
        else:
            sys.stderr.write("Please use valid choice for CountVectorizer!")


    def fit(self, data, label, choice):
        if choice == 1:
            # at this point, transformed_data is considered as a sparse
            vectoriced_data = self.vectorizer.fit_transform(data)

        elif choice == 2:
            vectoriced_data = self.vectorizer.transform(data)
        else:
            sys.stderr.write("Please use valid choice for CountVectorizer!")
        #
        # data_arr = vectoriced_data.toarray()
        # vector = np.zeros(2*len(data_arr), dtype=int)
        #
        # y_label = list(map(int, label))
        #
        # for i, x in enumerate(data_arr):
        #     if y_label[i] == 0:
        #         vector[::2] = x
        #     elif y_label[i] == 1:
        #         vector[1::2] = x

        return vectoriced_data





