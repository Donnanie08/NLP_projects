from xn9vc_bag_of_words import BagOfWords
from xn9vc_feature_vector import FeatureVector
import numpy as np
import random
import pandas as pd

class Perceptron:
    def __init__(self, epoches=5, learning_rate=0.01, random_state=1):
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.random_state = random_state
        self.errors = []
        self.feature_vector = FeatureVector()



    def perceptron(self, X, y, epoch=1):

        theta = np.zeros([1, len(X[0])], dtype=int)
        # y is list of strings
        y = list(map(int, y))

        dataset = [(X[i], y[i]) for i in range(len(X))]

        for iter in range(epoch+1):
            t = 0

            for i, data in enumerate(dataset):
                t += 1
                # find max
                predicted_y = 0 if np.dot(theta, self.feature_vector.feature_vector(data[0], 0)) >= np.dot(theta, self.feature_vector.feature_vector(data[0], 1)) else 1

                if predicted_y != data[1]:
                    theta = theta + self.feature_vector.feature_vector(data[0], data[1]) - self.feature_vector.feature_vector(data[0], predicted_y)

            # shuffule X mapping to y for next epoch
            random.shuffle(dataset)



        return theta


bag = BagOfWords()
bag.bag_of_words()
trn_data_vectorizer = bag.vectorizer
percep = Perceptron(epoches=1)
# tst_input = open('./tst.data').read().strip().split('\n')
trn_label = open('./trn.label').read().strip().split('\n')

percep.perceptron(trn_data_vectorizer[0:5], trn_label[0:5])

