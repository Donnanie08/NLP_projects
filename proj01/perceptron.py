import random
import numpy as np
from commons import Commons

class Perceptron:
    def __init__(self):
        self.commons = Commons()


    def perceptron(self, X, y, epoch=5):
        theta = np.zeros(2 * len(X[0]), dtype=int)

        dataset = [(X[i], y[i]) for i in range(len(X))]
        t = 0
        for iter in range(epoch + 1):
            for i, data in enumerate(dataset):
                t += 1
                # find max
                predicted_y = 0 if np.dot(theta, self.commons.feature_vector(data[0], 0)) \
                                   >= np.dot(theta, self.commons.feature_vector(data[0], 1)) else 1

                if predicted_y != data[1]:
                    theta = theta + self.commons.feature_vector(data[0], data[1]) \
                            - self.commons.feature_vector(data[0], predicted_y)

            # shuffule X mapping to y for next epoch
            random.shuffle(dataset)

        return theta
