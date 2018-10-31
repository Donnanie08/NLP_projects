from xn9vc_commons import Commons
import numpy as np
import random

class AveragedPerceptron:
    def __init__(self):
        self.commons = Commons()



    def averaged_perceptron(self, X, y, epoch=5):
        theta = np.zeros(2 * len(X[0]), dtype=int)
        theta_sum = np.zeros(2 * len(X[0]), dtype=int)

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

                # accumulate theta
                theta_sum += theta

            # shuffule X mapping to y for next epoch
            random.shuffle(dataset)

        # average the theta
        averaged_theta = theta_sum / t

        return averaged_theta