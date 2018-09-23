from xn9vc_feature_vector import FeatureVector
import numpy as np


class AveragedPerceptron:
    def __init__(self):
        self.feature_vector = FeatureVector()

    def averaged_perceptron(self, X, y):
        t = 0
        theta = []
        theta.append(0)

        for i, x in enumerate(X):
            t += 1
            # find max
            predicted_y = max(np.dot(theta[t - 1], self.feature_vector.feature_vector(x, 0)),
                              np.dot(theta[t - 1], self.feature_vector.feature_vector(x, 0)))

            if predicted_y != y[i]:
                theta[t] = theta[t - 1] + self.feature_vector.feature_vector(x,y[i]) \
                           - self.feature_vector.feature_vector(x, predicted_y)
            else:
                theta[t] = theta[t - 1]

        # average the theta
        averaged_theta = sum(theta) / t

        return theta