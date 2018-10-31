import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class Commons:


    def feature_vector(self, x_i, y_i):
        vector = np.zeros(2*len(x_i), dtype=int)

        if y_i == 0:
            vector[::2] = x_i
        elif y_i == 1:
            vector[1::2] = x_i

        return vector



    def predict(self, X, theta):
        predicted_y = [0 if np.dot(theta, self.feature_vector(x, 0)) \
                               >= np.dot(theta, self.feature_vector(x, 1)) else 1
                           for i, x in enumerate(X)]
        return predicted_y


    def accuracy(self, y_true, y_pred):
        ## Use accuracy_score libraby here
        return accuracy_score(np.asarray(y_true), np.asarray(y_pred))



    def accuracy_curve(self, x, y, title):
        # x-axis should be number of epoches
        # y-axis should be accuracy score
        plt.title(title)
        plt.xlabel('number of epoches')
        plt.ylabel('accuracy score')
        plt.plot(x, y)
        plt.show()


