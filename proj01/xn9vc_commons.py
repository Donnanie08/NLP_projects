import numpy as np
from sklearn.metrics import accuracy_score

class Commons:

    def feature_vector(self, x_i, y_i):
        # line + vocab + label -> feature vector
        # construct new list with double size
        # couldn't find a way to construct two values at a time with oneliner
        vector = np.zeros(2*len(x_i), dtype=int)


        if y_i == 0:
            vector[::2] = x_i
        elif y_i == 1:
            vector[1::2] = x_i

        # for i in x_i:
        #     if y_i == 0:
        #         vector.append(i)
        #         vector.append(0)
        #     elif y_i == 1:
        #         vector.append(0)
        #         vector.append(i)

        return vector


    def predict(self, X, theta):
        predicted_y = [0 if np.dot(theta, self.feature_vector(x, 0)) \
                               >= np.dot(theta, self.feature_vector(x, 1)) else 1
                           for i, x in enumerate(X)]
        return predicted_y


    def accuracy(self, y_true, y_pred):
        ## Use accuracy_score libraby here
        return accuracy_score(np.asarray(y_true), np.asarray(y_pred))



    def str2int(self, str_list):
        # y is list of strings, convert it into integer
        return list(map(int, str_list))
