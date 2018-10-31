from bow import BagOfWords
from perceptron import Perceptron
from averaged_perceptron import AveragedPerceptron
from logistic_regression import LogisticRegression
from commons import Commons
import numpy as np


def bow_perceptron(trn_data, trn_label, dev_data, dev_label):
    ######################################################################
    ###### Implementation of BoW model and Perceptrons
    ## Bag of Words
    bag.bag_of_words()
    trn_data_vec = bag.fit_transform(trn_data)
    dev_data_vec = bag.transform(dev_data)
    # tst_data_vec = bag.transform(tst_data)
    #
    #
    # ######
    # ## Perceptron - trained for 12 different epochs
    num_epoches = [x for x in range(1, 13)]
    trn_theta_12 = []
    print("##############perceptron")
    for i, iters in enumerate(num_epoches):
        print(iters)
        trn_theta = percep.perceptron(trn_data_vec, trn_label, epoch=iters)
        trn_theta_12.append(trn_theta)
    # ## Averaged Perceptron
    trn_theta_avg_12 = []
    for i, iters in enumerate(num_epoches):
        print(iters)
        trn_theta_avg = avgPercep.averaged_perceptron(trn_data_vec, trn_label, epoch=iters)
        trn_theta_avg_12.append(trn_theta_avg)

    # write to file
    np.asarray(trn_theta_12).tofile('./trn_theta_12.bin')
    np.asarray(trn_theta_avg_12).tofile('./trn_theta_avg_12.bin')

    # trn_theta_12 = np.fromfile('./trn_theta_12.bin')
    # trn_theta_avg_12 = np.fromfile('./trn_theta_avg_12.bin')

    #
    #
    # ######
    ## Prediction
    # Perceptron
    # trn_score_12 = []
    # dev_score_12 = []
    # print("##############averaged perceptron")
    # for iters in range(len(num_epoches)):
    #     print(iters)
    #     trn_pred = commons.predict(trn_data_vec, trn_theta_12[iters])
    #     trn_score = commons.accuracy(trn_label, trn_pred)
    #     trn_score_12.append(trn_score)
    #     dev_pred = commons.predict(dev_data_vec, trn_theta_12[iters])
    #     dev_score = commons.accuracy(dev_label, dev_pred)
    #     dev_score_12.append(dev_score)
    #
    # # Averaged Perceptron
    # trn_score_avg_12 = []
    # dev_score_avg_12 = []
    # for iters in range(len(num_epoches)):
    #     print(iters)
    #     trn_pred_avg = commons.predict(trn_data_vec, trn_theta_avg_12[iters])
    #     trn_score_avg = commons.accuracy(trn_label, trn_pred_avg)
    #     trn_score_avg_12.append(trn_score_avg)
    #     dev_pred_avg = commons.predict(dev_data_vec, trn_theta_avg_12[iters])
    #     dev_score_avg = commons.accuracy(dev_label, dev_pred_avg)
    #     dev_score_avg_12.append(dev_score_avg)
    #
    # # ## Prediction
    # # # Averaged Perceptron
    # # tst_pred_avg = commons.predict(tst_data_vec, trn_theta_avg)
    # # # Best model
    # # lr.logistics_regression(choice=3, parameter=0.1)
    # # lr.fit(trn_data, trn_label)
    # # tst_pred_best = lr.predict_result(trn_data, trn_label)
    # # np.asarray(tst_pred_best).tofile('./xn9vc-lr-test.pred')
    # #
    # # ## Save prediction results to .pred file
    # # # Averaged Perceptron
    # # np.asarray(tst_pred_avg).tofile('./xn9vc-averaged-perceptron-test.pred')
    # #
    #
    #
    # ## Plots
    # # Perceptron
    # commons.accuracy_curve(num_epoches, trn_score_12, '2.2 Perceptron Accuracy - Training Data')
    # commons.accuracy_curve(num_epoches, dev_score_12, '2.2 Perceptron Accuracy - Development Data')
    # # Averaged Perceptron
    # commons.accuracy_curve(num_epoches, trn_score_avg_12, '2.3 Averaged Perceptron Accuracy - Training Data')
    # commons.accuracy_curve(num_epoches, dev_score_avg_12, '2.3 Averaged Perceptron Accuracy - Development Data')
    #



def lr_countVectorizer(trn_data, trn_label, dev_data, dev_label):
    ######################################################################
    ## Logistics Regression

    # 3.1 - default value
    lr.logistics_regression(choice=1)
    lr.fit(trn_data, trn_label)
    trn_score_lr_3_1 = lr.score(trn_data, trn_label)
    dev_score_lr_3_1 = lr.score(dev_data, dev_label)

    print("3.1 score of trn_data: ", trn_score_lr_3_1)
    print("3.1 score of dev_data: ", dev_score_lr_3_1)

    # 3.2 - Ngram-range
    lr.logistics_regression(choice=2)# update the countVectorizer
    lr.fit(trn_data, trn_label)
    trn_score_lr_3_2 = lr.score(trn_data, trn_label)
    dev_score_lr_3_2 = lr.score(dev_data, dev_label)

    print("3.2 score of trn_data: ", trn_score_lr_3_2)
    print("3.2 score of dev_data: ", dev_score_lr_3_2)


    # 3.3 - L2 Regularization
    for i, x in enumerate([0.0001, 0.001, 0.01, 0.1, 1, 10, 100]):
        lr.logistics_regression(choice=3, parameter=x)# update the countVectorizer
        lr.fit(trn_data, trn_label)
        trn_score_lr_3_3 = lr.score(trn_data, trn_label)
        dev_score_lr_3_3 = lr.score(dev_data, dev_label)

        print("3.3 score of trn_data, lambda=", x, " : ", trn_score_lr_3_3)
        print("3.3 score of dev_data, lambda=", x, " : ", dev_score_lr_3_3)


    # 3.4 - L1 Regularization
    for i, x in enumerate([0.0001, 0.001, 0.01, 0.1, 1, 10, 100]):
        lr.logistics_regression(choice=4, parameter=x)# update the countVectorizer
        lr.fit(trn_data, trn_label)
        trn_score_lr_3_4 = lr.score(trn_data, trn_label)
        dev_score_lr_3_4 = lr.score(dev_data, dev_label)

        print("3.4 score of trn_data, lambda=", x, " : ", trn_score_lr_3_4)
        print("3.4 score of dev_data, lambda=", x, " : ", dev_score_lr_3_4)



if __name__ == '__main__':
    bag = BagOfWords()
    percep = Perceptron()
    avgPercep = AveragedPerceptron()
    lr = LogisticRegression()
    commons = Commons()

    ######
    ## Parse data
    # Training data
    trn_data = open('./trn.data').read().strip().split('\n')
    trn_label = open('./trn.label').read().strip().split('\n')
    trn_label = list(map(int, trn_label))
    # Dev data
    dev_data = open('./dev.data').read().strip().split('\n')
    dev_label = open('./dev.label').read().strip().split('\n')
    dev_label = list(map(int, dev_label))
    # Tst data
    # tst_data = open('./tst.data').read().strip().split('\n')


    bow_perceptron(trn_data, trn_label, dev_data, dev_label)

    # lr_countVectorizer(trn_data, trn_label, dev_data, dev_label)




