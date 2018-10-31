# from xn9vc_bag_of_words import BagOfWords
from xn9vc_perceptron import Perceptron
from xn9vc_averaged_perceptron import AveragedPerceptron
from xn9vc_commons import Commons

from xn9vc_bag_of_words_countVectorizer import BagOfWords

if __name__ == '__main__':
    ## class initializations
    # BagOfWords
    bag = BagOfWords()
    # Perceptron
    percep = Perceptron()
    # Average Perceptron
    averPercep = AveragedPerceptron()
    # commons operation
    commons = Commons()

    ## parse training, dev, and testing data
    # Training data
    trn_data = open('./trn.data').read().strip().split('\n')
    trn_label = open('./trn.label').read().strip().split('\n')
    trn_label = commons.str2int(trn_label)
    # Dev data
    dev_data = open('./dev.data').read().strip().split('\n')
    dev_label = open('./dev.label').read().strip().split('\n')
    dev_label = commons.str2int(dev_label)
    # Testing data
    # tst_data = open('./dev.data').read().strip().split('\n')


    ## Create bag of words for each data
    bag.bag_of_words(choice=3)
    trn_data_vectorizer = bag.fit(trn_data, trn_label, choice=1)
    dev_data_vectorizer = bag.fit(dev_data, dev_label, choice=2)
    # tst_data_vectorizer = bag.bag_of_words(tst_data)


    ## Perceptron on training data
    trn_theta = percep.perceptron(trn_data_vectorizer.toarray(), trn_label)
    ## Average Perceptron on training data
    trn_avg_theta = averPercep.averaged_perceptron(trn_data_vectorizer.toarray(), trn_label)



    # ## Prediction with Dev data and Tune down parameters
    # # Traning Data - Perceptron
    # trn_pred = commons.predict(trn_data_vectorizer.toarray(), trn_theta)
    # trn_score = commons.accuracy(trn_label, trn_pred)
    # # Dev Data - Averaged Perceptron
    # trn_avg_pred = commons.predict(trn_data_vectorizer.toarray(), trn_avg_theta)
    # trn_avg_score = commons.accuracy(trn_label, trn_avg_pred)
    # print("score1:", trn_score)
    # print("score2: ", trn_avg_score)

    # Dev Data - Perceptron
    dev_pred = commons.predict(dev_data_vectorizer.toarray(), trn_theta)
    dev_score = commons.accuracy(dev_label, dev_pred)
    # Dev Data - Averaged Perceptron
    dev_avg_pred = commons.predict(dev_data_vectorizer.toarray(), trn_avg_theta)
    dev_avg_score = commons.accuracy(dev_label, dev_avg_pred)
    print("score1:", dev_score)
    print("score2: ", dev_avg_score)

    ## Prediction on Testing data
    # tst_pred = commons.predict(tst_data_vectorizer)




