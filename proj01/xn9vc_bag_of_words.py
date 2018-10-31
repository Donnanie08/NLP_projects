import string
import numpy as np
from collections import defaultdict
from itertools import islice
from matplotlib import pyplot as plt
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from nltk import FreqDist

class BagOfWords:

    # construct vocabulary first
    # construct the vectors based on vocabulary

    def __init__(self):
        self.stop_words = stopwords.words('english') + list(string.punctuation)



    def plot_histogram(self, vocabulary):
        plt.bar(vocabulary.keys(), vocabulary.values())
        plt.show()


    def bag_of_words(self, data):
        '''
        Finds all vocabulary from the input data
        :return:
        '''

        min_df = 1 - 0.6

        self.vocabulary = defaultdict(int) # dictionary of words with frequency

        tokenized_data = []

        # preprocess - remove punctuations, remove stop words


        # tokenize
        for s in data:
            words = [w for w in wordpunct_tokenize(s.lower()) if w not in self.stop_words]
            tokenized_data.append(words)
            for w in words:
                self.vocabulary[w] += 1

        # sorted dictionary on frequencies, by descending order
        # int(len(self.vocabulary)*min_df)
        sliced_dict = {key : value for key, value in islice(sorted(self.vocabulary.items(), key=lambda kv: kv[1], reverse=True), 1000)}

        # vectorize
        return np.asarray([[1 if w in sample else 0 for w in sliced_dict.keys()] for sample in tokenized_data])








