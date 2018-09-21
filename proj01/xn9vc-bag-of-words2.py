# used for removing some less meaningful high frequency words
from nltk.corpus import stopwords
# used for tokenize the words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class BagOfWords:
    def prepare_text(self, sentence):
        # data cleaning

        # tokenize & filter all tokens that are not alphabetic
        tokenized_sentence = word_tokenize(sentence)
        cleaned_sentence = [w for w in tokenized_sentence if w.isalpha()]


        # convert all text to lower case
        cleaned_sentence = [s.lower() for s in cleaned_sentence]

        # ignore a frequent word that doesn't contain much information like 'a', 'of'.
        # use stopwords here
        stop_words = set(stopwords.words('english')) # language English
        filter_sentence = [w for w in cleaned_sentence if w not in stop_words]

        ## can be used for comparing the performance
        # stemming
        ps = PorterStemmer()
        stem_sentence = [ps.stem(w) for w in filter_sentence]




        return filter_sentence



    def bag_of_words(self):
        # get a list of all sentences in the file
        trn_data = open('./trn.data').read().strip().split('\n')
        trn_label = map(str, open('./trn.label').read().strip().split('\n'))

        vecotrized = CountVectorizer(trn_data)

        # build vocabulary
        vocab = {'' : 0}
        for data in trn_data:
            cleaned_trn_data = wordpunct_tokenize(data.lower())
            cleaned_trn_data = [w for w in cleaned_trn_data if w.isalpha()]
            for w in cleaned_trn_data:
                vocab[w] += 1
            print(cleaned_trn_data)

        # trn_token = tokenize(trn_data[0])
        # clean_data(trn_token)

        return 0


bag = BagOfWords()
bag.bag_of_words()


## Use merge sort to add tokens to vocabulary
