# used for removing some less meaningful high frequency words
from nltk.corpus import stopwords
# used for tokenize the words
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# used for removing punctuations
from nltk.tokenize import RegexpTokenizer
import numpy as np

class BagOfWords:
    def clean_data(self, sentence):
        # data cleaning

        # ignore punctuations & tokenize
        regex_tokenizer = RegexpTokenizer(r'\w+')
        cleaned_sentence = regex_tokenizer.tokenize(sentence)

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


    def tokenize(self, sentences):
        tokens = word_tokenize(sentences)
        return tokens


    def bag_of_words(self):
        trn_label = map(str, open('./trn.label').read().strip().split('\n'))
        with open('./trn.data') as f:
            trn_data = f.readlines()
            trn_data = trn_data.strip().split('\n')


        # trn_token = tokenize(trn_data[0])
        # clean_data(trn_token)
        cleaned_trn_data = []
        for sentence in trn_data:
            print(self.clean_data(sentence))
            # cleaned_trn_data.append(self.clean_data(sentence))

        # print(cleaned_trn_data[1])

        return 0


bag = BagOfWords()
bag.bag_of_words()
