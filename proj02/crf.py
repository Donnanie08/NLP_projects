## crf.py
## Author: CS 6501-005 NLP @ UVa
## Time-stamp: <yangfeng 10/14/2018 16:14:05>

from util import *
from hmm import HMM
import sklearn_crfsuite as crfsuite
from sklearn_crfsuite import metrics

class CRF(object):
    def __init__(self, trnfile, devfile):
        self.trn_text = load_data(trnfile)
        self.dev_text = load_data(devfile)
        #
        print("Extracting features on training data ...")
        self.trn_feats, self.trn_tags = self.build_features(self.trn_text)
        print("Extracting features on dev data ...")
        self.dev_feats, self.dev_tags = self.build_features(self.dev_text)
        #
        self.model, self.labels = None, None

    def build_features(self, text):
        feats, tags = [], []
        for sent in text:
            N = len(sent.tokens)
            sent_feats = []
            for i in range(N):
                word_feats = self.get_word_features(sent, i)
                sent_feats.append(word_feats)
            feats.append(sent_feats)
            tags.append(sent.tags)
        return (feats, tags)

        
    def train(self):
        print("Training CRF ...")
        self.model = crfsuite.CRF(
            # algorithm='lbfgs',
            algorithm='ap',
            max_iterations=5)
        self.model.fit(self.trn_feats, self.trn_tags)
        trn_tags_pred = self.model.predict(self.trn_feats)
        self.eval(trn_tags_pred, self.trn_tags)
        dev_tags_pred = self.model.predict(self.dev_feats)
        self.eval(dev_tags_pred, self.dev_tags)


    def eval(self, pred_tags, gold_tags):
        if self.model is None:
            raise ValueError("No trained model")
        print(self.model.classes_)
        print("Acc =", metrics.flat_accuracy_score(pred_tags, gold_tags))

        
    def get_word_features(self, sent, i):
        """ Extract features with respect to time step i
        """
        # the i-th token
        word_feats = {'tok': sent.tokens[i]}
        # TODO for question 1
        # the i-th tag
        features_dict = {
            "tok.upper": sent.tokens[i].upper(),
            "tok.fl": sent.tokens[0],
            "tok.ll": sent.tokens[-1]
        }

        if len(sent.tokens[i]) > 2:
            features_dict.update({"tok.fwl": sent.tokens[0:1],
                                "tok.lwl": sent.tokens[-2:-1]})

        if i > 0:
            features_dict.update({"tok.prev": sent.tokens[i-1]})

        if i < len(sent.tokens) - 1:
            features_dict.update({"tok.next": sent.tokens[i+1]})

        word_feats.update(features_dict)

        # 
        # TODO for question 2
        # add more features here
        return word_feats


if __name__ == '__main__':
    # trnfile = "trn-tweet.pos"
    # devfile = "dev-tweet.pos"
    # crf = CRF(trnfile, devfile)
    # crf.train()

    hmm = HMM()
    word_tag, tag_tag, tags = hmm.scan_data("trn.pos")
    emission = hmm.emission_prob(word_tag, tags)
    write_txt(emission, "./xn9vc-eprob.txt")
    transition = hmm.transition_prob(tag_tag, tags)
    write_txt(transition, "./xn9vc-tprob.txt")
    emission_smoothed = hmm.emission_prob_smoothed(word_tag, tags)
    write_txt(emission_smoothed, "./xn9vc-eprob-smoothed.txt")
    transition_smoothed = hmm.transition_prob_smoothed(tag_tag, tags)
    write_txt(transition_smoothed, "./xn9vc-tprob-smoothed.txt")
    #
    #
    # # test viterbi data with dev.pos
    # sequences, tag_dev = hmm.scan_dev_data("dev.pos")
    # for seq in sequences:
    #     tag_pred = hmm.viterbi(seq)



    
