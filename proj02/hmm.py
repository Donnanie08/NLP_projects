from collections import defaultdict
import math
import numpy as np
from collections import OrderedDict

class HMM:
    def __init__(self):
        self.K = 5 # threshold
        self.emission = {}
        self.transition = {}
        self.emission_smoothed = {}
        self.transition_smoothed = {}
        self.emission_smoothed_log = {}
        self.transition_smoothed_log = {}
        self.word_tag = defaultdict(int)
        self.tag_tag = defaultdict(int)
        self.tags = defaultdict(int)
        self.V = set() # vocab set
        self.N = set() # tags set
        self.N_list = [] # copy of N in the list format


    def preprocess_data(self, line):
        line = line.strip('\n')
        if len(line) != 0:
            items = line.split(' ')
            states = ["START"]
            for t in items:
                t = t.split("/")

                # self.V.add(t[0])
                # self.N.add(t[1])

                # hidden states, add START and END to the list
                states.append(t[1])  # append the tag

                tup = (t[1], t[0])
                self.word_tag[tup] += 1
                # self.tags[t[1]] += 1
            self.tags["START"] += 1
            self.tags["END"] += 1
            states.append("END")

            for s_prev, s_cur in zip(states[:-1], states[1:]):  # construct tag_tag on previous and current
                self.tag_tag[(s_prev, s_cur)] += 1


    def scan_data(self, filename):

        print("Scan data from {}".format(filename))

        with open(filename, encoding='utf8') as filein:
            for line in filein:
                self.preprocess_data(line)


        # now count the frequency with less than K
        for key, value in list(self.word_tag.items()):
            if value <= self.K:
                self.word_tag[(key[0], "Unk")] += value
                del self.word_tag[key]
                # self.tags[key[0]] -= value
            else:
                self.V.add(key[1])
                self.N.add(key[0])
                self.tags[key[0]] += value




        self.N_list = list(self.N)

        # sort for better print out
        self.word_tag = OrderedDict(sorted(self.word_tag.items(), key=lambda t: t[0][1], reverse=True))
        self.tag_tag = OrderedDict(sorted(self.tag_tag.items(), key=lambda t: t[0][0], reverse=False))




        return self.word_tag, self.tag_tag, self.tags


    def scan_dev_data(self, filename):
        '''
        Temporary maintain for dev data testing, should not use self variables for both trn and test!
        '''
        print("Scan data from {}".format(filename))

        tag_true = [] # used for calculate accuracy
        sequences = []


        with open(filename, encoding='utf8') as filein:
            for line in filein:
                line = line.strip('\n')
                if len(line) != 0:
                    items = line.split(' ')
                    tag_sequence = []
                    seq = []
                    for t in items:
                        t = t.split("/")
                        tag_sequence.append(t[1])
                        seq.append(t[0]) # words

                    tag_true.append(tag_sequence)
                    sequences.append(seq)


        return sequences, tag_true


    def test_data(self, inputfile, outputfile):
        print("Scan data from {}".format(inputfile))
        print("Write data to {}".format(outputfile))
        output = open(outputfile, 'w')
        with open(inputfile, encoding='utf8') as filein:
            for line in filein:
                line = line.strip('\n')
                if len(line) != 0:
                    sentence = line.split(' ')
                    tag_sequence = self.viterbi(sentence)

                outline = ""
                for s, t in zip(sentence, tag_sequence):
                    outline += (s + "/" + t + " ")
                output.write(outline+"\n")





    def transition_prob(self, tag_tag, tags):
        for key, val in tag_tag.items():
            tag_prev = key[0]
            prob = val / tags[tag_prev]
            self.transition[key] = prob

        return self.transition


    def emission_prob(self, word_tag, tags):
        for key, val in word_tag.items():
            tag = key[0]
            prob = val / tags[tag]
            self.emission[key] = prob

        return self.emission


    def transition_prob_smoothed(self, tag_tag, tags, beta=1):
        for key, val in tag_tag.items():
            tag_prev = key[0]
            prob = (val + beta) / (tags[tag_prev] + len(self.N) * beta)
            self.transition_smoothed[key] = prob

        for key, value in self.transition_smoothed.items():
            self.transition_smoothed_log[key] = math.log(value)

        return self.transition_smoothed


    def emission_prob_smoothed(self, word_tag, tags, alpha=1):
        for key, val in word_tag.items():

            for tag in self.N_list:
                if self.word_tag.get((tag , key[1])) is None: # pair is not appear in the word_tag
                    self.emission_smoothed[(tag, key[1])] = (0 + alpha) / (tags[tag] + len(self.V) * alpha)
                else:
                    prob = (val + alpha) / (tags[tag] + len(self.V)*alpha)
                    self.emission_smoothed[key] = prob

        # convert the smoothed data into log space
        for key, value in self.emission_smoothed.items():
            self.emission_smoothed_log[key] = math.log(value)

        return self.emission_smoothed


    def get_score(self, word, tag_prev, tag_cur):
        # compute the emission and transition probs of given pair of word and tag
        if word == "":
            emission = 0
        else:
            emission = self.emission_smoothed_log.get((tag_cur, word))

        transition = self.transition_smoothed_log.get((tag_prev, tag_cur))

        if emission is None:
            emission = self.emission_smoothed_log.get((tag_cur, "Unk"))
        if transition is None:
            transition = 0


        score = emission + transition

        return score


    def viterbi(self, sentence):
        # sentence should a list only containing words!

        # viterbi_var = [0] * 10  # scores list for Viterbi
        viterbi_var = np.zeros((len(self.N_list), len(sentence)))
        b = np.chararray((len(self.N_list), len(sentence)), unicode=True)  # scores list for Viterbi
        b[:] = ""
        # calculate the score for first position
        for i, k in enumerate(self.N_list):
            viterbi_var[i][0] = self.get_score(sentence[0], "START", k)


        # b = [""] * 10 # backpoints tracking
        sequence = []
        for m in range(1, len(sentence)):
            # new_viterbi_var = [0] * 10
            # new_b = [""] * 10
            for k in range(0, len(self.N_list)):
                score = [0] * 10
                for i in range(0, len(self.N_list)):
                    score[i] = viterbi_var[i][m-1] + self.get_score(sentence[m], self.N_list[i], self.N_list[k])
                max_score = max(score)
                viterbi_var[k][m] = max_score
                # new_b[k] =
                b[k][m] = self.N_list[score.index(max_score)]
            # viterbi_var = new_viterbi_var
            cur_column = viterbi_var[:, m]
            cur_column = cur_column.tolist()
            sequence.append(b[cur_column.index(max(cur_column))][m])

        end_viterbi = []
        end_score = [0] * 10
        for i in range(0, len(self.N_list)):
            end_score[i] = viterbi_var[i][-1] + self.get_score("", self.N_list[i], "END")
        max_end_score = max(end_score)
        sequence.append(self.N_list[end_score.index(max_end_score)])

        return sequence


    def accuracy(self, y_true, y_pred):
        correct_cnt = 0
        total_tags = 0
        for true, pred in zip(y_true, y_pred):
            for t, p in zip(true, pred):
                total_tags += 1
                if t == p:
                    correct_cnt += 1

        return correct_cnt / total_tags