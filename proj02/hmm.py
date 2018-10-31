from collections import defaultdict
import math
import numpy as np

class HMM:
    def __init__(self):
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


    def preprocess_data(self, line):
        line = line.strip('\n')
        if len(line) != 0:
            items = line.split(' ')
            states = ["START"]
            for t in items:
                t = t.split("/")

                self.V.add(t[0])
                self.N.add(t[1])

                # hidden states, add START and END to the list
                states.append(t[1])  # append the tag

                tup = (t[1], t[0])
                self.word_tag[tup] += 1
                # tag_tag.append(t[1])
                self.tags[t[1]] += 1
                self.tags["START"] += 1
                self.tags["END"] += 1
            states.append("END")

            for s_prev, s_cur in zip(states[:-1], states[1:]):  # construct tag_tag on current and previous
                self.tag_tag[(s_prev, s_cur)] += 1



    def scan_data(self, filename):
        print("Scan data from {}".format(filename))

        with open(filename, encoding='utf8') as filein:
            for line in filein:
                self.preprocess_data(line)

        return self.word_tag, self.tag_tag, self.tags

    def read_saved_txt(self, word_tag_file, tag_tag_file):
        print("Read data from saved txt file {}".format(word_tag_file))
        print("Read data from saved txt file {}".format(tag_tag_file))

        word_tag, tag_tag, tags, vocab = {}, {}, set(), set()
        with open(word_tag_file) as filein:
            for line in filein:
                word_tag[(line[0], line[1])] = line[2]
                tags.add(line[0])
                vocab.add(line[1])

        with open(tag_tag_file) as filein:
            for line in filein:
                tag_tag[(line[0], line[1])] = line[2]

        return word_tag, tag_tag, tags, vocab


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


    def transition_prob_smoothed(self, tag_tag, tags):
        for key, val in tag_tag.items():
            tag_prev = key[0]
            prob = (val + 1) / (tags[tag_prev] + len(self.N))
            self.transition_smoothed[key] = prob

        for key, value in self.transition_smoothed:
            self.transition_smoothed_log[key] = math.log2(value)

        return self.transition_smoothed


    def emission_prob_smoothed(self, word_tag, tags):
        for key, val in word_tag.items():
            tag = key[0]
            prob = (val + 1) / (tags[tag] + len(self.V))
            self.emission_smoothed[key] = prob

        # convert the smoothed data into log space
        for key, value in self.emission_smoothed:
            self.emission_smoothed_log[key] = math.log2(value)

        return self.emission_smoothed


    def viterbi(self, sentence):
        # sentence should a list only containing words!!

        sequence = []

        score = defaultdict(int)  # scores list for Viterbi
        # calculate the score for first position
        for i, k in enumerate(self.N):
            score[i] = self.emission_smoothed_log[(k, sentence[0])] + self.transition_smoothed_log[("START", k)]


        # construct b list based on the size of sequence
        b = np.zeros([self.N, sentence])


        # Viterbi Algorithm
        for
            self.score.append()

        return sequence
