import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from xn9vc_perplexity import Perplexity
import pickle as plk
from torch.autograd import Variable
import time
import math

####device definition is from the PyTorch Documentation
#define a device option for using CPU or GPU
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
#set random generator
torch.manual_seed(8)


### read input data file into list
def read_data(in_file):
    sentence = []
    with open(in_file, 'r', encoding='UTF-8') as file:#need to specify UTF here!
        for line in file:
            line = line.strip()
            tokens = line.split(' ') #split by white space
            sentence.append(tokens)
    return sentence


# this function is from PyTorch documentation
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).to(device)#compute using GPU when cuda is available



# parse data into vocab
trn_sentence = read_data("trn-wiki.txt")
print(len(trn_sentence))
dev_sentence = read_data("dev-wiki.txt")
print(len(dev_sentence))
tst_sentence = read_data("tst-wiki.txt")
print(len(tst_sentence))
print(sum([len(i) for i in trn_sentence]))
print(sum([len(i) for i in dev_sentence]))
print(sum([len(i) for i in tst_sentence]))


# prepare the raw data as input for RNN
trn_dict = {} #key is the token and value should be an index
for sentence in trn_sentence:
    for token in sentence: # ignore the <stop> token
        #check if the token already in the dictionary
        if token not in trn_dict.keys():
            trn_dict[token] = len(trn_dict.keys())
trn_vocab_size = len(trn_dict)
print("trn vocab size",trn_vocab_size)


trn_data = []
for sentence in trn_sentence:
    x = torch.tensor(prepare_sequence(sentence[:-1], trn_dict))
    y = torch.tensor(prepare_sequence(sentence[1:], trn_dict))
    trn_data.append((x, y))

dev_data = []
for sentence in dev_sentence:
    x = torch.tensor(prepare_sequence(sentence[:-1], trn_dict))
    y = torch.tensor(prepare_sequence(sentence[1:], trn_dict))
    dev_data.append((x, y))

tst_data = []
for sentence in tst_sentence:
    x = torch.tensor(prepare_sequence(sentence[:-1], trn_dict))
    y = torch.tensor(prepare_sequence(sentence[1:], trn_dict))
    tst_data.append((x, y))

print(len(trn_data))
print(len(dev_data))
print(len(tst_data))

# import pickle as plk
# with open('trn_data.pickle', 'wb') as f1:
#     plk.dump(trn_data, f1)
# with open('dev_data.pickle', 'wb') as f2:
#     plk.dump(dev_data, f2)
# with open('tst_data.pickle', 'wb') as f3:
#     plk.dump(tst_data, f3)

class RNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size, mini_batch=1):
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mini_batch = mini_batch

        #define word embedding object
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers)

        # on testing LSTM input_size should be the vocab size, we use Linear here to trasform
        self.linear = nn.Linear(in_features=hidden_dim, out_features=vocab_size)



    ##### this is inspired by the coding example on PyTorch documentation
    def forward(self, sentence):
        embeds = self.embedding(sentence).view(len(sentence), 1, -1)#feed sentence into word embedding object

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        linear_out = self.linear(lstm_out.view(len(sentence), -1))
        scores = F.log_softmax(linear_out, 1)
        return scores


    ###### this is inspired by the coding example on PyTorch documentation
    def init_hidden(self):
        weight = torch.zeros(self.num_layers, self.mini_batch, self.hidden_dim).to(device)
        return (weight, weight)


#define loss function
loss_func = nn.NLLLoss()
#model
# embedding_dim , hidden_dim, layers = 32, 32, 1
# embedding_dim , hidden_dim, layers = 32, 128, 1
# embedding_dim , hidden_dim, layers = 64, 32, 1
# embedding_dim , hidden_dim, layers = 64, 64, 1
# embedding_dim , hidden_dim, layers = 128, 64, 1
# embedding_dim , hidden_dim, layers = 128, 128, 1
embedding_dim , hidden_dim, layers = 256, 256, 1


rnn = RNN(hidden_dim=hidden_dim, embedding_dim=embedding_dim,
          num_layers=layers, vocab_size=trn_vocab_size)
rnn.to(device)

epoch = 1
optimizer = optim.SGD(rnn.parameters(), lr=0.2)

for ep in range(epoch):
    start = time.time()
    print("Epoch", ep, "is running....")
    for data_x, data_y in trn_data:
        rnn.zero_grad()

        rnn.hidden = rnn.init_hidden()

        # run forward pass
        scores = rnn(data_x)
        #print(scores.shape)
        loss = loss_func(scores, data_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 10)

        optimizer.step()
        #print(loss)
    end = time.time()
    print("epoch {} uses {} minutes".format(ep, (end - start) / 60));

perplexity_obj = Perplexity()
perplexity_trn = perplexity_obj.compute_perplexity(trn_data, rnn)
print(perplexity_trn)
perplexity_dev = perplexity_obj.compute_perplexity(dev_data, rnn)
print(perplexity_dev);