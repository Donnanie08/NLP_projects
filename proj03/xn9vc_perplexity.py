import numpy as np
import time
import math
import torch
####device definition is from the PyTorch Documentation
#define a device option for using CPU or GPU
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

class Perplexity:
    def compute_perplexity(self, sentences, rnn):
        start = time.time()
        log_probs, sizes = 0, 0
        for i in range(len(sentences)):
            with torch.no_grad():
                inputs = torch.tensor(sentences[i][0], dtype=torch.long).to(device)
                score = rnn(inputs)
            for j in range(len(inputs) - 1):
                log_probs += score[j][inputs[j + 1]]

            sizes += (len(inputs) + 1)
        avg = (1 / sizes) * log_probs

        perplexity = math.exp(-avg)

        end = time.time()
        print("perplexity calculation takes {} minutes".format((end-start)/60))

        return (perplexity)
