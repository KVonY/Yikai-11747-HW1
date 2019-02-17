import io
import torch
import numpy as np
from collections import defaultdict

w2i = defaultdict(lambda: len(w2i))
t2i = defaultdict(lambda: len(t2i))
UNK = w2i["<unk>"]
def read_dataset(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            tag, words = line.lower().strip().split(" ||| ")
            words = words.split(" ")
            yield (words, [w2i[x] for x in words], t2i[tag])


# Read in the data
train = list(read_dataset("topicclass/topicclass_train.txt"))
nwords = len(w2i)
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset("topicclass/topicclass_valid.txt"))
# nwords = len(w2i)
ntags = len(t2i)

# https://fasttext.cc/docs/en/english-vectors.html
fin = io.open('../wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
n, d = map(int, fin.readline().split())
data = {}
weight_matrix = []
for line in fin:
    tokens = line.rstrip().split(' ')
    data[tokens[0]] = map(float, tokens[1:])

# w2i_keys = w2i.keys()
w2i_inverse = defaultdict()
for i in w2i.keys():
    w2i_inverse[w2i[i]] = i
for j in w2i_inverse.keys():
    if w2i_inverse[j] in data.keys():
        weight_matrix.append(list(data[w2i_inverse[j]]))
    else:
        weight_matrix.append(list(np.random.uniform(low=-0.25, high=0.25, size=300)))
weight_matrix = torch.from_numpy(np.array(weight_matrix))
print(weight_matrix)
# return weight_matrix