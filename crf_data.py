import re
import numpy as np


# Loss rather than supplement
def norm(seq, x2id, max_len=None):
    ids = [x2id.get(x, 0) for x in seq]
    if max_len is None:
        return ids
    if len(ids) >= max_len:
        ids = ids[: max_len]
        return ids
    ids.extend([0] * (max_len - len(ids)))
    return ids


# Log softmax
def log_softmax(vec):
    max_val = np.max(vec)
    log_sum_exp = np.log(np.sum(np.exp(vec - max_val)))
    for i in range(np.size(vec)):
        vec[i] -= log_sum_exp + max_val
    return vec


class CrfData(object):
    def __init__(self, train, test=None, testr=None):
        self.train = train
        self.data = []
        self.tag = []
        self.word2id = None
        self.id2word = None
        self.tag2id = {'': 0,
                  'B_ns': 1,
                  'I_ns': 2,
                  'B_nr': 3,
                  'I_nr': 4,
                  'B_nt': 5,
                  'I_nt': 6,
                  'o': 7}
        self.id2tag = {v: k for k, v in self.tag2id.items()}
        self.read_train_data()
        self.map_word_id()

    def read_train_data(self):
        i = 0
        with open(self.train, "r", encoding="utf-8-sig") as file:
            for line in file.readlines():
                if i >= 1000: break
                else: i += 1
                line = line.strip().split()
                seq = ''
                if len(line) != 0:
                    for word in line:
                        word = word.split('/')
                        if word[1] == 'o':
                            seq += ''.join([char + "/o " for char in word[0]])
                        else:
                            seq += ''.join([word[0][0] + "/B_" + word[1] + ' '] +
                                           [char + "/I_" + word[1] + ' ' for char in word[0][1:]])

                line = re.split('[，。；！：？、‘’”“]/[o]', seq.strip())
                for subSeq in line:
                    subSeq = subSeq.strip().split()
                    if len(subSeq):
                        subData = []
                        subtag = []
                        noEntity = True
                        for word in subSeq:
                            word = word.split('/')
                            subData.append(word[0])
                            subtag.append(word[1])
                            noEntity &= (word[1] == 'o')
                        if not noEntity:
                            self.data.append(subData)
                            self.tag.append(subtag)

    def map_word_id(self):
        wordbag = sum(self.data, [])
        wordset = set(wordbag)
        wordict = {k: 0 for k in wordset}
        for k in wordbag:
            wordict[k] += 1
        wordlst = sorted(wordict.items(), key=lambda x: x[1], reverse=True)
        wordset = [x for x, _ in wordlst]
        idset = range(1, len(wordset) + 1)
        self.word2id = {k: v for k, v in zip(wordset, idset)}
        self.id2word = {k: v for k, v in zip(idset, wordset)}

    def get_train_data(self, max_len):
        train_x = []
        train_y = []
        for data, tag in zip(self.data, self.tag):
            train_x.append(norm(data, self.word2id, max_len))
            train_y.append(norm(tag, self.tag2id, max_len))
        return list(filter(None, train_x)), list(filter(None, train_y))

    def log_likelihood(self):
        row = len(self.word2id) + 1
        col = len(self.tag2id)
        loglikelihoods = np.zeros((row, col), dtype="float32")
        wordbag = norm(sum(self.data, []), self.word2id)
        tagbag = norm(sum(self.tag, []), self.tag2id)

        for word, tag in zip(wordbag, tagbag):
            loglikelihoods[word, tag] += 1

        for j in range(col):
            log_softmax(loglikelihoods[:, j])

        return loglikelihoods
