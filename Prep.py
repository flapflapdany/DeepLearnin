import random
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def MySeed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
MySeed(5)  # Генерируем зерно


def ReadRawTxt(parth):
    raw_text = dict()
    with open(parth, 'r') as f:
        raw_text = f.readlines()
    for line in range(len(raw_text)):
        raw_text[line] = raw_text[line].lower()
        raw_text[line] = re.findall(r'[\w\d]+', raw_text[line])
    return raw_text


def ensurelen(txt, l, p):
    txt = list(txt) + [p] * (l - len(txt)) if (len(txt) < l) else txt[:l]
    return txt


class Prep():
    def __init__(self, parth, freq=0.8):
        self.txt = ReadRawTxt(parth)
        self.doccount = len(self.txt)
        self.freqword = dict()
        self.idword = dict()
        self.words = dict()
        self.freq = freq
        self.Uniquewords()

    def Uniquewords(self):
        id1 = 0
        for i in range(len(self.txt)):
            for j in range(len(self.txt[i])):
                if self.txt[i][j] not in self.words.keys():
                    self.words[self.txt[i][j]] = id1
                    self.idword[id1] = self.txt[i][j]
                    self.freqword[self.txt[i][j]] = 1
                    id1 = id1 + 1
                else:
                    self.freqword[self.txt[i][j]] = self.freqword[self.txt[i][j]] + 1
        #print(self.words)

    def Textstotokenids(self, text, word2id):
        # self.Uniquewords()
        text1 = []
        k = 0
        #print(self.idword)
        for i in range(len(text)):
            t = []
            for j in range(len(text[i])):
                if text[i][j] in word2id:
                    t.append(self.words[text[i][j]])
            if (len(t) != 0):
                text1.append(t)
        return text1

    def Freaqsort(self, a, min_count=5):
        for i in a:
            if (self.freqword[i] < min_count) or (self.freqword[i] / self.doccount > self.freq):
                self.freqword.pop(i)
        sort = sorted(self.freqword.items(), reverse=True, key=lambda pair: pair[1])
        return sort

    def build_vocabulary(self, maxs=1000000, min_count=5,pad_word=None):  # нужно посчитать строчки чтобы отнормировать при чтении
        sorted_word = self.Freaqsort(self.words, min_count=5)
        if len(self.words) > maxs:
            sorted_word = sorted_word[:maxs]
        word2id = {word: i for i, (word, _) in enumerate(sorted_word)}
        word2freq = np.array([cnt / self.doccount for _, cnt in sorted_word], dtype='float32')
        return word2id, word2freq


class PadSeqData(Dataset):
    def __init__(self, texts, targetscount, lenofsent=50, p=0):
        self.texts = texts
        self.targetscount = targetscount
        self.lenofsent = lenofsent
        self.pad = p

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        newtxt = self.texts[item]
        newtxt = ensurelen(newtxt, self.lenofsent, self.pad)
        newtxt = torch.tensor(newtxt, dtype=torch.long)
        purp = torch.tensor(self.targetscount[item], dtype=torch.long)
        return newtxt, purp


def maskfromlengths(len1, maxlen):
    res = torch.zeros(len(len1), maxlen, dtype=torch.bool)
    for i, cur in enumerate(len1):
        res[i, :cur] = 1
    return res


