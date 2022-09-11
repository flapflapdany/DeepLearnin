from torch.nn import functional as F
import random
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def makediagmask(size, rad):
    idxs = torch.arange(size)
    idxdif = (idxs.unsqueeze(0) - idxs.unsqueeze(1)).abs()
    return ((idxdif <= rad) & (idxdif > 0)).float()


class SkipGram(nn.Module):
    def __init__(self, vsize, esize, sentlen, rad=5, nsampl=5):
        self.nsampl = nsampl
        super().__init__()
        self.vsize = vsize
        print(esize)
        self.cenb = nn.Embedding(vsize, esize, padding_idx=0)  # центральный
        print(self.cenb)
        k = 1.0 / esize
        self.cenb.weight.data.uniform_(-k, k)
        self.cenb.weight.data[0] = 0  # center emb
        self.cemb = nn.Embedding(vsize, esize, padding_idx=0)
        self.cemb.weight.data.uniform_(-k, k)
        self.cemb.weight.data[0] = 0
        self.posmask = makediagmask(sentlen, rad)

    def forward(self, sent):
        batchcount = sent.shape[0]  # Размер батча
        cenbeddings = self.cenb(sent)  # Batch x MaxSentLength x EmbSize
        positconembs = self.cemb(sent).permute(0, 2, 1)  # Позитивный контекст емб
        possims = torch.bmm(cenbeddings, positconembs)
        posprobs = torch.sigmoid(possims)
        pmask = self.posmask.to(possims)  # Тут надо что-то сделать
        # print(possims)
        mull = posprobs * pmask
        # print(mull)
        posloss = F.binary_cross_entropy(mull, pmask.expand_as(posprobs))

        negwords = torch.randint(1, self.vsize, size=(batchcount, self.nsampl))
        negconembs = self.cemb(negwords).permute(0, 2, 1)
        negsims = torch.bmm(cenbeddings, negconembs)
        loss = F.binary_cross_entropy_with_logits(negsims, negsims.new_zeros(negsims.shape)) + posloss
        return loss
