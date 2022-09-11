import random
import numpy as np
import re
import torch
import torch.nn as nn

class Gener:
    def __init__(self, embeddings, word2id):
        self.embeddings = embeddings
        self.embeddings /= (np.linalg.norm(self.embeddings, ord=2, axis=-1, keepdims=True) + 1e-4)
        self.word2id = word2id
        self.id2word = {i: w for w, i in word2id.items()}

    def similar(self, query_vector, topk=10):
        similarities = (self.embeddings * query_vector).sum(-1)
        best_indices = np.argpartition(-similarities, topk, axis=0)[:topk]
        result = [self.id2word[i] for i in best_indices]
        return result

    def get_vectors(self, *words):
        word_ids = [self.word2id[i] for i in words]
        vectors = np.stack([self.embeddings[i] for i in word_ids], axis=0)
        return vectors

    def generate(self, firstword):
        text = " "
        for i in range(30):
            firstword = self.self.embeddings[self.word2id[firstword]]
            firstword = self.embeddings.similar(firstword)
            new_token = np.random.choice(len(self.word2id))
            text = text + str(firstword[1]) + " "
            firstword = self.id2word[new_token]
        return text
