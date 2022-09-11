import numpy as np
import matplotlib.pyplot as plt
import re
import torch
import random
import torch.nn as nn
import Prep
import Model
import Train
import generate
import traceback

from generate  import  Gener
from Model import SkipGram
from Prep import Prep
from Train import fit
from Train import myloss
from Prep import PadSeqData

def ReadRawTxt(parth):
	raw_text=dict()
	with open(parth, 'r') as f:
		raw_text = f.readlines()
		#txt=f.read()
	for line in range (len(raw_text)):
		raw_text[line]=raw_text[line].lower()
		raw_text[line]=re.findall(r'[\w\d]+',raw_text[line])

	#re.findall(pattern, string)
	#raw_text = [ for line in raw_text]
	return raw_text


def create_dictionary(text):
	char_to_idx = dict()
	idx_to_char = dict()

	idx = 0
	for char in text:
		if char not in char_to_idx.keys():
			char_to_idx[char] = idx
			idx_to_char[idx] = char
			idx += 1

	print("Vocab: ", len(char_to_idx))
	return char_to_idx, idx_to_char


parth ="C:\\Users\\vikto\\PycharmProjects\\pythonProject1\\HP3RUS.txt"
a=Prep(parth)
i,j=a.build_vocabulary()
b=ReadRawTxt(parth)
random.shuffle(b)
train_source = b[:int(len(b) * 0.7)]
train=a.Textstotokenids(train_source,i)
test_source = b[int(len(b) * 0.7):]
test=a.Textstotokenids(test_source,i)
train_dataset = PadSeqData(train,np.zeros(len(train)),lenofsent=20)
#print(train_dataset[9].item())
test_dataset = PadSeqData(test,np.zeros(len(test)),lenofsent=20)





trainer = SkipGram(len(i), 100, 20, rad=5, nsampl=25)
#self, vsize, esize, sentlen, rad=3, nsampl=1):
best_val_loss, model = fit(trainer, train_dataset,test_dataset,myloss,lr=1e-3,
                                            epoch_n=10,
                                            batch_size=8,
                                            lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True))


embeddings = Gener(trainer.cemb.weight.detach().cpu().numpy(), i)
print(embeddings.generate('я'))