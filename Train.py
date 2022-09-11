from torch.utils.data import DataLoader
import copy
import datetime
import traceback
import random
import numpy as np
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def fit(model, train_dataset, val_dataset, criterion,lr=1e-4, epoch_n=100, batch_size=32, l2_reg_alpha=0,shuffle_train=True,dataloader_workers_n=0,  lr_scheduler_ctor=None):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg_alpha)
    lr_scheduler = lr_scheduler_ctor(optimizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=dataloader_workers_n)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=dataloader_workers_n)
    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)
    for epoch in range(epoch_n):
        epoch_start = datetime.datetime.now()
        model.train()
        mean_train_loss = 0
        train_batches_n = 0
        for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            mean_train_loss += float(loss)
            train_batches_n += 1
        mean_train_loss /= train_batches_n
        model.eval()
        meanloss = 0
        valbatch= 0
        with torch.no_grad():
            for batch_i, (batch_x, batch_y) in enumerate(val_dataloader):
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                meanloss += float(loss)
                valbatch += 1
        meanloss =meanloss / valbatch
        if meanloss < best_val_loss:
            best_epoch_i = epoch
            best_val_loss = meanloss
            best_model = copy.deepcopy(model)
    return best_val_loss, best_model

def myloss(pred, target):
    return pred