#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 10:09:12 2024

@author: nca
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import HubertModel
from transformers import AutoFeatureExtractor, WhisperModel

from model import GRBASPredictor
from dataset import PVQDDataset

# Directories
datadir = "/home/nca/Downloads/PVQD/GRBASAssessment-main/dataset/"
ckptdir = '/home/nca/Downloads/PVQD/GRBASAssessment-main/checkpoints/pvqd/'

if not os.path.exists(ckptdir):
    os.makedirs(ckptdir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE: ' + str(device))

# PVQD-S
trainlist = os.path.join(datadir, 'sets/16k_slice_label/train_speech_reg.csv')
validlist = os.path.join(datadir, 'sets/16k_slice_label/dev_speech_reg.csv')

# PVQD-A
# trainlist = os.path.join(datadir, 'sets/16k_slice_label/train_a_reg.csv')
# validlist = os.path.join(datadir, 'sets/16k_slice_label/dev_a_reg.csv')

# SSL module
ssl_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
# ssl_model = HubertModel.from_pretrained("rinna/japanese-hubert-base")

# ASR module
asr_model = WhisperModel.from_pretrained("openai/whisper-small", output_attentions=True)
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
decoder_input_ids = torch.tensor([[1, 1]]) * asr_model.config.decoder_start_token_id
decoder_input_ids = decoder_input_ids.to(device)

# Freeze the pre-trained weight
# for param in ssl_model.base_model.parameters():
#     param.requires_grad = False
ssl_out_dim = 768

grbas_dim = 1  # 1: Grade  5: GRBAS
multi_indicator = False

trainset = PVQDDataset(datadir, trainlist, feature_extractor, multi_indicator)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1, collate_fn=trainset.collate_fn)

validset = PVQDDataset(datadir, validlist, feature_extractor, multi_indicator)
validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=1, collate_fn=validset.collate_fn)

net = GRBASPredictor(ssl_model, asr_model, decoder_input_ids, ssl_out_dim, grbas_dim)
#print("Model output shape:", net)
net = net.to(device)

criterion = nn.L1Loss(reduction='sum')
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

orig_patience = 20
patience = orig_patience
PREV_VAL_LOSS = float('inf')

for epoch in range(1, 200):
    STEPS = 0
    net.train()
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        asr_mel_features, inputs, mel_specgrams, labels, filenames = data
        
        asr_mel_features = asr_mel_features.to(device)
        inputs = inputs.to(device)
        mel_specgrams = mel_specgrams.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(asr_mel_features, inputs, mel_specgrams)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        STEPS += 1
        running_loss += loss.item()
    
    print(f'Epoch: {epoch} Averaged train loss: {running_loss / STEPS}')
    
    epoch_val_loss = 0.0
    net.eval()

    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # Validation
    VALSTEPS = 0
    for i, data in enumerate(validloader, 0):
        VALSTEPS += 1
        asr_mel_features, inputs, mel_specgrams, labels, filenames = data
        
        asr_mel_features = asr_mel_features.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        mel_specgrams = mel_specgrams.to(device)
        
        outputs = net(asr_mel_features, inputs, mel_specgrams)
        loss = criterion(outputs, labels)
        epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / VALSTEPS
    scheduler.step(avg_val_loss)
    print(f'Averaged val loss: {avg_val_loss}')
    
    if avg_val_loss < PREV_VAL_LOSS:
        print('Loss has decreased')
        PREV_VAL_LOSS = avg_val_loss
        PATH = os.path.join(ckptdir, f'ckpt_{epoch}.pth')
        torch.save(net.state_dict(), PATH)
        patience = orig_patience
    else:
        patience -= 1
        if patience == 0:
            print(f'Loss has not decreased for {orig_patience} epochs; early stopping at epoch {epoch}')
            break

print('Finished Training')
