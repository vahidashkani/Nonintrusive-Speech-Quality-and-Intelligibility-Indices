#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:18:21 2024

@author: nca
"""

import os
import numpy as np
from tqdm import tqdm
import re
import random
import shutil
import pandas as pd
import csv
import soundfile as sf
import librosa

source = os.listdir("data/16k")
# print(len(source))
for i in source:
	# print(i)
	audio_data = os.listdir(os.path.join("data/16k",i))

	for j in tqdm(audio_data):
		# print(j)
		data,sr = sf.read(os.path.join("data/16k",i,j))
		NUM = len(data)/sr
		AUDIO_LIST = []
		Ini = 0
		o=0
		while NUM>4:
			length = random.randint(2,4)
			sf.write(os.path.join("data/16k_slice",i,j.split(".")[0]+"_" + str(o)+".wav"), data[Ini*16000:(Ini+length)*16000],16000)
			NUM = NUM - length
			Ini = Ini + length
			o += 1
		if NUM >= 2:
			sf.write(os.path.join("data/16k_slice",i,j.split(".")[0]+"_" + str(o)+".wav"), data[Ini*16000:-1],16000)