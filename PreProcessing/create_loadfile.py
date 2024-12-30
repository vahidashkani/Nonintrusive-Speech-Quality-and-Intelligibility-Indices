import os
import numpy as np
from tqdm import tqdm
import re
import random
import shutil
import pandas as pd
import csv

Train_patient_index = os.listdir("data/16k/train_speech")
Dev_patient_index = os.listdir("data/16k/dev_speech")
Test_patient_index = os.listdir("data/16k/test_speech")

supervised_data = pd.read_excel("data/all_PVQD_mode.xlsx")

for i in tqdm(Train_patient_index):
	patient_index = i.split("_")[0]

	for row in supervised_data.iterrows():
		if patient_index == row[1]["File"].strip():
			# print(row[1]["File"])
			# print(row[1]["File"]==patient_index)
			with open("data/label/train_speech_reg.csv","a+") as csvfile:
				writer =csv.writer(csvfile)
				input_data = ["train_speech/"+i,row[1]["Average_G"]]
				writer.writerow(input_data)
			'''with open("label/train_speech_cla.csv","a+") as csvfile:
				writer =csv.writer(csvfile)
				input_data = ["train_speech/"+i,row[1]["Category_G"],row[1]["Category_R"],row[1]["Category_B"],row[1]["Category_A"],row[1]["Category_S"]]
				# print(input_data)
				writer.writerow(input_data)'''

for i in tqdm(Dev_patient_index):
	patient_index = i.split("_")[0]

	for row in supervised_data.iterrows():
		if patient_index == row[1]["File"].strip():
			# print(row[1]["File"])
			# print(row[1]["File"]==patient_index)
			with open("data/label/dev_speech_reg.csv","a+") as csvfile:
				writer =csv.writer(csvfile)
				input_data = ["dev_speech/"+i,row[1]["Average_G"]]
				# print(input_data)
				writer.writerow(input_data)
			'''with open("label/dev_speech_cla.csv","a+") as csvfile:
				writer =csv.writer(csvfile)
				input_data = ["dev_speech/"+i,row[1]["Category_G"],row[1]["Category_R"],row[1]["Category_B"],row[1]["Category_A"],row[1]["Category_S"]]
				# print(input_data)
				writer.writerow(input_data)'''

for i in tqdm(Test_patient_index):
	patient_index = i.split("_")[0]

	for row in supervised_data.iterrows():
		if patient_index == row[1]["File"].strip():
			# print(row[1]["File"])
			# print(row[1]["File"]==patient_index)
			with open("data/label/test_speech_reg.csv","a+") as csvfile:
				writer =csv.writer(csvfile)
				input_data = ["test_speech/"+i,row[1]["Average_G"]]
				# print(input_data)
				writer.writerow(input_data)
			'''with open("label/test_speech_cla.csv","a+") as csvfile:
				writer =csv.writer(csvfile)
				input_data = ["test_speech/"+i,row[1]["Category_G"],row[1]["Category_R"],row[1]["Category_B"],row[1]["Category_A"],row[1]["Category_S"]]
				# print(input_data)
				writer.writerow(input_data)'''