#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:16:53 2024

@author: nca
"""

import os
import csv

# Define the directory where the wav files are located
wav_directory = '/media/nca/T7/ICASSP/ICASSP/PSAPStudy/50343/combined_files/'  # Change this to your wav directory path

# Define the path where the CSV file will be saved
csv_file_path = '/media/nca/T7/ICASSP/ICASSP/PSAPStudy/50343/50343_combined.csv'  # Change this to your desired output path

# Prefix to be added to each file name
prefix = 'test_speech/50343/combined_files/'

# List to store the modified file names
file_names_with_prefix = []

# Loop through the wav directory and get all wav files
for file_name in os.listdir(wav_directory):
    if file_name.endswith('.wav'):  # Check if the file is a .wav file
        # Add the prefix to the file name and store it in the list
        file_names_with_prefix.append(prefix + file_name)

# Write the file names to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['File Name'])  # Write header
    for file_name in file_names_with_prefix:
        writer.writerow([file_name])  # Write each file name with the prefix

print(f"File names have been written to {csv_file_path} with the prefix added.")
