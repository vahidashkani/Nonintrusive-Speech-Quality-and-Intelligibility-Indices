#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:06:33 2024

@author: nca
"""

import os
import shutil

# Define the source directory where your folders are located
source_directory = '/media/nca/T7/ICASSP/ICASSP/PSAPStudy/50343/concat/'  # Change this to your source directory path

# Define the destination directory where all wav files will be copied
destination_directory = '/media/nca/T7/ICASSP/ICASSP/PSAPStudy/50343/combined_files/'  # Change this to your destination folder path

# Ensure the destination directory exists, if not, create it
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Loop through all the folders and subfolders in the source directory
for root, dirs, files in os.walk(source_directory):
    for file in files:
        if file.endswith('.wav'):  # Check if the file is a .wav file
            # Construct full file path
            file_path = os.path.join(root, file)
            
            # Copy the wav file to the destination folder
            shutil.copy(file_path, destination_directory)

print("All .wav files have been copied successfully!")
