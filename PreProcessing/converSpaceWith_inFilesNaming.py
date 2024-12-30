#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 10:35:43 2024

@author: nca
"""

import os

# Specify the directory containing the .wav files
directory = '/home/nca/Downloads/PVQD/Audio Files/'

# Loop through all the files in the directory
for filename in os.listdir(directory):
    # Check if the file is a .wav file and contains spaces
    if filename.endswith('.wav') and ' ' in filename:
        # Replace spaces with underscores
        new_filename = filename.replace(' ', '_')
        # Construct full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')
