#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:18:34 2024

@author: nca
"""

import pandas as pd

def append_extension_to_filenames(csv_file_path, output_file_path, extension='.wav'):
    # Read the CSV file into a DataFrame, without a header
    df = pd.read_csv(csv_file_path, header=None)
    
    # Append the specified extension to the filenames in the first column
    df[0] = df[0].apply(lambda x: x + extension)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False, header=False)
    print(f"Updated CSV saved to {output_file_path}")



# Specify your input CSV file path and the output CSV file path
input_csv_file = '/home/nca/Downloads/PVQD/GRBASAssessment-main/dataset/sets/16k_slice_label/train_speech_reg.csv'  # Replace with your input CSV file path
output_csv_file = '/home/nca/Downloads/PVQD/GRBASAssessment-main/dataset/sets/16k_slice_label/train_speech_reg.csv'  # Replace with your desired output CSV file path

# Call the function
append_extension_to_filenames(input_csv_file, output_csv_file)
