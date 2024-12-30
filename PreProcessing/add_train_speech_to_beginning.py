#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:54:52 2024

@author: nca
"""

import pandas as pd

def prepend_to_filenames(csv_file_path, output_file_path, prepend_str='train_speech/'):
    # Read the CSV file into a DataFrame, without a header
    df = pd.read_csv(csv_file_path, header=None)
    
    # Prepend the specified string to the filenames in the first column
    df[0] = df[0].apply(lambda x: prepend_str + x)
    
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False, header=False)
    print(f"Updated CSV saved to {output_file_path}")

# Specify your input CSV file path and the output CSV file path
input_csv_file = 'input_file.csv'  # Replace with your input CSV file path
output_csv_file = 'output_file.csv'  # Replace with your desired output CSV file path

# Call the function
prepend_to_filenames(input_csv_file, output_csv_file)
