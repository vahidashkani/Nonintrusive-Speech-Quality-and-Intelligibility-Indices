#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 08:56:24 2024

@author: nca
"""

import os
import pandas as pd

def convert_xlsx_to_csv(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is an xlsx file
        if filename.endswith('.xlsx'):
            xlsx_path = os.path.join(directory, filename)
            # Read the Excel file
            try:
                # Load the Excel file
                excel_data = pd.ExcelFile(xlsx_path)
                # Iterate through all sheets
                for sheet_name in excel_data.sheet_names:
                    # Read each sheet into a DataFrame
                    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
                    # Construct the CSV filename
                    csv_filename = f"{os.path.splitext(filename)[0]}_{sheet_name}.csv"
                    csv_path = os.path.join(directory, csv_filename)
                    # Save DataFrame to CSV
                    df.to_csv(csv_path, index=False)
                    print(f"Converted {xlsx_path} (sheet: {sheet_name}) to {csv_path}")
            except Exception as e:
                print(f"Error processing {xlsx_path}: {e}")

# Specify your directory containing the .xlsx files
directory_path = '/path/to/your/directory'
convert_xlsx_to_csv(directory_path)
