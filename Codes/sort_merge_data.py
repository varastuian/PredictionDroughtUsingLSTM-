import os
import pandas as pd
import re

files = []

pattern = r"(\d{1,2})\D*(\d{1,2})\D*(\d{4})"

input_folder = "Data"
output_file = "Codes/header.csv"


for filename in os.listdir(input_folder):
    match = re.search(pattern, filename)
    if match:
        month = int(match.group(1))  # Extract month
        day = int(match.group(2))  # Extract day (optional)
        year = int(match.group(3))  # Extract year
        files.append((year, month, filename))  # Store the year, month, and filename

files.sort(key=lambda x: (x[0], x[1]))

header_written = False

with open(output_file, "w", newline="") as output_csv:
    for _, _, file_name in files:
            file_path = os.path.join(input_folder, file_name)
            try:
                tables = pd.read_html(file_path)
                df = tables[0]
                # df = df.iloc[:].reset_index(drop=True)
                df.to_csv(output_csv, index=False, header=not header_written,sep=",", mode="a")
                header_written = True 
                print(f"processing file {file_name}")
   
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

