import os
import pandas as pd

filepath = './models/TV_POT/results/'

# Create an empty DataFrame
files = os.listdir(filepath)
file_path = os.path.join(filepath, files[0])
concatenated_data = pd.read_csv(file_path)

# Loop through each file in the filepath
for filename in files[1:]:

    if filename.endswith('.csv'):
        # Read the CSV file into a DataFrame
        file_path = os.path.join(filepath, filename)
        df = pd.read_csv(file_path)
        
        # Append the data to the concatenated DataFrame
        for col in df.columns:
            if col not in concatenated_data.columns:
                concatenated_data[col] = df[col]
        
concatenated_data.to_csv(f'{filepath}concatenated.csv', index=False)