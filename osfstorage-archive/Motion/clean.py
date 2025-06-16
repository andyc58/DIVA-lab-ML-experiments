import os
import pandas as pd

def print_file_names(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                df = df.iloc[1:, 1:]
                df.to_csv(file_path, index=False)

# Example usage
directory = os.getcwd()
print_file_names(directory)