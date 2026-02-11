import pandas as pd
import os


def read_csv(dataset_path):
    try:
        # Reading the CSV file
        dataframe = pd.read_csv(dataset_path)
        return dataframe
    except FileNotFoundError:
        print(f"File not found at path: {dataset_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
