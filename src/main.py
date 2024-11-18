import json
import os
from tasks.data_imputation import *
from constants.datasets import *
from tools.dual_output import *
from constants.paths import *
import warnings
import pandas as pd
import openpyxl
warnings.filterwarnings("ignore")

# MODELS = ["GPT 3.5"]  # , "GPT 4", "Llama", "Gemini",

# BUY_DATASET_PATH = os.path.join(
#     DATA_PATH, BUY_DATASET_CONSTANTS.VALUE['REL_PATH'])
# RESTAURANT_DATASET_PATH = os.path.join(
#     DATA_PATH, RESTAURANT_DATASET_CONSTANTS.VALUE['REL_PATH'])


def DI(config):
    print("Task: Data Imputation\n")
    
    repeat = config.get("repeat")
    result_path = os.path.join(RES_PATH, config.get("result_path"))
    for i in range(repeat):
        df = data_imputation(config)

        if not os.path.exists(result_path):
            # Write the first sheet if the file does not exist
            with pd.ExcelWriter(result_path, mode='w', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=f'Run {i+1}', index=False)
        else:
            # Append to the existing file
            with pd.ExcelWriter(result_path, mode='a', if_sheet_exists='new', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=f'Run {i+1}', index=False)
        
        print("\nResult Dataframe: ")
        print(df.head())
    
    #df.to_csv(os.path.join(RES_PATH, config.get("result_path")), index=False)
    


def read_config():
    # Get the path to the current directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the config.json file
    config_path = os.path.join(script_dir, 'config.json')

    # Read the JSON file and convert it to a Python dictionary
    with open(config_path, 'r') as file:
        config = json.load(file)

    return config


def set_output(config):
    output_path = os.path.join(RES_PATH, config.get("output_path"))
    result_path = os.path.join(RES_PATH, config.get("result_path"))

    out = dual_output(output_path)

    if os.path.exists(output_path):
        os.remove(output_path)
    
    if os.path.exists(result_path):
        os.remove(result_path)

    sys.stdout = out

    return


if __name__ == "__main__":

    config = read_config()

    set_output(config)

    if config.get("task") == "Data Imputation":
        DI(config)
