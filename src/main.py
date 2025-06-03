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

def DI(config):
    print("Task: Data Imputation\n")
    
    repeat = config.get("repeat")
    result_path = os.path.join(RES_PATH, config.get("result_path"))
    
    dfs = []
    for i in range(repeat):
        df = data_imputation(config, i)
        if df is not None:
            df = add_accuracy_row(df)
            dfs.append(df)
            
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
    
    
    if dfs:
        #save results
        accuracy_stats_df = compute_accuracy_stats(dfs)
        with pd.ExcelWriter(result_path, mode='a', if_sheet_exists='new', engine='openpyxl') as writer:
                    accuracy_stats_df.to_excel(writer, sheet_name=f'Results', index=False)

        #save config
        flat_data = flatten_json(config)
        config_df = pd.DataFrame(flat_data.items(), columns=["Key", "Value"])
        with pd.ExcelWriter(result_path, mode='a', if_sheet_exists='new', engine='openpyxl') as writer:
                    config_df.to_excel(writer, sheet_name=f'Config', index=False)
    
    #df.to_csv(os.path.join(RES_PATH, config.get("result_path")), index=False)

def add_accuracy_row(df):

    # Identify non-metric columns (first three are always 'id', 'key', and a third one that may change)
    non_metric_cols = df.columns[:3]
    metric_cols = df.columns[3:]  # All remaining columns are metrics

    # Compute accuracy for metric columns
    accuracy_row = df[metric_cols].mean().to_frame().T  # Compute mean (accuracy) and convert to DataFrame

    # Add placeholders for non-metric columns
    for col in non_metric_cols:
        accuracy_row[col] = "" # "Accuracy" if col == "id" else "" 

    # Ensure column order is preserved
    accuracy_row = accuracy_row[df.columns]

    # Append accuracy row to DataFrame
    return pd.concat([df, accuracy_row], ignore_index=True)
 

def compute_accuracy_stats(dfs):

    metric_cols = dfs[0].columns[3:]  # Exclude 'id', 'key', and the third column
    metric_groups = {}

    for col in metric_cols:
        prefix = col.split(":")[0]  # Extract the part before ':'
        if prefix not in metric_groups:
            metric_groups[prefix] = []
        metric_groups[prefix].append(col)

    # Initialize dictionary to store accuracies for each metric group
    accuracy_results = {}

    for prefix, cols in metric_groups.items():
        accuracies = []
        for df in dfs:
            accuracy_row = df.iloc[-1]  # The last row contains accuracy values

            # Find the correct columns in this dataframe that match the metric prefix
            matched_cols = [c for c in df.columns if c.startswith(prefix)]
            if not matched_cols:  # Skip if no matching columns are found
                continue

            # Extract accuracy values only from matched columns
            values = [float(accuracy_row[c]) for c in matched_cols]  
            accuracies.append(np.mean(values))  # Compute the mean accuracy per DataFrame

        if accuracies:  # Avoid division by zero
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies, ddof=1)
            accuracy_results[prefix] = f"{mean_accuracy:.3f}±{std_accuracy:.3f}"
        else:
            accuracy_results[prefix] = "N/A"

    # Convert to DataFrame with a single row
    return pd.DataFrame([accuracy_results])



def flatten_json(y, prefix=""):
    # Normalize and flatten the JSON data for Excel
    out = {}
    for key, value in y.items():
        if isinstance(value, dict):
            out.update(flatten_json(value, f"{prefix}{key}."))
        elif isinstance(value, list):
            out[f"{prefix}{key}"] = ", ".join(map(str, value))  # Convert lists to comma-separated strings
        else:
            out[f"{prefix}{key}"] = value
    return out


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
