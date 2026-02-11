import yaml
import os
import sys
from tasks.data_imputation import *
from constants.datasets import *
from tools.dual_output import *
from constants.paths import *
import warnings
import pandas as pd
import numpy as np
import openpyxl
warnings.filterwarnings("ignore")


def DI(config):
    print("Task: Data Imputation\n")

    repeat = config.get("repeat")
    result_path = os.path.join(RES_PATH, config.get("result_path"))

    dfs = []
    times = []
    for i in range(repeat):
        result = data_imputation(config, i)
        if result is None:
            print(f"Warning: Run {i+1} returned None, skipping...")
            continue

        df, t = result
        if df is not None:
            df = add_accuracy_row(df)
            dfs.append(df)
            times.append(t)

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

            print(
                "Runtime: " + " | ".join(f"{phase}: {runtime:.3f}s" for phase, runtime in t.items()))

    if dfs:
        # save results
        accuracy_stats_df = compute_accuracy_stats(dfs)
        with pd.ExcelWriter(result_path, mode='a', if_sheet_exists='new', engine='openpyxl') as writer:
            accuracy_stats_df.to_excel(
                writer, sheet_name=f'Results', index=False)

        # save runtime
        time_avg = pd.DataFrame([pd.DataFrame(times).mean()])
        with pd.ExcelWriter(result_path, mode='a', if_sheet_exists='new', engine='openpyxl') as writer:
            time_avg.to_excel(writer, sheet_name=f'Runtime', index=False)

        # save config
        flat_data = flatten_json(config)
        # Convert dict to DataFrame using from_dict with index orientation
        config_df = pd.DataFrame.from_dict(flat_data, orient='index')
        config_df.columns = ['Value']
        config_df.index.name = 'Key'
        config_df = config_df.reset_index()
        with pd.ExcelWriter(result_path, mode='a', if_sheet_exists='new', engine='openpyxl') as writer:
            config_df.to_excel(writer, sheet_name=f'Config', index=False)

    # df.to_csv(os.path.join(RES_PATH, config.get("result_path")), index=False)


def add_accuracy_row(df):

    # Identify non-metric columns (first three are always 'id', 'key', and a third one that may change)
    non_metric_cols = df.columns[:3]
    metric_cols = df.columns[3:]  # All remaining columns are metrics

    # Compute accuracy for metric columns
    # Compute mean (accuracy) and convert to DataFrame
    accuracy_row = df[metric_cols].mean().to_frame().T

    # Add placeholders for non-metric columns
    for col in non_metric_cols:
        accuracy_row[col] = ""  # "Accuracy" if col == "id" else ""

    # Ensure column order is preserved
    accuracy_row = accuracy_row[df.columns]

    # Append accuracy row to DataFrame
    return pd.concat([df, accuracy_row], ignore_index=True)


def compute_accuracy_stats(dfs):

    # Exclude 'id', 'key', and the third column
    metric_cols = dfs[0].columns[3:]
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
            # Compute the mean accuracy per DataFrame
            accuracies.append(np.mean(values))

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
            # Convert lists to comma-separated strings
            out[f"{prefix}{key}"] = ", ".join(map(str, value))
        else:
            out[f"{prefix}{key}"] = value
    return out


def read_config(config_path=None):
    if config_path is None:
        # Get the path to the current directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        config_path = os.path.join(root_dir, 'config.yaml')
    else:
        config_path = os.path.abspath(config_path)

    # Read the YAML file and convert it to a Python dictionary
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def set_output(config):
    os.makedirs(RES_PATH, exist_ok=True)
    output_path = os.path.join(RES_PATH, config.get("output_path"))
    result_path = os.path.join(RES_PATH, config.get("result_path"))

    out = dual_output(output_path)

    if os.path.exists(output_path):
        os.remove(output_path)

    if os.path.exists(result_path):
        os.remove(result_path)

    sys.stdout = out

    return


def validate_config(config: dict) -> bool:
    """
    Validate that the configuration contains required fields.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    if not config:
        print("Error: Configuration is empty or invalid.", file=sys.stderr)
        return False

    if "task" not in config:
        print("Error: 'task' field is missing from configuration.", file=sys.stderr)
        return False

    task = config.get("task")
    if task == "Data Imputation":
        required_fields = ["dataset", "model", "repeat"]
        missing_fields = [
            field for field in required_fields if field not in config]
        if missing_fields:
            print(
                f"Error: Missing required fields for Data Imputation: {', '.join(missing_fields)}", file=sys.stderr)
            return False

    return True


def main(config_path=None):
    """
    Main entry point for the application.
    Handles configuration loading, validation, and task execution.

    Args:
        config_path: Optional path to config file. If not provided, uses config.yaml in the project root.
    """
    try:
        # Load configuration
        print("Loading configuration...")
        config = read_config(config_path)

        if not validate_config(config):
            sys.exit(1)

        # Set up output redirection
        print("Setting up output...")
        set_output(config)

        # Execute task based on configuration
        task = config.get("task")
        print(f"\n{'='*60}")
        print(f"Starting task: {task}")
        print(f"{'='*60}\n")

        if task == "Data Imputation":
            DI(config)
            print(f"\n{'='*60}")
            print(f"Task '{task}' completed successfully!")
            print(f"{'='*60}")
        else:
            print(
                f"Error: Unknown task '{task}'. Supported tasks: 'Data Imputation'", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(
            f"Error: Invalid YAML in configuration file: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(
            f"Error: Missing required configuration key: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
