import os
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tools.comparison_metrics import *
from main import compute_accuracy_stats
import pandas as pd
import numpy as np
import nltk
# nltk.download('brown')
# nltk.download('reuters')
# nltk.download('gutenberg')
# nltk.download('punkt_tab')

def process_run_sheet(df):
    """ Process individual 'Run' sheets to add BLEU and ROUGE metrics. """
    if df.shape[1] < 4:
        raise ValueError("Unexpected sheet format: At least 4 columns are required.")

    # Identify columns
    id_col, key_col, model_output_col, exact_match_col = df.columns[:4]
    
    # Extract metric suffix from 'exact_match:xyz'
    metric_suffix = exact_match_col.split(":")[-1]

    # Keep only relevant columns
    df = df.iloc[:, :4]  

    # New column names with suffix
    bleu_col = f"bleu:{metric_suffix}"
    rouge_p_col = f"rouge_p:{metric_suffix}"
    rouge_r_col = f"rouge_r:{metric_suffix}"
    rouge_f1_col = f"rouge_f1:{metric_suffix}"

    # Compute BLEU and ROUGE scores
    bleu_scores = []
    rouge_p_scores = []
    rouge_r_scores = []
    rouge_f1_scores = []

    for _, row in df.iterrows():
        reference, candidate = str(row[key_col]), str(row[model_output_col])

        bleu = compute_bleu(reference, candidate)
        rouge = compute_rouge(reference, candidate)

        bleu_scores.append(bleu)
        rouge_p_scores.append(rouge["P"])
        rouge_r_scores.append(rouge["R"])
        rouge_f1_scores.append(rouge["F1"])

    # Add new metric columns
    df[bleu_col] = bleu_scores
    df[rouge_p_col] = rouge_p_scores
    df[rouge_r_col] = rouge_r_scores
    df[rouge_f1_col] = rouge_f1_scores

    # Compute averages and update the last row
    metric_cols = [exact_match_col, bleu_col, rouge_p_col, rouge_r_col, rouge_f1_col]
    df.iloc[-1, :3] = [np.nan, np.nan, np.nan]
    df.iloc[-1, 3:] = df[metric_cols].mean().values

    return df

def post_process_excel(file_path):
    """ Process an Excel file to update 'Run' and 'Results' sheets while keeping 'config' unchanged. """
    with pd.ExcelFile(file_path) as xls:
        sheet_names = xls.sheet_names

        # Identify sheets
        run_sheets = [name for name in sheet_names if name.startswith("Run")]
        results_sheet = "Results"
        config_sheet = "Config" if "Config" in sheet_names else None  # Check if 'config' exists

        if not run_sheets:
            raise ValueError("No 'Run' sheets found!")

        # Process 'Run' sheets
        run_dfs = {sheet: process_run_sheet(pd.read_excel(xls, sheet_name=sheet)) for sheet in run_sheets}

        # Read the 'config' sheet if it exists
        config_df = pd.read_excel(xls, sheet_name=config_sheet) if config_sheet else None
        if config_df is not None:
            # Find the row where the key is 'evaluate.methods'
            config_df.loc[config_df['Key'] == 'evaluate.methods', 'Value'] = 'exact_match, bleu_score, rouge_score'

        # Compute summary for 'Results' sheet
        results_df = compute_accuracy_stats(list(run_dfs.values()))

    # Save the modified Excel file
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # Write 'Run' sheets
        for sheet, df in run_dfs.items():
            df.to_excel(writer, sheet_name=sheet, index=False)

        # Write 'Results' sheet
        results_df.to_excel(writer, sheet_name=results_sheet, index=False)

        # Write 'config' sheet back without modifications
        if config_df is not None:
            config_df.to_excel(writer, sheet_name=config_sheet, index=False)

    print(f"Processing complete. Updated file saved: {file_path}")


def process_excel_files_in_directory(directory):
    """ Process all Excel files in a directory and its subdirectories. """
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                try:
                    post_process_excel(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Example usage
directory = "post_process" 
process_excel_files_in_directory(directory)

