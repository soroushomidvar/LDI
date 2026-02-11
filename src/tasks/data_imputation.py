from models.model_execution import *
from tools.serializer import *
from tools.data_handler import *
from tools.entity_recognition import *
from tools.sketch_generator import *
from tools.dependency_finder import *
from tools.knn_finder import *
from tools.comparison_metrics import *
import json
import string
import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from constants.datasets import *
from constants.prompts import *
from constants.paths import *
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import ast
from sklearn.impute import SimpleImputer
import unicodedata
import re
import time
from typing import Dict, Any, Tuple
import tiktoken


def preprocessing(dataset_name, df):
    if dataset_name == DATASETS[dataset_name]['NAME']:
        df = df[DATASETS[dataset_name]['ALL_COLUMNS']]
        key = df[DATASETS[dataset_name]['KEY']]
        rest_cols = [x for x in DATASETS[dataset_name]['ALL_COLUMNS'] if x not in DATASETS[dataset_name]['KEY']]
        rest = df[rest_cols]
    return key, rest, df

def rule_generator_old(dataset_name, rest, df, shot_number, method='random', target_row=-1, annotation='GPT 3.5'):
    prompt = ''
    fixed_initial = ''
    serialized_samples = ''
    serialized_target_row = ''
    fixed_querry = ''

    # def dataset_specific_prompt_generator():

    #     if dataset_name == BUY_DATASET_CONSTANTS.VALUE['NAME']:
    #         fixed_initial = BUY_DATASET_PROMPTS.VALUE['FIXED_INITIAL']
    #         fixed_querry = BUY_DATASET_PROMPTS.VALUE['FIXED_QUERRY']

    #     elif dataset_name == RESTAURANT_DATASET_CONSTANTS.VALUE['NAME']:
    #         fixed_initial = RESTAURANT_DATASET_PROMPTS.VALUE['FIXED_INITIAL']
    #         fixed_querry = RESTAURANT_DATASET_PROMPTS.VALUE['FIXED_QUERRY']

    #     return fixed_initial, fixed_querry

    def serializer():
        serialized_target_row = serialize_rows(rest, [target_row])
        if method == 'random':
            random_indices = np.random.randint(
                0, len(df)-1, size=shot_number)
            serialized_samples = serialize_rows(df, random_indices)
        return str(serialized_target_row), str(serialized_samples)

    def shot_handler():  # TODO
        # serialized_target_row_entities = json.dumps(entity_recognizer(
        #     serialized_target_row, "LLM", models=["GPT 3.5"]))

        annotated_target_row = text_annotator(
            serialized_target_row, annotation)

        if shot_number == 0:
            prompt = str(fixed_initial) + \
                str(annotated_target_row) + ', ' + fixed_querry
        elif shot_number == 1:
            fixed_one_shot_text = FIXED_ONE_SHOT_TEXT
            prompt = fixed_initial + fixed_one_shot_text + serialized_samples + \
                ', ' + serialized_target_row + ', ' + fixed_querry
        elif shot_number > 1:
            fixed_few_shot_text = FIXED_FEW_SHOT_TEXT
            prompt = fixed_initial + fixed_few_shot_text + serialized_samples + \
                ', ' + serialized_target_row + ', ' + fixed_querry
        return prompt

    # fixed_initial, fixed_querry = dataset_specific_prompt_generator()
    serialized_target_row, serialized_samples = serializer()
    named_entities = named_entity_recognizer(serialized_target_row, annotation)
    prompt = shot_handler()

    return prompt, serialized_target_row, named_entities

def sampling(df, sample_size, apply_random_seed):
    if sample_size is None:
        samples = np.arange(len(df))  # all data
    else:
        np.random.seed(apply_random_seed)
        samples = np.random.choice(
            np.arange(len(df)-1), size=sample_size, replace=False)
    return samples

def example_generator(method, test_df, train_df, src, trg, example_method, example_rows, examples_number, random_seed, sample, verbose=False):
    if examples_number > train_df[trg].nunique():
        raise ValueError(
            "The number of unique examples requested exceeds the number of unique values in the trg column.")

    columns=[]
    if method == 'ALL':
        columns= train_df.columns
        # unique_examples = pd.DataFrame(columns=train_df.columns)
    if method == 'LCS':
        columns = src + [trg]
        # if column_selection == 'single':
        #     unique_examples = pd.DataFrame(columns = [src, trg])
        # elif column_selection == 'multi':
    unique_examples = pd.DataFrame(columns = columns)
    train_df = train_df.reset_index(drop=True) # Ensure sequential indices!

    if example_method == "similarity":
        columns_without_trg = pd.Index([col for col in columns if col != trg])
        example_rows, _ = find_top_k_similar_rows(test_df, train_df, sample, examples_number, columns_without_trg, trg, False, verbose=verbose)
    
    if example_method == "diverse_similarity":
        columns_without_trg = pd.Index([col for col in columns if col != trg])
        start_time = time.time()
        example_rows, _ = find_top_k_similar_rows(test_df, train_df, sample, examples_number, columns_without_trg, trg, True, verbose=verbose)
        end_time = time.time()
        if verbose:
            print("Time taken for finding similar examples: {:.6f} seconds".format(end_time - start_time))


    # unique_examples = pd.DataFrame(columns=df.columns if method == 'ALL' else [src, trg])
    selected_trg_values = set()
    rng = random.Random(random_seed)

    if example_method == "random":
        example_rows = rng.sample(list(train_df.index), examples_number)

    k = 0 # manual method
    while len(unique_examples) < examples_number:
        # Randomly select a row index
        
        #random_index = random.choice(df.index)

        #random_index = np.random.choice(df.index)
        if example_method == "diversity":
            i = rng.choice(train_df.index)

        elif example_method == "manual" or example_method == "random" or example_method == "similarity" or example_method == "diverse_similarity":
            i = example_rows[k]
            k += 1

        # print(f"Trying to access row {i} and column {trg}")
        # print(train_df.iloc[i])
        # Get the trg value of the randomly selected row
        trg_value = train_df.loc[i, trg]# train_df.iloc[i][trg] #train_df.loc[i, trg]

        # If the trg value has not been selected before, add the row to unique_examples
        if example_method == "manual" or example_method == "similarity" or example_method == "random" or trg_value not in selected_trg_values:
            selected_trg_values.add(trg_value)
            if method == 'LCS':
                # if column_selection == 'single':
                #     selected_row = df.loc[[i], [src, trg]]
                # elif column_selection == 'multi':
                selected_row = train_df.iloc[i, train_df.columns.get_indexer(src + [trg])] #train_df.loc[[i], src + [trg]]
            elif method == 'ALL':
                selected_row = train_df.iloc[i, :] #train_df.loc[[i], :] 
            selected_row = selected_row.to_frame().T
            unique_examples = pd.concat([unique_examples, selected_row])

    return unique_examples.reset_index(drop=True)

def text_to_vector(text, model):
    """Convert text to a fixed-size vector using the provided Word2Vec model."""
    tokens = text.split()  # Simple tokenization
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) == 0:  # Handle case where no tokens are in the model's vocabulary
        return np.zeros(model.vector_size)
    # Average the vectors
    vector = np.mean(vectors, axis=0)
    return vector

def dependency_finder(method, df_main, target_col, p, q, verbose=False):
    
    df = df_main.copy()
    rels = {}

    start = time.time()

    if method == "LCS":
        # Check if the target column is categorical
        if not pd.api.types.is_string_dtype(df[target_col]):
            raise ValueError("The target column must be categorical.")

        # Encode categorical columns, including the target column
        for column in df.columns:
            if column != target_col:  # df[column].dtype == 'object' and
                rel, res = is_dependant(df[[column, target_col]], p, q)
                if (rel):
                    rels[column] = 1
                    if verbose:
                        print("\n" + str(column) + ": " + "\nStatus: " + str(rel) + "\nLCSs: " + str(res)) #  "\nFrequency: " + str(freq) +
                else:
                    if verbose:
                        print("\n" + str(column) + ": " + "\nStatus: " + str(rel)) # "\nFrequency: " + str(freq)
    elif method == None: 
        for column in df.columns:
            rels[column] = -1
    
    end = time.time()
    runtime = round(end - start, 3)

    return rels, runtime  # feature_importance_dict

def dependency_finder_combinations_random_forest(df_main, target_col):
    df = df_main.copy()
    # Check if the target column is categorical
    # if df[target_col].dtype != 'object':
    #     raise ValueError("The target column must be categorical.")
    # Encode categorical columns, including the target column
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_combinations = []
    accuracy_scores = []

    # Iterate through all possible combinations of features
    for i in range(1, len(X.columns) + 1):
        for combo in itertools.combinations(X.columns, i):
            # Select the features in the current combination
            X_combo = X[list(combo)]

            # Initialize and train the RandomForestClassifier
            model = RandomForestClassifier()

            # Perform cross-validation to evaluate model performance
            scores = cross_val_score(
                model, X_combo, y, cv=5, scoring='accuracy')
            avg_score = np.mean(scores)

            # Store the combination and its corresponding score
            feature_combinations.append(combo)
            accuracy_scores.append(avg_score)

    # Create a dictionary with feature combinations and their corresponding accuracy scores
    combination_importance_dict = {combo: score for combo, score in zip(
        feature_combinations, accuracy_scores)}

    return combination_importance_dict

def dependency_finder_combinations_decision_tree(df_main, target_col):
    df = df_main.copy()
    # Check if the target column is categorical
    # if df[target_col].dtype != 'object':
    #     raise ValueError("The target column must be categorical.")
    # Encode categorical columns, including the target column
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_combinations = []
    accuracy_scores = []

    # Iterate through all possible combinations of features
    for i in range(1, len(X.columns) + 1):
        for combo in itertools.combinations(X.columns, i):
            # Select the features in the current combination
            X_combo = X[list(combo)]

            # Initialize and train the DecisionTreeClassifier
            model = DecisionTreeClassifier()

            # Perform cross-validation to evaluate model performance
            scores = cross_val_score(
                model, X_combo, y, cv=5, scoring='accuracy')
            avg_score = np.mean(scores)

            # Store the combination and its corresponding score
            feature_combinations.append(combo)
            accuracy_scores.append(avg_score)

    # Create a dictionary with feature combinations and their corresponding accuracy scores
    combination_importance_dict = {combo: score for combo, score in zip(
        feature_combinations, accuracy_scores)}

    return combination_importance_dict


def assign_probabilities_to_sketch(sketch, dependencies):
    result = {}
    for key, values in sketch.items():
        if key in dependencies:
            result[key] = {}
            for value in values:
                result[key][value] = dependencies[key]
    return result


def assign_probabilities_to_sketch_minimal(columns, dependencies):
    result = {}
    max_category = None
    max_value = -float('inf')

    for column in columns:
        if column in dependencies:
            result[column] = dependencies[column]
            # Check if the current value is greater than the current max
            if dependencies[column] > max_value:
                max_value = dependencies[column]
                max_category = column

    # Return the hierarchical dictionary for the category with the highest probability value
    if max_category is not None:
        return {max_category: result[max_category]}, max_category
    else:
        return {}, None


def get_top_n_features(feature_importance_dict, n):
    # Sort the dictionary by values in descending order and get the top n keys
    sorted_features = sorted(
        feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    if n!= -1:
        top_n_keys = [item[0] for item in sorted_features[:n]]
    else: 
        top_n_keys = [item[0] for item in sorted_features[:]]
    return top_n_keys


def rules_to_str(lst, key):
    result = []
    for i, item in enumerate(lst):
        if i == len(lst) - 1:
            result.append(f"{item} -> {key}")
        else:
            result.append(f"{item} -> {key}, ")
    return ''.join(result)


def calculate_data_reduction(df, selected_columns, trg):
    """
    Calculate data reduction ratio based on characters and tokens.
    Returns reduction ratios (0-1) for characters and tokens.
    Shows how much data (characters/tokens) is in unused columns vs all columns.
    
    Args:
        df: The dataframe
        selected_columns: List of columns that were selected (used)
        trg: Target column name (excluded from calculations)
    """
    # Initialize tiktoken encoder
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except:
        encoding = tiktoken.get_encoding("gpt2")
    
    # Get ALL columns in the dataframe (excluding target) as baseline
    all_columns = [col for col in df.columns if col != trg]
    
    # Get unused columns (columns not selected, excluding target)
    selected_set = set(selected_columns)
    unused_columns = [col for col in all_columns if col not in selected_set]
    
    # If no unused columns, return 0.0
    if len(unused_columns) == 0:
        return 0.0, 0.0
    
    # Calculate total characters and tokens for all available columns (excluding target)
    total_chars = 0
    total_tokens = 0
    
    for col in all_columns:
        if col in df.columns and df[col].dtype == 'object':  # String columns only
            col_data = df[col].dropna()
            # Total characters
            total_chars += col_data.apply(len).sum()
            # Total tokens
            def count_tokens(text):
                if pd.isna(text):
                    return 0
                return len(encoding.encode(str(text)))
            total_tokens += col_data.apply(count_tokens).sum()
    
    # Calculate unused characters and tokens
    unused_chars = 0
    unused_tokens = 0
    
    for col in unused_columns:
        if col in df.columns and df[col].dtype == 'object':  # String columns only
            col_data = df[col].dropna()
            # Unused characters
            unused_chars += col_data.apply(len).sum()
            # Unused tokens
            def count_tokens(text):
                if pd.isna(text):
                    return 0
                return len(encoding.encode(str(text)))
            unused_tokens += col_data.apply(count_tokens).sum()
    
    # Calculate reduction ratios (0-1)
    char_reduction = unused_chars / total_chars if total_chars > 0 else 0.0
    token_reduction = unused_tokens / total_tokens if total_tokens > 0 else 0.0
    
    return char_reduction, token_reduction


def rule_generator(dataset_name, df, trg, method, p, q, number_of_rules=3, verbose=False):

    dependency_values, runtime = dependency_finder(method, df, trg, p, q, verbose=verbose)

    if verbose:
        print("\nColumn Dependencies (Importance):")
        print(list(dependency_values.keys()))

    left_hand_columns = get_top_n_features(dependency_values, number_of_rules)

    left_hand_columns_str = rules_to_str(left_hand_columns, trg)

    # Always show selected attributes
    print("\nSelected attributes: " + ", ".join(f"'{col}'" for col in left_hand_columns))

    if verbose:
        print("Rule(s):")
        print(left_hand_columns_str)

    return left_hand_columns, runtime


def fill_keys(df, samples, trg, key_response_pairs):
    for sample in samples:
        key = df.iloc[sample][trg]
        key_response_pair = pd.DataFrame({'id': sample, 'key': [key]})
        key_response_pairs = pd.concat(
            [key_response_pairs, key_response_pair], ignore_index=True)

    return key_response_pairs


def get_sample_by_row_number(df, row_number):
    if row_number < 0 or row_number >= len(df):
        raise IndexError("Row number out of bounds.")

    # Select the row by its index
    row_df = df.iloc[[row_number]]

    return row_df.reset_index(drop=True)


def is_atomic(df, method, sample_size, threshold):
    if sample_size is None:
        sampled_df = df
    else:
        sampled_df = df.sample(n=sample_size)

    atomic_status = []

    for column in sampled_df.columns:
        column_atomic = True
        if method is not None:
            threshold_value = int(len(sampled_df[column]) * (1-threshold))
            count = 0

            print("Column Name: " + column)

            for value in sampled_df[column]:
                entity_types = named_entity_recognizer(
                    str(value), method, "list")
                entity_types_list = ast.literal_eval(str(entity_types))
                print(str(value) + " " + str(entity_types_list))
                if len(entity_types_list) > 1:
                    count += 1
                    if count > threshold_value:
                        column_atomic = False
                        break

        atomic_status.append(column_atomic)

    return atomic_status


def entity_extractor(df, atomic_status, sample_size, threshold):

    def entity_extractor_helper(text, entity_type):
        try:
            entities = named_entity_recognizer(
                str(text), "GPT 3.5", "dictionary")
            entities_dict = ast.literal_eval(entities)
        except:
            entities_dict = {}
        return ', '.join(str(entities_dict.get(entity_type, [])))

    if sample_size is None:
        sampled_df = df
    else:
        sampled_df = df.sample(n=sample_size)

    new_columns = {}

    for column, is_atomic in zip(sampled_df.columns, atomic_status):
        if is_atomic:
            # If the column is atomic, keep it as is.
            new_columns[column] = df[column]
        else:
            # Analyze non-atomic columns
            entity_count = {}

            for value in sampled_df[column]:
                entities_str = named_entity_recognizer(
                    str(value), "GPT 3.5", "dictionary")
                # print(entities_str)

                try:
                    entities = ast.literal_eval(entities_str)
                except:
                    entities = {}

                for entity_type, entity_list in entities.items():
                    if entity_type not in entity_count:
                        entity_count[entity_type] = 0
                    entity_count[entity_type] += 1

            # Determine threshold count
            threshold_count = threshold * sample_size

            # Filter entity types that exceed the threshold
            common_entities = [
                etype for etype, count in entity_count.items() if count > threshold_count]

            if not common_entities:
                # If no common entities exceed the threshold, treat column as atomic
                new_columns[column] = df[column]
            else:
                # Create new columns for each common entity type
                # for entity_type in common_entities:
                #     new_column_name = f"{column}:{entity_type}"
                #     new_columns[new_column_name] = df[column].apply(
                #         lambda x: ', '.join(ast.literal_eval(named_entity_recognizer(
                #             str(x), "GPT 3.5", "dictionary")).get(entity_type, []))
                #     )
                for entity_type in common_entities:
                    new_column_name = f"{column}:{entity_type}"
                    # Use functools.partial to create a function with a preset entity_type argument
                    from functools import partial
                    extract_specific_entities = partial(
                        entity_extractor_helper, entity_type=entity_type)
                    new_columns[new_column_name] = df[column].apply(
                        extract_specific_entities)

    # Create new dataframe with the extracted columns
    new_df = pd.DataFrame(new_columns)

    return new_df


def mpping_handler(method, rule, trg, model, examples, sample, verbose=False):

    source_columns = [col for col in examples.columns if col != trg] #examples.columns[:-1]  # All columns except the last one
    target_column = trg #examples.columns[-1] 
    src_names_str = ', '.join(source_columns)
    rule = src_names_str + ' -> ' + target_column
    
    initial_prompt = MAPPING_HANDLER_PROMPTS.VALUE['MAPPING_HANDLER_INITIAL']
    initial_prompt = initial_prompt.replace(
        '<src>', src_names_str).replace('<trg>', target_column).replace('<rule>', rule)
    serialized_examples = serialize_rows(examples)

    # src_name = examples.columns[0]
    # sample_src = sample[[src_name]]
    sample_src = sample[source_columns]
    serialized_target_row = serialize_rows(sample_src)
    
    middle_prompt = MAPPING_HANDLER_PROMPTS.VALUE['MAPPING_HANDLER_MIDDLE'].replace('<trg>', target_column)
    fixed_prompt = MAPPING_HANDLER_PROMPTS.VALUE['MAPPING_HANDLER_QUERRY'].replace('<trg>', target_column)
    prompt = initial_prompt + serialized_examples + middle_prompt + serialized_target_row + fixed_prompt
    response = prompt_runner(model, prompt)
    print('prompt: ' + prompt)
    print(f"response ({model}): {response}")
    print()
    return response


def example_sampling(train_df, trg, examples_sample_size, examples_random_seed):
    unique_trg_values = train_df[trg].unique()
    
    if len(unique_trg_values) > examples_sample_size:
        # Randomly select `examples_sample_size` unique trg values
        selected_trg_values = pd.Series(unique_trg_values).sample(n=examples_sample_size, random_state=examples_random_seed)
        # Pick one row for each selected trg value
        sampled_df = train_df[train_df[trg].isin(selected_trg_values)].groupby(trg).sample(n=1, random_state=examples_random_seed)
    else:
        # Pick one row from each unique trg value
        sampled_df = train_df.groupby(trg).sample(n=1, random_state=examples_random_seed)
        remaining_size = examples_sample_size - len(sampled_df)
        
        if remaining_size > 0:
            # Sample additional rows randomly to reach the desired sample size
            additional_samples = train_df.drop(sampled_df.index).sample(n=remaining_size, random_state=examples_random_seed, replace=False)
            sampled_df = pd.concat([sampled_df, additional_samples])
    
    return sampled_df

def run_models(config, test_df, train_df, src_list, trg, key_response_pairs, rule, samples, run_number):

    method = config.get("dependency_finder", {}).get("method")
    model = config.get("model")
    #column_selection = config.get("dependency_finder", {}).get("column_selection")
    example_method = config.get("examples", {}).get("method")
    example_rows = config.get("examples", {}).get("rows")
    examples_sample_size = config.get("examples", {}).get("sample_size")
    number_of_examples = config.get("examples", {}).get("number_of_examples")
    examples_random_seed = config.get("examples", {}).get("random_seed") + run_number
    verbose = config.get("verbose", False)
    
    if examples_sample_size == None:
        sampled_train_df = train_df
    else:
        sampled_train_df = example_sampling(train_df, trg, examples_sample_size, examples_random_seed)
    
    if verbose:
        print("Selected dataframe for example generation: ")
        print(sampled_train_df.head(10))
        print("# Rows: " + str(len(sampled_train_df)))

    #sampled_train_df = train_df.sample(n=examples_sample_size, frac=None if examples_sample_size else 1, random_state=examples_random_seed)    
    
    times= []

    for sample in samples:

        # if method == 'LCS' and column_selection == 'single':
        #     apply_examples_df = example_generator(method, column_selection, df, src, trg, example_method, example_rows, number_of_examples, examples_random_seed)
        # if method == 'LCS': #and column_selection == 'multi':
        start_time = time.time()
        examples = example_generator(method, test_df, sampled_train_df, src_list, trg, example_method, example_rows, number_of_examples, examples_random_seed, sample, verbose=verbose)
        end_time = time.time()
        times.append(end_time - start_time)
        # elif method == 'ALL':
        #     apply_examples_df = example_generator(method, df, None, trg, example_method, example_rows, number_of_examples, examples_random_seed)
        

        value = mpping_handler(
            method, rule, trg, model, examples, get_sample_by_row_number(test_df, sample), verbose=verbose)
        
        key_response_pairs.loc[key_response_pairs['id']
                               == sample, rule] = value  # detect_value_from_response(rule)

    example_generation_avg_time = round(sum(times) / len(samples), 3)

    return key_response_pairs, example_generation_avg_time

def print_dataframe_info(df):
    print("### DataFrame Information ###\n")
    
    # Number of rows and columns
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}\n")
    
    # Column names
    print("Column Names:")
    for col in df.columns:
        print(f" - {col}")
    print()
    
    # Average Length of Values per Column
    print("Average Length of Values per Column:")
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            avg_len = df[col].dropna().apply(len).mean()
            print(f" - {col}: {avg_len:.2f} characters (average)")
        else:
            print(f" - {col}: Not applicable (non-string column)")
    print()
    
    # Average Number of Tokens per Column
    print("Average Number of Tokens per Column:")
    # Initialize tiktoken encoder (using cl100k_base which is used by GPT-4 and GPT-3.5)
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except:
        # Fallback to default encoding if cl100k_base is not available
        encoding = tiktoken.get_encoding("gpt2")
    
    for col in df.columns:
        if df[col].dtype == 'object':  # String columns
            # Count tokens for each non-null value
            def count_tokens(text):
                if pd.isna(text):
                    return 0
                return len(encoding.encode(str(text)))
            
            token_counts = df[col].dropna().apply(count_tokens)
            avg_tokens = token_counts.mean()
            print(f" - {col}: {avg_tokens:.2f} tokens (average)")
        else:
            print(f" - {col}: Not applicable (non-string column)")
    print()
    
    # Data types
    print("Data Types:")
    print(df.dtypes)
    print()
    
    # General overview using df.info()
    print("Detailed DataFrame Info:")
    df.info()


def group_sampling(df, trg, m, n, seed, verbose=False):
    grouped = df.groupby(trg)
    group_sizes = grouped.size()

    # Find all (m', n') pairs where at least m' groups have ≥ n' rows
    candidates = []
    for n_candidate in range(1, n+1):  # try from 1 up to n
        eligible_groups = group_sizes[group_sizes >= n_candidate].index.tolist()
        m_candidate = min(m, len(eligible_groups))  # max groups available
        if m_candidate > 0:
            candidates.append((m_candidate, n_candidate, eligible_groups))

    # Pick the (m', n') that maximizes m'*n'
    best_m, best_n, best_groups = max(candidates, key=lambda x: x[0]*x[1])

    # Randomly sample best_m groups
    sampled_groups = pd.Series(best_groups).sample(best_m, random_state=seed).tolist()

    if verbose:
        print(f"Selected {best_m} groups with {best_n} samples each "
              f"→ total {best_m * best_n} samples.")

    # Collect samples
    samples = []
    for group_name in sampled_groups:
        group_sample = grouped.get_group(group_name).sample(best_n, random_state=seed)
        samples.append(group_sample)

    return pd.concat(samples).reset_index(drop=True)


# def group_sampling(df, trg, m, n, seed):
#     # Group the dataframe by the target column
#     grouped = df.groupby(trg)

#     # Get groups that have at least n rows
#     eligible_groups = [name for name, group in grouped if len(group) >= n]

#     # Check if we have enough groups to sample from
#     if len(eligible_groups) < m:
#         m = len(eligible_groups)
#         # raise ValueError(
#         #     f"Not enough groups with at least {n} rows. Only found {len(eligible_groups)} such groups.")

#     # Randomly sample m groups
#     sampled_groups = pd.Series(eligible_groups).sample(m, random_state=seed).tolist()

#     # Initialize an empty list to hold the samples
#     samples = []

#     # For each sampled group, randomly select n rows
#     for group_name in sampled_groups:
#         group_sample = grouped.get_group(group_name).sample(n, random_state=seed)
#         samples.append(group_sample)

#     # Concatenate all the sampled groups to form the final dataframe
#     return pd.concat(samples).reset_index(drop=True)


def dependency_level_to_categorical(dependency_level, dataset_name):
    def get_level(value):
        if value is None:
            return "unknown"
        elif value > 0.66:
            return "high"
        elif value >= 0.33:
            return "med"
        else:
            return "low"

    if dataset_name not in dependency_level:
        return f"Dataset '{dataset_name}' not found."
    
    dataset = dependency_level[dataset_name]
    categorized_items = [(key, get_level(value)) for key, value in dataset.items()]

    # Sort: high first, then med, then low, and finally unknown
    sorted_items = sorted(categorized_items, key=lambda x: ("high med low unknown".split().index(x[1])))

    # Format the result as requested
    return {key: level for key, level in sorted_items}


def drop_long_columns(df, length_limit):
    # Calculate the average length of each column
    avg_lengths = df.apply(lambda col: col.astype(str).str.len().mean()).astype(int)
    
    # Drop columns with average length greater than the threshold
    filtered_df = df.loc[:, avg_lengths <= length_limit]
    
    return filtered_df, avg_lengths.to_dict()

def flexible_match(key_value, col_value):

    def normalize_string(s):
        # Convert to lowercase
        s = s.lower()
        # Normalize Unicode characters (e.g., 'è' -> 'e')
        s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
        # Remove all punctuation
        s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
        # Remove any remaining spaces
        s = s.replace(" ", "")
        return s
    

    def check_subsequence(small, large):
        it = iter(large)
        return all(char in it for char in small)
    

    # Normalize inputs: strip spaces and convert to lowercase
    key_value = normalize_string(key_value)
    col_value = normalize_string(col_value)
    
    if check_subsequence(key_value, col_value) or check_subsequence(col_value, key_value):
        return 1
    return 0


def evaluate(key_response_df, methods):
    res = key_response_df.copy()
    methods = set(methods) 

    for col in res.columns:
        if col not in ['id', 'key']:
            if 'exact_match' in methods:
                # Add a column for exact match
                exact_col_name = f"exact_match:{col}"
                res[exact_col_name] = res.apply(
                    lambda row: 1 if str(row['key']).strip().lower() == str(row[col]).strip().lower() else 0, 
                    axis=1
                )
            # if 'substr_match' in methods:
            #     # Add a column for substring match
            #     substr_col_name = f"substr_match:{col}"
            #     res[substr_col_name] = res.apply(
            #         lambda row: 1 if (
            #             str(row['key']).strip().lower() in str(row[col]).strip().lower() or
            #             str(row[col]).strip().lower() in str(row['key']).strip().lower()
            #         ) else 0, 
            #         axis=1
            #     )
            # if 'flexible_match' in methods:
            #     # Add a column for chunk match
            #     chunk_col_name = f"flexible_match:{col}"
            #     res[chunk_col_name] = res.apply(
            #         lambda row: flexible_match(row['key'], row[col]), 
            #         axis=1
            #     )
            if 'bleu_score' in methods:
                # Add a column for chunk match
                chunk_col_name = f"bleu:{col}"
                res[chunk_col_name] = res.apply(
                    lambda row: compute_bleu(row['key'].strip().lower(), row[col].strip().lower()), 
                    axis=1
                )
            if 'rouge_score' in methods:
                # Add columns for each ROUGE-1 metric
                res[f"rouge_p:{col}"] = res.apply(
                    lambda row: compute_rouge(row['key'].strip().lower(), row[col].strip().lower())["P"],
                    axis=1
                )

                res[f"rouge_r:{col}"] = res.apply(
                    lambda row: compute_rouge(row['key'].strip().lower(), row[col].strip().lower())["R"],
                    axis=1
                )

                res[f"rouge_f1:{col}"] = res.apply(
                    lambda row: compute_rouge(row['key'].strip().lower(), row[col].strip().lower())["F1"],
                    axis=1
                )


    return res


def print_average_column_length(df):
    avg_lengths = {}
    total_sum = 0
    
    for column in df.columns:
        avg_length = df[column].astype(str).apply(len).mean()  # Compute average length without mutating df
        avg_lengths[column] = avg_length
        total_sum += avg_length
    
    print("Average length per column:")
    for col, length in avg_lengths.items():
        print(f"{col}: {length:.2f}")
    
    print(f"Total sum of average lengths: {total_sum:.2f}")

def na_handler(df, na):
    no_nan_df = df.copy()    
    if na == "fill":
        no_nan_df = no_nan_df.fillna("")
    elif na == "drop":
        no_nan_df.replace(["", "NULL", "-"], pd.NA, inplace=True)
        no_nan_df.dropna(how="any", inplace=True)
    return no_nan_df


# def value_handler(dataset_name, df, trg):
#     values = []
#     if dataset_name == "phone_2":
#         values = ["samsung", "nokia", "lg", "apple", "lenovo", "huawei", "amazon", "motorola", "blackberry", "asus", "zte", "acer", "blu", "google", "hp", "microsoft", "sony", "sanyo", "alcatel", "htc", "palm", "Pantech", "Atoah", "Mango Natural", "OnePlus", "OtterBox", "Pandaoo", "Pantech", "Plum", "Posh Mobile"]
#     values = [item.lower() for item in values]
#     return df[df[trg].isin(values)]

# def data_imputation(dataset_name='', path='', models=[], number_of_rows=100, number_of_examples=3, sample_size=None, few_shot_sampling_method='random', shot_number=3, annotation='GPT 3.5'):
def data_imputation(config, run_number):
    
    dataset_name = config.get("dataset", {}).get("name")
    dataset_path = os.path.join(DATA_PATH,DATASETS[dataset_name]['REL_PATH'])
    trg = config.get("dataset", {}).get("target_column")
    na = config.get("na")
    model = config.get("model")

    verbose = config.get("verbose", False)

    df = read_csv(dataset_path)

    if df is not None:

        if verbose:
            print(len(df))
        df = na_handler(df, na)
        if verbose:
            print(len(df))

        if verbose:
            print_dataframe_info(df)
            print_average_column_length(df)

            # Get number of unique values in each column
            unique_counts = df.nunique()
            print(unique_counts)

        # convert all columns to string (needed for downstream dependency analysis)
        df = df.astype(str)
        # to lowercase
        df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x) # to lowercase
        
        # filter rows with specific target values
        # df = value_handler(dataset_name, df,trg)

        # preprocessing
        _, _, df = preprocessing(dataset_name, df)

        df.replace(["", "NULL", "-"], pd.NA, inplace=True)
        df.dropna(how="any", inplace=True)
        if verbose:
            print(len(df))

        train_ratio = config.get("dataset_partition", 80).get("train_ratio")
        number_of_test_rows = config.get("dataset_partition", {}).get("number_of_test_rows")
        dataset_partition_random_seed = config.get("dataset_partition", {}).get("random_seed") + run_number

        train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=dataset_partition_random_seed)
        
        if verbose:
            print("Train Dataframe: ")
            print(train_df.head(10))
            print("# Rows: " + str(len(train_df)))

            print("Test Dataframe: ")
            print(test_df.head(10))
            print("# Rows: " + str(len(test_df)))

        sample_method = config.get("sampling", {}).get("method")
        sample_number = config.get("sampling", {}).get("number_of_samples")
        sample_m = config.get("sampling", {}).get("m")
        sample_n = config.get("sampling", {}).get("n")

        if sample_method is not None:
            if sample_method == "Random Sampling":
                sampled_df = train_df.sample(n=sample_number, random_state=dataset_partition_random_seed)
            elif sample_method == "Group Sampling":
                sampled_df = group_sampling(train_df, trg, sample_m, sample_n, dataset_partition_random_seed, verbose=verbose)

        # _, _, df = preprocessing(dataset_name, df)

        if verbose:
            print("Sampling Method: " + str(sample_method) +
                  " # Samples: " + str(len(sampled_df)))
            print("Sampled Labels: " + str(sampled_df[trg].tolist()))

        # Drop columns with average length greater than the threshold
        length_limit = config.get("column_length_limit")
        # Store original sampled_df before dropping columns for data reduction calculation
        # Also store the full df (used for stats) for accurate reduction calculation
        original_sampled_df = sampled_df.copy()
        sampled_df, avg_len = drop_long_columns(sampled_df, length_limit)
        # print(avg_len)
        # print(sampled_df.columns)

        # Which column(s) are atomic?
        # ner_method = config.get("ner", {}).get("method")
        # ner_number_of_examples = config.get(
        #     "ner", {}).get("number_of_examples")
        # ner_atomicity_threshold = config.get(
        #     "ner", {}).get("atomicity_threshold")
        # atomicity_status = is_atomic(
        #     sampled_df, ner_method, ner_number_of_examples, ner_atomicity_threshold)

        # print("\nNER Method: " + str(ner_method))
        # if ner_method is not None: print("Columns Atomicity Status: " + str(atomicity_status) + "\n")

        # entity detection
        # entity_detection_threshold = config.get(
        #     "ner", {}).get("entity_detection_threshold")
        # sampled_df = entity_extractor(
        #     sampled_df, atomicity_status, ner_number_of_examples, entity_detection_threshold)

        method = config.get("dependency_finder", {}).get("method")
        # column_selection = config.get("dependency_finder", {}).get("column_selection")
        number_of_rules = config.get("dependency_finder", {}).get("number_of_rules", 3)
        p = config.get("dependency_finder", {}).get("inner_threshold")
        q = config.get("dependency_finder", {}).get("outer_threshold")
        evaluate_methods = config.get("evaluate", {}).get("methods")

        samples = sampling(test_df, number_of_test_rows, dataset_partition_random_seed)

        # train:
        src_list = None
        if method == 'LCS':
            src_list, attr_detection__runtime = rule_generator(dataset_name, sampled_df, trg, method, p, q, number_of_rules, verbose=verbose)
            if len(src_list) == 0: return None
            # assert src_list, "No dependencies found!"

            # result
            if verbose:
                print(dependency_level_to_categorical(dependency_level, dataset_name))
            # test:

            # if column_selection == 'single':

            #     rules = [src + " -> " + trg for src in src_list]
            #     key_response_pairs = pd.DataFrame(columns=['id', 'key'] + rules) 
            #     key_response_pairs = fill_keys(df, samples, trg, key_response_pairs)

            #     for src in src_list:
            #         rule = src + " -> " + trg
            #         # apply_examples_df = example_generator(method, column_selection, df, src, trg, example_method, example_rows, number_of_examples, examples_random_seed)
            #         # print("Selected Dataframe: ")
                    # print(apply_examples_df.head(10))
                    # key_response_pairs = run_models(method, model, df, key_response_pairs, rule, samples, apply_examples_df)
            
            #elif column_selection == 'multi':
            rule = str(src_list) + ' -> ' + trg
            # key_response_pairs = pd.DataFrame(columns=['id', 'key', rule])
            # key_response_pairs = fill_keys(df, samples, trg, key_response_pairs)
            # apply_examples_df = example_generator(method, column_selection, df, src_list, trg, example_method, example_rows, number_of_examples, examples_random_seed)
        
        elif method == 'ALL':
            rule = 'ALL' + ' -> ' + trg
            # key_response_pairs = pd.DataFrame(columns=['id', 'key', rule])
            # key_response_pairs = fill_keys(df, samples, trg, key_response_pairs)
            # apply_examples_df = example_generator(method, column_selection, df, None, trg, example_method, example_rows, number_of_examples, examples_random_seed)
            
        # print("Selected Dataframe: ")
        # print(apply_examples_df.head(10))

        key_response_pairs = pd.DataFrame(columns=['id', 'key', rule])
        key_response_pairs = fill_keys(test_df, samples, trg, key_response_pairs)
        key_response_pairs, example_generation_runtime = run_models(config, test_df, train_df, src_list, trg, key_response_pairs, rule, samples, run_number)

        res = evaluate(key_response_pairs, evaluate_methods)

        t: Dict[str, float] = {"Phase-1": 0.0, "Phase-2": 0.0}
        t["Phase-1"]= attr_detection__runtime
        t["Phase-2"]= example_generation_runtime


    return res, t
