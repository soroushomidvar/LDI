from models.model_execution import *
from tools.serializer import *
from tools.data_handler import *
from tools.entity_recognition import *
from tools.sketch_generator import *
from tools.dependency_finder import *
import json
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
import ast
from gensim.models import Word2Vec
from sklearn.impute import SimpleImputer


def preprocessing(dataset_name, df):

    if dataset_name == BUY_DATASET_CONSTANTS.VALUE['NAME']:
        df = df[BUY_DATASET_CONSTANTS.VALUE['ALL_COLUMNS']]
        key = df[BUY_DATASET_CONSTANTS.VALUE['KEY']]
        rest = df[BUY_DATASET_CONSTANTS.VALUE['REST']]

    elif dataset_name == RESTAURANT_DATASET_CONSTANTS.VALUE['NAME']:
        df = df[RESTAURANT_DATASET_CONSTANTS.VALUE['ALL_COLUMNS']]
        key = df[RESTAURANT_DATASET_CONSTANTS.VALUE['KEY']]
        rest = df[RESTAURANT_DATASET_CONSTANTS.VALUE['REST']]

    elif dataset_name == FLIPKART_DATASET_CONSTANTS.VALUE['NAME']:
        df = df[FLIPKART_DATASET_CONSTANTS.VALUE['ALL_COLUMNS']]
        key = df[FLIPKART_DATASET_CONSTANTS.VALUE['KEY']]
        rest = df[FLIPKART_DATASET_CONSTANTS.VALUE['REST']]

    return key, rest, df


def rule_generator_old(dataset_name, rest, df, shot_number, method='random', target_row=-1, annotation='GPT 3.5'):
    prompt = ''
    fixed_initial = ''
    serialized_samples = ''
    serialized_target_row = ''
    fixed_querry = ''

    def dataset_specific_prompt_generator():

        if dataset_name == BUY_DATASET_CONSTANTS.VALUE['NAME']:
            fixed_initial = BUY_DATASET_PROMPTS.VALUE['FIXED_INITIAL']
            fixed_querry = BUY_DATASET_PROMPTS.VALUE['FIXED_QUERRY']

        elif dataset_name == RESTAURANT_DATASET_CONSTANTS.VALUE['NAME']:
            fixed_initial = RESTAURANT_DATASET_PROMPTS.VALUE['FIXED_INITIAL']
            fixed_querry = RESTAURANT_DATASET_PROMPTS.VALUE['FIXED_QUERRY']

        return fixed_initial, fixed_querry

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

    fixed_initial, fixed_querry = dataset_specific_prompt_generator()
    serialized_target_row, serialized_samples = serializer()
    named_entities = named_entity_recognizer(serialized_target_row, annotation)
    prompt = shot_handler()

    return prompt, serialized_target_row, named_entities


def sampling(df, sample_size):
    if sample_size is None:
        samples = np.arange(len(df))  # all data
    else:
        samples = np.random.choice(
            np.arange(len(df)-1), size=sample_size, replace=False)
    return samples


def example_generator(df, src, trg, examples_number):
    if examples_number > df[trg].nunique():
        raise ValueError(
            "The number of unique examples requested exceeds the number of unique values in the trg column.")

    # Initialize an empty DataFrame to store the unique examples
    unique_examples = pd.DataFrame(columns=[src, trg])

    # Create a list of already selected trg values
    selected_trg_values = set()

    while len(unique_examples) < examples_number:
        # Randomly select a row index
        random_index = random.choice(df.index)

        # Get the trg value of the randomly selected row
        trg_value = df.loc[random_index, trg]

        # If the trg value has not been selected before, add the row to unique_examples
        if trg_value not in selected_trg_values:
            selected_trg_values.add(trg_value)
            selected_row = df.loc[[random_index], [src, trg]]
            unique_examples = pd.concat([unique_examples, selected_row])

    return unique_examples.reset_index(drop=True)


def train_word2vec_model(text_data):
    """Train a Word2Vec model on the given text data."""
    tokenized_text = [text.split()
                      for text in text_data]  # Simple tokenization
    model = Word2Vec(sentences=tokenized_text, vector_size=50,
                     window=5, min_count=1, workers=4)
    return model


def text_to_vector(text, model):
    """Convert text to a fixed-size vector using the provided Word2Vec model."""
    tokens = text.split()  # Simple tokenization
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if len(vectors) == 0:  # Handle case where no tokens are in the model's vocabulary
        return np.zeros(model.vector_size)
    # Average the vectors
    vector = np.mean(vectors, axis=0)
    return vector


def dependency_finder(method, df_main, target_col, p, q):
    df = df_main.copy()
    # Check if the target column is categorical
    if df[target_col].dtype != 'object':
        raise ValueError("The target column must be categorical.")

    # Encode categorical columns, including the target column
    rels = {}
    for column in df.columns:
        if column != target_col:  # df[column].dtype == 'object' and
            deg, rel = is_dependant(df[[column, target_col]], p, q)
            # if (rel):
            rels[column] = deg
            print(str(column) + ": " + str(rel))
            # le = LabelEncoder()
            # df[column] = le.fit_transform(df[column])
            # label_encoders[column] = le

    # # Separate features and target
    # X = df.drop(columns=[target_col])
    # y = df[target_col]

    # # Initialize and train the RandomForestClassifier
    # model = DecisionTreeClassifier()  # RandomForestClassifier()
    # model.fit(X, y)

    # # plt.figure(figsize=(20, 10))
    # # plot_tree(model, filled=True, max_depth=5,
    # #           rounded=True, feature_names=df.columns)
    # # plt.show()

    # # Get feature importances
    # importances = pd.Series(model.feature_importances_,
    #                         index=X.columns).sort_values(ascending=False)

    # # Convert the importances to a dictionary
    # feature_importance_dict = importances.to_dict()

    return rels  # feature_importance_dict


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
    top_n_keys = [item[0] for item in sorted_features[:n]]
    return top_n_keys


def rules_to_str(lst, key):
    result = []
    for i, item in enumerate(lst):
        if i == len(lst) - 1:
            result.append(f"{item} -> {key}")
        else:
            result.append(f"{item} -> {key}, ")
    return ''.join(result)


def rule_generator(dataset_name, df, trg, method, p, q,  number_of_examples=3, number_of_rules=3):
    rule = None
    # key, rest, df = preprocessing(dataset_name, df)

    # key_sketch = get_sketch(key, number_of_examples)
    # rest_sketch = get_sketch(rest, number_of_examples)
    # df_sketch = get_sketch(df, number_of_examples)

    # print("\nDataset Sketch:")
    # print(df_sketch)

    dependency_values = dependency_finder(method, df, trg, p, q)
    # combination_importance_DT = dependency_finder_combinations_decision_tree(
    #     df, key.columns[0])
    # print("Combination Importance (DT):", combination_importance_DT)

    # combination_importance_RF = dependency_finder_combinations_random_forest(
    #     df, key.columns[0])
    # print("Combination Importance (RF):", combination_importance_RF)

    print("\nColumn Dependencies (Importance):")
    print(dependency_values)

    left_hand_columns = get_top_n_features(dependency_values, number_of_rules)

    # dependant_column_details, dependant_column = assign_probabilities_to_sketch_minimal(
    #     df.columns.tolist(), dependency_values)

    # dependant_columns_details = assign_probabilities_to_sketch(
    #     df_sketch, dependency_values)

    left_hand_columns_str = rules_to_str(left_hand_columns, trg)
    # str(left_hand_columns) + " -> " + key.columns[0]

    print("\nRule(s):")
    print(left_hand_columns_str)

    # dependant_column_details_str = str(
    #     dependant_column_details)+" -> "+key.columns[0]
    # print("\nRule with Details:")
    # print(dependant_column_details_str)

    # dependant_columns_details_str = str(
    #     dependant_columns_details)+" -> "+key.columns[0]
    # print("\nRule with more Details")
    # print(dependant_columns_details_str)

    # for example in examples:

    #     serialized_example_rest = serialize_rows(rest, example)
    #     serialized_example_key = serialize_rows(key, example)

    # entities_rest = named_entity_recognizer(
    #     serialized_example_rest, "GPT 3.5")
    # entities_keys = named_entity_recognizer(
    #     serialized_example_key, "GPT 3.5")

    # print(entities_rest)
    # print(entities_keys)

    # return dependant_column, dependant_column_details, dependant_columns_details, dependant_column_str, dependant_column_details_str, dependant_columns_details_str
    return left_hand_columns, None, None, left_hand_columns_str, None, None


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


# def is_atomic(df, sample_size):
#     if sample_size is None:
#         sampled_df = df
#     else:
#         sampled_df = df.sample(n=sample_size)

#     atomic_status = []

#     for column in sampled_df.columns:
#         column_atomic = True

#         for value in sampled_df[column]:
#             entity_types = named_entity_recognizer(str(value), "GPT 3.5")
#             entity_types_list = ast.literal_eval(str(entity_types))
#             if len(entity_types_list) > 1:
#                 column_atomic = False
#                 break

#         atomic_status.append(column_atomic)

#     return atomic_status


# def dataframe_entity_type_detection(df, sample_size, threshold):
#     atomic_status = is_atomic(df, sample_size)

#     if sample_size is None:
#         df = df
#     else:
#         df = df.sample(n=sample_size)

#     entity_results = {column: [] for column in df.columns}

#     for column in df.columns:
#         # if not atomic_status[df.columns.get_loc(column)]:
#         for value in df[column]:
#             entity_types_str = named_entity_recognizer(
#                 str(value), "GPT 3.5")
#             print(entity_types_str)
#             try:
#                 entity_types = ast.literal_eval(entity_types_str)
#             except:
#                 entity_types = None

#             entity_results[column].append(entity_types)

#     new_columns = {}

#     for col_idx, column in enumerate(df.columns):
#         col_data = df[column]
#         entity_counter = Counter()

#         if atomic_status[col_idx]:
#             for entity_types in entity_results[column]:
#                 entity_counter.update(entity_types)

#             if entity_counter:
#                 majority_entity = entity_counter.most_common(1)[0][0]
#                 new_col_name = f"{column}[{majority_entity}]"
#                 new_columns[new_col_name] = col_data
#         else:
#             if entity_results[column]:
#                 for entities_list in entity_results[column]:
#                     for entity in entities_list:
#                         entity_counter[entity] += 1

#                 for entity, count in entity_counter.items():
#                     if count / len(entity_results[column]) > threshold:
#                         new_col_name = f"{column}[{entity}]"
#                         new_col_data = [value if entity in entities else None for value, entities in zip(
#                             col_data, entity_results[column])]
#                         new_columns[new_col_name] = new_col_data

#     return pd.DataFrame(new_columns)


def mpping_handler(rule, model, examples, sample):

    initial_prompt = MAPPING_HANDLER_PROMPTS.VALUE['MAPPING_HANDLER_INITIAL']
    initial_prompt = initial_prompt.replace(
        '<src>', examples.columns[0]).replace('<trg>', examples.columns[1]).replace('<rule>', rule)
    serialized_examples = serialize_rows(examples)

    src_name = examples.columns[0]
    sample_src = sample[[src_name]]
    serialized_target_row = serialize_rows(sample_src)

    fixed_prompt = MAPPING_HANDLER_PROMPTS.VALUE['MAPPING_HANDLER_FIXED_QUERRY']
    fixed_prompt = fixed_prompt.replace('<trg>', examples.columns[1])
    prompt = initial_prompt + serialized_examples + ' ' + \
        serialized_target_row + fixed_prompt
    response = prompt_runner(model, prompt)
    print('prompt: ' + prompt)
    # print('response : '+str(response))
    print(f"response ({model}): {response}")
    return response


def run_models(model, df, key_response_pairs, rule, samples, examples):

    # def detect_value_from_response(r):
    #     if r is not None:
    #         parts = r.split('->')
    #         if len(parts) >= 2:
    #             return parts[1]
    #     return None

    # def label_corrector(input_str, string_list, method='jaccard'):

    #     def jaccard_similarity(str1, str2):
    #         set1 = set(str1)
    #         set2 = set(str2)
    #         intersection = len(set1.intersection(set2))
    #         union = len(set1.union(set2))
    #         return intersection / union if union != 0 else 0

    #     if method == 'cosine':
    #         # Vectorize the input string and the string list
    #         vectorizer = CountVectorizer().fit_transform(
    #             [input_str] + string_list)
    #         vectors = vectorizer.toarray()

    #         # Calculate cosine similarity between input string and each string in the list
    #         similarities = cosine_similarity(vectors)[0][1:]

    #         # Get the index of the most similar string
    #         most_similar_index = similarities.argmax()

    #         # Return the most similar string
    #         return string_list[most_similar_index]
    #     elif method == 'jaccard':
    #         # Calculate Jaccard similarity between input string and each string in the list
    #         similarities = [(jaccard_similarity(input_str, s), s)
    #                         for s in string_list]

    #         # Sort by Jaccard similarity in descending order
    #         similarities.sort(reverse=True)

    #         # Return the most similar string
    #         return similarities[0][1]

    # def corerct_label_in_response(r, labels):
    #     if r is not None:
    #         parts = r.split('->')
    #         if len(parts) >= 2:
    #             label = label_corrector(parts[1], labels)
    #             return parts[0]+'->'+label
    #     return None

    for sample in samples:
        # for model in models:
        # prompt, serialized_target_row, named_entities = rule_generator(
        #     dataset_name, rest, df, shot_number, few_shot_sampling_method, sample, annotation)
        # print(prompt)
        value = mpping_handler(
            rule, model, examples, get_sample_by_row_number(df, sample))
        # rule = prompt_runner(model, prompt)
        # print(value)
        # response = corerct_label_in_response(response, labels)
        # key_response_pairs.loc[key_response_pairs['id']
        #                        == sample, model_output_names] = rule
        key_response_pairs.loc[key_response_pairs['id']
                               == sample, rule] = value  # detect_value_from_response(rule)

    return key_response_pairs


def group_sampling(df, trg, m, n):
    # Group the dataframe by the target column
    grouped = df.groupby(trg)

    # Get groups that have at least n rows
    eligible_groups = [name for name, group in grouped if len(group) >= n]

    # Check if we have enough groups to sample from
    if len(eligible_groups) < m:
        raise ValueError(
            f"Not enough groups with at least {n} rows. Only found {len(eligible_groups)} such groups.")

    # Randomly sample m groups
    sampled_groups = pd.Series(eligible_groups).sample(m).tolist()

    # Initialize an empty list to hold the samples
    samples = []

    # For each sampled group, randomly select n rows
    for group_name in sampled_groups:
        group_sample = grouped.get_group(group_name).sample(n)
        samples.append(group_sample)

    # Concatenate all the sampled groups to form the final dataframe
    return pd.concat(samples).reset_index(drop=True)


# def data_imputation(dataset_name='', path='', models=[], number_of_rows=100, number_of_examples=3, sample_size=None, few_shot_sampling_method='random', shot_number=3, annotation='GPT 3.5'):
def data_imputation(config):
    dataset = config.get("dataset", {}).get("name")
    if dataset.lower() == "restaurant":
        dataset_name = RESTAURANT_DATASET_CONSTANTS.VALUE['NAME']
        dataset_path = os.path.join(
            DATA_PATH, RESTAURANT_DATASET_CONSTANTS.VALUE['REL_PATH'])
    elif dataset.lower() == "buy":
        dataset_name = BUY_DATASET_CONSTANTS.VALUE['NAME']
        dataset_path = os.path.join(
            DATA_PATH, BUY_DATASET_CONSTANTS.VALUE['REL_PATH'])
    elif dataset.lower() == "flipkart":
        dataset_name = FLIPKART_DATASET_CONSTANTS.VALUE['NAME']
        dataset_path = os.path.join(
            DATA_PATH, FLIPKART_DATASET_CONSTANTS.VALUE['REL_PATH'])

    model = config.get("model")
    # model_output_names = [item + " rule" for item in models]

    df = read_csv(dataset_path)

    if df is not None:

        # drop NaNs
        df = df.dropna()

        trg = config.get("dataset", {}).get("target_column")

        sample_method = config.get("sampling", {}).get("method")
        sample_number = config.get("sampling", {}).get("number_of_samples")
        sample_m = config.get("sampling", {}).get("m")
        sample_n = config.get("sampling", {}).get("n")

        if sample_method is not None:
            if sample_method == "Random Sampling":
                sampled_df = df.sample(n=sample_number)
            elif sample_method == "Group Sampling":
                sampled_df = group_sampling(df, trg, sample_m, sample_n)

        _, _, df = preprocessing(dataset_name, df)

        print("Sampling Method: " + str(sample_method) +
              " # Samples: " + str(len(sampled_df)) + "\n")
        print("Sampled Labels: " + str(sampled_df[trg].tolist()))

        # Which column(s) are atomic?
        ner_method = config.get("ner", {}).get("method")
        ner_number_of_examples = config.get(
            "ner", {}).get("number_of_examples")
        ner_atomicity_threshold = config.get(
            "ner", {}).get("atomicity_threshold")
        atomicity_status = is_atomic(
            sampled_df, ner_method, ner_number_of_examples, ner_atomicity_threshold)

        print("NER Method: " + str(ner_method))
        print("Columns Atomicity Status: " + str(atomicity_status) + "\n")

        # entity detection
        entity_detection_threshold = config.get(
            "ner", {}).get("entity_detection_threshold")
        sampled_df = entity_extractor(
            sampled_df, atomicity_status, ner_number_of_examples, entity_detection_threshold)

        number_of_rules = config.get("number_of_rules", 3)
        method = config.get("dependency_finder", {}).get("method")
        p = config.get("dependency_finder", {}).get("inner_threshold")
        q = config.get("dependency_finder", {}).get("outer_threshold")

        # train:
        src_list, _, _, rules_str, _, _ = rule_generator(
            dataset_name, sampled_df, trg, method, p, q, ner_number_of_examples, number_of_rules)

        # test:

        apply_examples = config.get("apply", {}).get("number_of_examples")
        apply_rows = config.get("apply", {}).get("number_of_rows")
        # samples are used for the test phase
        samples = sampling(sampled_df, apply_rows)

        # print(samples)

        rules = [src + " -> " + trg for src in src_list]

        key_response_pairs = pd.DataFrame(
            columns=['id', 'key'] + rules)  # + model_output_names)

        # key_response_pairs = fill_keys(samples, df[trg], key_response_pairs)
        key_response_pairs = fill_keys(df, samples, trg, key_response_pairs)

        for src in src_list:

            rule = src + " -> " + trg

            # examples are used for the applying rule
            apply_examples_df = example_generator(
                df, src, trg, apply_examples)

            print("Selected Dataframe: ")
            print(apply_examples_df.head(5))

            key_response_pairs = run_models(
                model, df, key_response_pairs, rule, samples, apply_examples_df)

    return key_response_pairs
