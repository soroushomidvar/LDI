import pandas as pd
from collections import Counter
from typing import List, Dict, Union
from tools.entity_recognition import *
import random
import ast

# Assume this is a placeholder for the actual named_entity_recognizer function


def get_sketch(df: pd.DataFrame, num_samples: int) -> dict:
    sketch = {}
    model_name = "GPT 3.5"

    # Sample the dataframe
    sampled_df = df.sample(n=num_samples) if num_samples < len(df) else df

    for column in df.columns:
        # Serialize the column values
        # serialized_values = f"{column}: " + \
        #     ", ".join(sampled_df[column].astype(str).tolist())
        serialized_values = ', '.join(
            [f"{column}: {str(value)}" for value in df[column][:num_samples]])

        # Get entity types
        entity_types = named_entity_recognizer(serialized_values, model_name)

        # Determine if the column is atomic or non-atomic
        entity_types_list = ast.literal_eval(entity_types)
        sketch[column] = entity_types_list

    return sketch
