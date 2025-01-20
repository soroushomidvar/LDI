# import json
# import pandas as pd
# import random
# import os

# def example_generator(dataset_name, k, repeat, example_path, df, trg):

#     if os.path.exists(example_path):
#         with open(example_path, 'r') as file:
#             data = example_path.load(file)
#         if data.get("dataset") == dataset_name and data.get("k") == k and data.get("repeat") == repeat:
#             print("The example file exists.")
#             return data.get("runs")

#     print("The example file does not exist/match.")

#     if k > df[trg].nunique():
#         raise ValueError(
#             "The number of unique examples requested exceeds the number of unique values in the trg column.")
    
#     runs=[]
#     while len(runs) < repeat:
#         examples = []
#         # Create a list of already selected trg values
#         selected_trg_values = set()

#         while len(examples) < k:
#             # Randomly select a row index
#             random_index = random.choice(df.index)
#             # Get the trg value of the randomly selected row
#             trg_value = df.loc[random_index, trg]
#             if trg_value not in selected_trg_values:
#                 selected_trg_values.add(trg_value)

#             examples.append(random_index)
        
#         runs.append(examples)

#     examples_data = {
#         "dataset": dataset_name,
#         "k": k,
#         "repeat": repeat,
#         "runs": list(runs)
#     }

#     with open(example_path, 'w') as file:
#         json.dump(examples_data, file, indent=4)

#     return runs