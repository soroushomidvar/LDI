import pandas as pd
import numpy as np


def serialize_rows(df, row_numbers=None):
    serialized_data_list = []
    row_numbers = row_numbers

    if row_numbers is None:  # serialize all rows
        row_numbers = range(0, len(df))

    # serialize one row (Check if row_numbers is an integer)
    # isinstance(row_numbers, np.int64):
    if np.issubdtype(type(row_numbers), np.int64):
        row_numbers = np.array([row_numbers])
        # row_numbers = [row_numbers]

    for row_number in row_numbers:  # serialize specific rows
        try:
            selected_row = df.iloc[row_number]
            serialized_data = {attr: value for attr,
                               value in selected_row.items()}
            serialized_data_list.append(serialized_data)

        except IndexError:
            print(f"Row {row_number} not found.")

    r = ''
    for d in serialized_data_list:
        for key, value in d.items():
            r += f'{key}: {value}, '
        r = r[:-2] + '\n'
    return r
