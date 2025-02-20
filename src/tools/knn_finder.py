import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Function to compute Longest Common Subsequence (LCS)
def lcs(s1, s2):
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]

# Function to calculate LCS similarity between two strings
def lcs_similarity(s1, s2):
    max_len = max(len(s1), len(s2))
    return 0 if max_len == 0 else lcs(s1, s2) / max_len
    #return lcs(s1, s2) / max(len(s1), len(s2))

# Main function to find top k similar rows
# def find_top_k_similar_rows(test_df, train_df, test_row_idx, k, columns_without_trg, trg, diversity):
#     test_row = test_df.iloc[test_row_idx][columns_without_trg]

#     # Compute LCS similarity between the test row and each row in the train dataframe
#     similarities = []
#     for idx, train_row in train_df.iterrows():
#         row_similarity = np.mean([
#             lcs_similarity(str(test_val), str(train_val))
#             for test_val, train_val in zip(test_row, train_row[columns_without_trg])
#         ])
#         similarities.append(row_similarity)

#     # Sort similarities and get top k indices and values
#     sorted_indices = np.argsort(similarities)[::-1][:k].tolist()
#     top_k_similarities = [similarities[i] for i in sorted_indices]

#     return sorted_indices, top_k_similarities

def find_top_k_similar_rows(test_df, train_df, test_row_idx, k, columns_without_trg, trg, diversity):
    test_row = test_df.iloc[test_row_idx][columns_without_trg]
    test_trg_value = test_df.iloc[test_row_idx][trg]

    # Compute LCS similarity between the test row and each row in the train dataframe
    similarities = []
    for idx, train_row in train_df.iterrows():
        row_similarity = np.mean([
            lcs_similarity(str(test_val), str(train_val))
            for test_val, train_val in zip(test_row, train_row[columns_without_trg])
        ])
        similarities.append((row_similarity, idx, train_row[trg]))

    # Sort by similarity in descending order
    similarities.sort(reverse=True, key=lambda x: x[0])

    # Select top k with diversity if required
    selected_indices = []
    selected_similarities = []
    seen_trg_values = set()

    for sim, idx, trg_value in similarities:
        if diversity and trg_value in seen_trg_values:
            continue  # Skip if diversity is required and trg_value is already selected
        
        selected_indices.append(idx)
        selected_similarities.append(sim)
        seen_trg_values.add(trg_value)

        if len(selected_indices) == k:
            break

    return selected_indices, selected_similarities
