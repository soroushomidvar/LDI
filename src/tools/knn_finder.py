import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import time
from difflib import SequenceMatcher

# def fast_similarity(s1, s2):
#     """Fast similarity using SequenceMatcher (ratio between 0 and 1)."""
#     return SequenceMatcher(None, s1, s2).ratio()

# Function to compute Longest Common Subsequence (LCS)
# def lcs(s1, s2):
#     n, m = len(s1), len(s2)
#     dp = [[0] * (m + 1) for _ in range(n + 1)]

#     for i in range(1, n + 1):
#         for j in range(1, m + 1):
#             if s1[i - 1] == s2[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1] + 1
#             else:
#                 dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

#     return dp[n][m]

# Function to calculate LCS similarity between two strings
# def lcs_similarity(s1, s2):
#     max_len = max(len(s1), len(s2))
#     return 0 if max_len == 0 else lcs(s1, s2) / max_len
#     #return lcs(s1, s2) / max(len(s1), len(s2))

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

# def longest_common_substring_len(a: str, b: str) -> int:
#     sm = SequenceMatcher(None, a, b)
#     # get_matching_blocks returns a list of Match(a,i,b,j,size)
#     return max(block.size for block in sm.get_matching_blocks())
class SuffixAutomaton:
    def __init__(self, s: str):
        self.states = [{}]       # transitions
        self.link = [-1]         # suffix links
        self.length = [0]        # longest length for each state
        self.last = 0            # index of the state for whole string so far

        for c in s:
            self._extend(c)

    def _extend(self, c: str):
        cur = len(self.states)
        self.states.append({})
        self.length.append(self.length[self.last] + 1)
        self.link.append(0)

        p = self.last
        while p >= 0 and c not in self.states[p]:
            self.states[p][c] = cur
            p = self.link[p]

        if p == -1:
            self.link[cur] = 0
        else:
            q = self.states[p][c]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = len(self.states)
                self.states.append(self.states[q].copy())
                self.length.append(self.length[p] + 1)
                self.link.append(self.link[q])
                while p >= 0 and self.states[p].get(c) == q:
                    self.states[p][c] = clone
                    p = self.link[p]
                self.link[q] = self.link[cur] = clone
        self.last = cur


def longest_common_substring(a: str, b: str):
    sa = SuffixAutomaton(a)
    v, l, best = 0, 0, 0
    substrings = set()
    end_pos = -1

    for i, c in enumerate(b):
        if c in sa.states[v]:
            v = sa.states[v][c]
            l += 1
        else:
            while v != -1 and c not in sa.states[v]:
                v = sa.link[v]
            if v == -1:
                v, l = 0, 0
                continue
            l = sa.length[v] + 1
            v = sa.states[v][c]

        if l > best:
            best = l
            end_pos = i
            substrings = {b[end_pos - best + 1:end_pos + 1]}
        elif l == best and best > 0:
            substrings.add(b[i - best + 1:i + 1])

    return best, substrings


def substring_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    L, _ = longest_common_substring(a, b)
    return L / max(len(a), len(b))

def find_top_k_similar_rows(test_df, train_df, test_row_idx, k, columns_without_trg, trg, diversity, verbose=False):
    test_row = test_df.iloc[test_row_idx][columns_without_trg]
    test_trg_value = test_df.iloc[test_row_idx][trg]

    start_time = time.time()
    # Compute LCS similarity between the test row and each row in the train dataframe
    similarities = []
    for idx, train_row in train_df.iterrows():
        row_similarity = np.mean([
            # lcs_similarity(str(test_val), str(train_val))
            substring_similarity(str(test_val), str(train_val))
            for test_val, train_val in zip(test_row, train_row[columns_without_trg])
        ])
        similarities.append((row_similarity, idx, train_row[trg]))

    end_time_1 = time.time()
    if verbose:
        print("Time taken for LCS: {:.6f} seconds".format(end_time_1 - start_time))

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
    end_time_2 = time.time()
    if verbose:
        print("Time taken for selecting top k: {:.6f} seconds".format(end_time_2 - start_time))

    return selected_indices, selected_similarities
