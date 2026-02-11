import pandas as pd
from collections import defaultdict
from collections import Counter
import math


def common_patterns_tree_based(strings, p):
    
    def build_suffix_array(s):
        n = len(s)
        k = 1
        rank = [ord(c) for c in s]
        tmp = [0] * n
        sa = list(range(n))

        while True:
            sa.sort(key=lambda x: (rank[x], rank[x + k] if x + k < n else -1))
            tmp[sa[0]] = 0
            for i in range(1, n):
                tmp[sa[i]] = tmp[sa[i - 1]]
                if rank[sa[i]] != rank[sa[i - 1]] or \
                (rank[sa[i] + k] if sa[i] + k < n else -1) != (rank[sa[i - 1] + k] if sa[i - 1] + k < n else -1):
                    tmp[sa[i]] += 1
            rank = tmp[:]
            if rank[sa[-1]] == n - 1:
                break
            k *= 2
        return sa
    
    def build_lcp(s, sa):
        n = len(s)
        rank = [0] * n
        for i in range(n):
            rank[sa[i]] = i
        lcp = [0] * (n - 1)
        h = 0
        for i in range(n):
            if rank[i] > 0:
                j = sa[rank[i] - 1]
                while i + h < n and j + h < n and s[i + h] == s[j + h]:
                    h += 1
                lcp[rank[i] - 1] = h
                if h > 0:
                    h -= 1
        return lcp

    def all_substrings(s):
        """Return all contiguous substrings of s"""
        result = set()
        n = len(s)
        for length in range(1, n + 1):
            for start in range(n - length + 1):
                result.add(s[start:start+length])
        return result


    k = len(strings)
    min_required = math.ceil(p * k)
    if min_required == 0:
        return set()

    delimiter_base = 123456
    concat = ""
    owners = []
    for idx, s in enumerate(strings):
        concat += s + chr(delimiter_base + idx)
        owners.extend([idx] * (len(s) + 1))
    
    sa = build_suffix_array(concat)
    lcp = build_lcp(concat, sa)

    maximal_substrings = set()
    n = len(sa)
    window_count = Counter()
    left = 0

    for right in range(n):
        window_count[owners[sa[right]]] += 1
        while len(window_count) >= min_required:
            if right > left:
                min_lcp = min(lcp[left:right])
                if min_lcp > 0:
                    maximal_substrings.add(concat[sa[right]: sa[right] + min_lcp])
            window_count[owners[sa[left]]] -= 1
            if window_count[owners[sa[left]]] == 0:
                del window_count[owners[sa[left]]]
            left += 1

    # Generate all subparts of each maximal substring
    all_common = set()
    for s in maximal_substrings:
        all_common.update(all_substrings(s))

    sorted_common = sorted(all_common, key=lambda x: (-len(x), x))

    return list(sorted_common)



# def common_patterns(strings, p):
#     def generate_substrings(s):
#         s = str(s)
#         # Generate all substrings of a given string.
#         substrings = set()
#         length = len(s)
#         for i in range(length):
#             for j in range(i + 1, length + 1):
#                 substrings.add(s[i:j])
#         return substrings

#     def count_substring_occurrences(strings):
#         # Count the occurrences of each substring in the list of strings.
#         substring_count = defaultdict(int)
#         for s in strings:
#             found_substrings = generate_substrings(s)
#             for substring in found_substrings:
#                 substring_count[substring] += 1
#         return substring_count

#     def filter_substrings(substring_count, threshold, total_strings):
#         # Filter substrings based on the percentage threshold.
#         return [substring for substring, count in substring_count.items()
#                 if count / total_strings >= threshold]

#     threshold = p
#     total_strings = len(strings)

#     # Generate the substring counts
#     substring_count = count_substring_occurrences(strings)

#     # Filter substrings that appear in at least p*100 percent of strings
#     valid_substrings = filter_substrings(
#         substring_count, threshold, total_strings)

#     # Remove substrings that are contained within longer substrings
#     # filtered_substrings = remove_substrings_within_longer(valid_substrings)

#     # Sort substrings by length (longest first) and then alphabetically
#     sorted_substrings = sorted(valid_substrings, key=lambda x: (-len(x), x))

#     # Return the top n substrings
#     return sorted_substrings

def remove_substrings_within_longer(data):
    # Group substrings by their value
    grouped_data = {}
    for substring, label in data.items():
        if label not in grouped_data:
            grouped_data[label] = []
        grouped_data[label].append(substring)

    # Function to remove substrings contained within longer ones for each group
    def remove_substrings_within_longer_per_value(substrings):
        sorted_substrings = sorted(substrings, key=lambda x: -len(x))
        filtered_substrings = []
        seen = set()

        for substring in sorted_substrings:
            if not any(substring in seen_substring for seen_substring in seen):
                filtered_substrings.append(substring)
                seen.add(substring)

        return filtered_substrings

    # Process each group and build the resulting dictionary
    result = {}
    for label, substrings in grouped_data.items():
        filtered_substrings = remove_substrings_within_longer_per_value(substrings)
        for substring in filtered_substrings:
            result[substring] = label

    return result


class TrieNode:
    def __init__(self):
        self.children = {}
        self.groups = set()  # groups that contain this substring
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, group):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.groups.add(group)  # mark group presence along the path
        node.is_end = True

    def contains_in_other_group(self, word, group):
        """Check if `word` appears inside a longer substring
        that belongs to a different group."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]

        # If this node has descendants that belong to other groups → not unique
        stack = [node]
        while stack:
            cur = stack.pop()
            if (cur.is_end or cur.children) and not (cur.groups <= {group}):
                return True
            for child in cur.children.values():
                stack.append(child)
        return False

def grouped_key_patterns(df, p):
    grouped = df.groupby(df.columns[1])
    group_substrings = {}
    all_substrings = defaultdict(set)

    # Collect substrings
    for group, subset in grouped:
        strings = subset[df.columns[0]].values
        top_substrings = common_patterns_tree_based(strings, p)

        group_substrings[group] = top_substrings
        for substring in top_substrings:
            all_substrings[substring].add(group)

    # Build a trie of all substrings
    trie = Trie()
    for substring, groups in all_substrings.items():
        for g in groups:
            trie.insert(substring, g)

    result = {}
    for group, substrings in group_substrings.items():
        for substring in substrings:
            # check uniqueness with trie
            if not trie.contains_in_other_group(substring, group):
                result[substring] = group

    # Remove substrings contained in longer ones
    long_result = remove_substrings_within_longer(result)

    # Sort the result
    sorted_result = dict(
        sorted(long_result.items(), key=lambda x: (-len(x[0]), x[0]))
    )
    return sorted_result


# def grouped_key_patterns(df, p):
#     #df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x) # to lowercase
#     grouped = df.groupby(df.columns[1])
#     all_substrings = defaultdict(set)
#     group_substrings = {}

#     # Collect substrings for each group
#     for group, subset in grouped:
#         strings = subset[df.columns[0]].tolist()
#         top_substrings = common_patterns_tree_based(strings, p)

#         group_substrings[group] = top_substrings
#         for substring in top_substrings:
#             all_substrings[substring].add(group)

#     result = {}

#     # # Filter substrings to include only those unique to one group
#     # for group, substrings in group_substrings.items():
#     #     unique_substrings = [substring for substring in substrings if len(
#     #         all_substrings[substring]) == 1]

#     #     for substring in unique_substrings:
#     #         result[substring] = group

#     # Filter substrings to include only those unique to one group
#     for group, substrings in group_substrings.items():
#         unique_substrings = []

#         # Check if a substring is a part of a longer substring in other groups
#         for substring in substrings:
#             is_unique = True
#             for other_substring in all_substrings:
#                 #TODO
#                 if substring != other_substring and substring in other_substring:
#                     # If the substring is part of a longer substring in another group, it's not unique
#                     if group not in all_substrings[other_substring]:
#                         is_unique = False
#                         break

#             # Only keep substring if it's unique
#             if is_unique: # and len(all_substrings[substring]) == 1: #TODO
#                 unique_substrings.append(substring)

#         # Add unique substrings to result
#         for substring in unique_substrings:
#             result[substring] = group

#     # Remove substrings that are contained within longer substrings
#     long_result = remove_substrings_within_longer(result)

#     # Sort the result by substring length and alphabetically
#     sorted_result = dict(
#         sorted(long_result.items(), key=lambda x: (-len(x[0]), x[0])))

#     # Return the top n results
#     return dict(list(sorted_result.items()))


def is_dependant(df, p, q):
    result = grouped_key_patterns(df, p)
    # TODO
    groups_with_unique_substrings = set(result.values())
    total_groups = df[df.columns[1]].nunique()
    required_groups = q * total_groups
    status = len(groups_with_unique_substrings) >= required_groups
    # degree = len(groups_with_unique_substrings)/total_groups
    # frequency = -1 # word_frequency(list(result.keys())) if status else -1
    return status, result


# # Example usage
# data = {
#     'text': ['http://www.flipkart.com/ alisha-solid-men-s-cycling-shorts/p/itmeh2ffvzetthbb?pid=SRTEH2FF9KEDEFGF',
#              'http://www.flipkart.com/aalisha-solid-men-s-cycling-shorts/p/itmeh2f6sdgah2pq?pid=SRTEH2F6HUZMQ6SJ',
#              'http://www.flipkart.com/galisha-solid-women-s-cycling-shorts/p/itmeh2fgcjz4mzvu?pid=SRTEH2FGBDJGX8FW',
#              'http://www.flipkart.com/ealisha-solid-women-s-cycling-shorts/p/itmeh2fehghynve9?pid=SRTEH2FECMGNZJXJ',
#              'http://www.flipkart.com/ efurst-usb-adapter-cable-mt-x-2nd-gen-battery-charger/p/itmekfvwggckx25z?pid=ACCEKFVWCGHSY6NF',
#              'http://www.flipkart.com/rfurst-usb-adapter-cable-lnvo-vibe-p1m-battery-charger/p/itmekfvwhatvgqdn?pid=ACCEKFVWTGZXG5XK',
#              'http://www.flipkart.com/3 furst-sync-data-charging-mzu-m3-note-usb-cable/p/itmejud8ynwgqwrp?pid=ACCEJUD8SHZPSDWQ',
#              'http://www.flipkart.com/furst-sync-data-charging-blade-d2-usb-cable/p/itmejud9hcqqx8hn?pid=ACCEJUD92GWTPCN2',],
#     'group': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
# }
# df = pd.DataFrame(data)

# Output will depend on the data and p, n values
#print(grouped_key_patterns(df, 0.8))
# print(is_dependant(df, 0.8, 0.9))
