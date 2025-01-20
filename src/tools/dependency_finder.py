import pandas as pd
from collections import defaultdict
from tools.word_frequency import *


def common_patterns(strings, p):
    def generate_substrings(s):
        s = str(s)
        # Generate all substrings of a given string.
        substrings = set()
        length = len(s)
        for i in range(length):
            for j in range(i + 1, length + 1):
                substrings.add(s[i:j])
        return substrings

    def count_substring_occurrences(strings):
        # Count the occurrences of each substring in the list of strings.
        substring_count = defaultdict(int)
        for s in strings:
            found_substrings = generate_substrings(s)
            for substring in found_substrings:
                substring_count[substring] += 1
        return substring_count

    def filter_substrings(substring_count, threshold, total_strings):
        # Filter substrings based on the percentage threshold.
        return [substring for substring, count in substring_count.items()
                if count / total_strings >= threshold]

    # def remove_substrings_within_longer(substrings):
    #     # Remove substrings that are contained within longer substrings.
    #     sorted_substrings = sorted(substrings, key=lambda x: -len(x))
    #     filtered_substrings = []
    #     seen = set()

    #     for substring in sorted_substrings:
    #         if not any(substring in seen_substring for seen_substring in seen):
    #             filtered_substrings.append(substring)
    #             seen.add(substring)

    #     return filtered_substrings

    threshold = p
    total_strings = len(strings)

    # Generate the substring counts
    substring_count = count_substring_occurrences(strings)

    # Filter substrings that appear in at least p*100 percent of strings
    valid_substrings = filter_substrings(
        substring_count, threshold, total_strings)

    # Remove substrings that are contained within longer substrings
    # filtered_substrings = remove_substrings_within_longer(valid_substrings)

    # Sort substrings by length (longest first) and then alphabetically
    sorted_substrings = sorted(valid_substrings, key=lambda x: (-len(x), x))

    # Return the top n substrings
    return sorted_substrings

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


def grouped_key_patterns(df, p):
    #df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x) # to lowercase
    grouped = df.groupby(df.columns[1])
    all_substrings = defaultdict(set)
    group_substrings = {}

    # Collect substrings for each group
    for group, subset in grouped:
        strings = subset[df.columns[0]].tolist()
        top_substrings = common_patterns(strings, p)

        group_substrings[group] = top_substrings
        for substring in top_substrings:
            all_substrings[substring].add(group)

    result = {}

    # # Filter substrings to include only those unique to one group
    # for group, substrings in group_substrings.items():
    #     unique_substrings = [substring for substring in substrings if len(
    #         all_substrings[substring]) == 1]

    #     for substring in unique_substrings:
    #         result[substring] = group

    # Filter substrings to include only those unique to one group
    for group, substrings in group_substrings.items():
        unique_substrings = []

        # Check if a substring is a part of a longer substring in other groups
        for substring in substrings:
            is_unique = True
            for other_substring in all_substrings:
                #TODO
                if substring != other_substring and substring in other_substring:
                    # If the substring is part of a longer substring in another group, it's not unique
                    if group not in all_substrings[other_substring]:
                        is_unique = False
                        break

            # Only keep substring if it's unique
            if is_unique: # and len(all_substrings[substring]) == 1: #TODO
                unique_substrings.append(substring)

        # Add unique substrings to result
        for substring in unique_substrings:
            result[substring] = group

    # Remove substrings that are contained within longer substrings
    long_result = remove_substrings_within_longer(result)

    # Sort the result by substring length and alphabetically
    sorted_result = dict(
        sorted(long_result.items(), key=lambda x: (-len(x[0]), x[0])))

    # Return the top n results
    return dict(list(sorted_result.items()))


def is_dependant(df, p, q):
    result = grouped_key_patterns(df, p)
    # TODO
    groups_with_unique_substrings = set(result.values())
    total_groups = df[df.columns[1]].nunique()
    required_groups = q * total_groups
    status = len(groups_with_unique_substrings) >= required_groups
    # degree = len(groups_with_unique_substrings)/total_groups
    frequency = -1 # word_frequency(list(result.keys())) if status else -1
    return status, frequency, result


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
