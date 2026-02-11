import random
import string
import math
from collections import Counter
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import seaborn as sns
from tools.dependency_finder import is_dependant
from tools.knn_finder import substring_similarity
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import time
import statistics


def random_string(length):
    return ''.join(random.choices(string.ascii_lowercase, k=length))


def generate_dataframe(num_groups=5, strings_per_group=50, avg_len=50):
    """Generate a DataFrame with random strings across num_groups groups."""
    data = {"text": [], "group": []}
    groups = [chr(65 + i % 26) + str(i // 26)
              for i in range(num_groups)]  # supports >26 groups

    for g in groups:
        for _ in range(strings_per_group):
            data["text"].append(random_string(avg_len))
            data["group"].append(g)
    return pd.DataFrame(data)


def runtime_heatmap(
    df,
    filename="Runtime.pdf",
    xlabel="#Tuples",
    ylabel="Length",
    number_format=".3f",
    number_fontsize=18,
    axis_label_fontsize=25,
    tick_fontsize=22,
    figsize=(10, 8),
    low_color="#89a8f5",
    high_color="#ffff00"
):
    # Green to Yellow colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", [low_color, high_color])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df,
        annot=True,
        fmt=number_format,
        cmap=custom_cmap,
        cbar=True,
        linewidths=0,
        annot_kws={"size": number_fontsize},
        ax=ax
    )

    # Flip y-axis so numbers increase from bottom to top
    ax.invert_yaxis()

    ax.set_xlabel(xlabel, fontsize=axis_label_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(),
                       fontsize=tick_fontsize, rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(),
                       fontsize=tick_fontsize, rotation=0)

    plt.margins(0)  # remove extra margins
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Heatmap saved as {filename}")


def attribute_selection():

    string_lens = [10, 50, 250, 500, 1000]
    # [100, 400, 900, 1600, 2500]
    sample_sizes = [10*10, 20*20, 30*30, 40*40, 50*50]
    p, q = 0.8, 0.8

    data = {}
    for sample_size in sample_sizes:
        size = int(sample_size ** 0.5)
        runtimes = []
        for avg_len in string_lens:
            df = generate_dataframe(size, size, avg_len)
            start = time.time()
            status, result = is_dependant(df, p, q)
            end = time.time()
            runtimes.append(end - start)
        data[sample_size] = runtimes

    df_results = pd.DataFrame(data, index=string_lens)
    df_results.index.name = "StringLen"
    df_results.to_csv("Runtime.csv")
    print(df_results.to_string())

    runtime_heatmap(df_results)

    return df_results


def tuple_selection():
    # Parameters
    string_lengths = [10, 50, 250, 500, 1000]
    sample_sizes = [100, 400, 900, 1600, 2500]
    trials = 1  # number of repeated runs per setting

    results = []
    random.seed(42)  # fix seed for reproducibility

    for length in string_lengths:
        for reps in sample_sizes:
            trial_runtimes = []
            for _ in range(trials):
                # Generate X random strings
                strings = [random_string(length) for _ in range(reps)]
                base = strings[0]

                start = time.time()
                similarities = [substring_similarity(
                    base, s) for s in strings[1:]]
                similarities.sort()
                end = time.time()

                trial_runtimes.append(end - start)

            avg_runtime = statistics.mean(trial_runtimes)
            results.append((length, reps, avg_runtime))

    # Create pivoted DataFrame (rows = String Length, columns = Sample Size)
    df_results = pd.DataFrame(
        results, columns=["String Length", "Sample Size", "Runtime"])
    df_pivot = df_results.pivot(
        index="String Length", columns="Sample Size", values="Runtime")
    df_pivot.to_csv("Runtime.csv")

    print("Average Runtime Table (seconds):")
    print(df_pivot.to_string(float_format="%.3f"))

    runtime_heatmap(df_pivot)

    return df_pivot


# attribute_selection()
tuple_selection()
