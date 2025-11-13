# LDI: Localized Data Imputation for Text-Rich Tables

**LDI** is a novel framework that leverages Large Language Models (LLMs) for imputing missing values in text-rich tabular data, where dependencies are implicit, complex, and scattered across long textual attributes.

Unlike prior LLM-based methods that process entire tables globally, LDI performs localized reasoning by selecting a compact, contextually relevant subset of attributes and tuples for each missing value. This targeted selection significantly improves accuracy, scalability, and explainability.


## 🚀 Key Features

**Localized Context Selection**: For each missing value, LDI identifies a minimal and informative subset of related attributes and tuples.

**Explainability**: Every imputed value can be traced back to the specific rows and columns that influenced the prediction.

**Scalability**: Reduces noise and computational overhead by processing smaller, relevant subsets instead of entire tables.

**Model Flexibility**: Works with both hosted LLMs (e.g., GPT-4) and smaller open-source models (e.g., Llama 3.2 3B).


## 🧠 Core Idea

LDI decomposes imputation into three stages:

Attribute Selection: Detects dependencies using a relaxed notion of (p, q)-approximate functional dependency, finding text-based patterns (e.g., area codes → cities).

Tuple Selection: Finds similar and diverse examples using Longest Common Substring (LCS) similarity over selected attributes.

LLM-based Imputation: Prompts an LLM with a few-shot context built from selected tuples, guiding the model to infer missing values.

This localized reasoning allows LDI to achieve high accuracy with fewer tokens, enabling efficient and interpretable imputation even for large, heterogeneous tables.

## 📊 Experimental Results

Evaluated on four diverse real-world datasets: Buy, Restaurant, Zomato, and Phone.

Outperforms traditional and modern baselines such as KNN, MIBOS, HoloClean, IPM, and FMW.

Achieved up to 8% higher accuracy compared to state-of-the-art LLM-based methods.

Reduced input size by up to 95.7%, making results easier to interpret and reproduce.

## 🧩 Example

For a table missing the City column:

LDI detects that Phone numbers (e.g., prefixes like 212 or 702) are strongly related.

Selects only the Phone attribute and the most relevant tuples.

Prompts the LLM to infer missing cities based on learned dependencies such as “212 → New York” and “702 → Las Vegas.”

## 💡 Why It Matters

Most existing imputation methods struggle to uncover dependencies hidden in textual fields, where relationships are implied through words, phrases, or patterns rather than exact matches. LDI bridges this gap by detecting such implicit, text-based dependencies—like area codes, product tags, or location hints—and using them for accurate and explainable imputations. This allows LDI to reason over real-world, noisy, and heterogeneous tables where traditional dependency-based or global LLM approaches fail.

---

## ⚙️ How to Use the Config File

The config file is in JSON format. Below is a guide to each key:

### Root-level Keys
- `task`: Type of task (e.g., data imputation).
- `model`: The LLM model to use (e.g., `GPT`, `LLaMA`).
- `na`: How to handle missing values during attribute selection.
- `repeat`: Number of times to repeat the experiment.
- `result_path`: Where to save the final results.
- `output_path`: Where to save logs.

### `dataset`
- `name`: Dataset name.
- `target_column`: The column that contains missing values (target attribute).

### `sampling`
- `method`: Sampling method for attribute selection (e.g., `Group Sampling`).
- `number_of_samples`: How many samples to take (for `Random Sampling`).
- `m`: Number of groups (for `Group Sampling`).
- `n`: Number of records per group (for `Group Sampling`).

### `examples`
- `method`: How to choose examples.
- `number_of_examples`: How many examples to use.
- `random_seed`: Set a seed for reproducibility.
- `sample_size`: Size of the pool to sample from.
- `rows`: Specific row indices to use instead of random selection.

### `dependency_finder`
- `method`: Method used for finding dependent attributes.
- `number_of_rules`: Maximum number of rules to extract (set to `-1` to remove the limit).
- `inner_threshold`: Controls the in-group condition (`q` in the paper).
- `outer_threshold`: Controls the across-group condition (`p` in the paper).

### `dataset_partition`
- `train_ratio`: Ratio of training data.
- `number_of_test_rows`: Number of test rows to evaluate.
- `random_seed`: Seed for dataset shuffling.

### `evaluate`
- `methods`: Evaluation metrics to use (e.g., `exact match`, `BLEU`, `ROUGE`).

---

## ▶️ Installation & Running the Code

Before running the code, please follow these steps:

1. Create and activate a virtual environment
<pre>
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
</pre>

2. Install all dependencies listed in requirements.txt:
<pre> pip install -r requirements.txt </pre>

3. Download datasets

Make sure the datasets are properly downloaded and placed in the following directory structure:
<pre>
data/
  data_imputation/
    <dataset_name>/
      <dataset_file>.csv
</pre>

4. (Optional) Update external model API key

If you are using an external model, update your API key in:
<pre>src/constants/api.py</pre>


5. Configure parameters

Edit the `config.json` file located next to `main.py` to set your parameters. Save the file.

6. Run the program:
<pre> python main.py </pre>


---

## 📚 Citation
If you have used the codes in this repository, we would appreciate it if you cite the LDI paper:

> TBA
