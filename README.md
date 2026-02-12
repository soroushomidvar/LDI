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

The config file is in YAML format (`config.yaml`). Below is a guide to each key:

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

This project uses `uv` for fast dependency management.

### Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync dependencies**:
   ```bash
   uv sync
   ```

3. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

4. **Download additional data (first time only)**:
   ```bash
   python -m spacy download en_core_web_sm
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('brown'); nltk.download('reuters'); nltk.download('gutenberg')"
   ```

### Running the Project

1. **Run**:
   ```bash
   python run.py
   # Or with a custom config file:
   python run.py path/to/config.yaml
   # Or using uv directly:
   uv run python run.py
   uv run python run.py my_config.yaml
   ```

### Data Setup

Make sure the datasets are properly downloaded and placed in the following directory structure:
```
data/
  data_imputation/
    <dataset_name>/
      <dataset_file>.csv
```

> **Note:** The dataset files are tracked using **Git LFS (Large File Storage)**.
> If you clone the repository without Git LFS installed, you may see placeholder files like:
>
> ```
> version https://git-lfs.github.com/spec/v1
> oid sha256:...
> size ...
> ```
>
> instead of the actual CSV content.
>
> To download the real dataset files, please install Git LFS and run:
>
> ```bash
> git lfs install
> git lfs pull
> ```
>
> After that, the full CSV files will be available in the directories above.

### Configuration

Edit the `config.yaml` file located in the root directory to set your parameters.

### API Keys

API keys are loaded from a `.env` file in the project root. Create a `.env` file with your keys:

```
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-gemini-api-key
REPLICATE_API_TOKEN=your-replicate-api-token
```

Only the keys for the model you plan to use need to be set.

---

## 📚 Citation
If you have used the codes in this repository, we would appreciate it if you cite the LDI paper:

> TBA
