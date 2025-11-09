# LDI: Localized Data Imputation for Text-Rich Tables

**LDI** is a novel framework that leverages Large Language Models (LLMs) for imputing missing values in text-rich tabular data, where dependencies are implicit, complex, and scattered across long textual attributes.

Unlike prior LLM-based methods that process entire tables globally, LDI performs localized reasoning by selecting a compact, contextually relevant subset of attributes and tuples for each missing value. This targeted selection significantly improves accuracy, scalability, and explainability.

---

## 🚀 Key Features

**Localized Context Selection**: For each missing value, LDI identifies a minimal and informative subset of related attributes and tuples.

**Explainability**: Every imputed value can be traced back to the specific rows and columns that influenced the prediction.

**Scalability**: Reduces noise and computational overhead by processing smaller, relevant subsets instead of entire tables.

**Model Flexibility**: Works with both hosted LLMs (e.g., GPT-4) and smaller open-source models (e.g., Llama 3.2 3B).


---

## ⚙️ How to Use the Config File

The config file is in JSON format. Below is a guide to each key:

### 🔧 Root-level Keys
- `task`: Type of task (e.g., data imputation).
- `model`: The LLM model to use (e.g., `GPT`, `LLaMA`).
- `na`: How to handle missing values during attribute selection.
- `repeat`: Number of times to repeat the experiment.
- `result_path`: Where to save the final results.
- `output_path`: Where to save logs.

### 📁 `dataset`
- `name`: Dataset name.
- `target_column`: The column that contains missing values (target attribute).

### 📊 `sampling`
- `method`: Sampling method for attribute selection (e.g., `Group Sampling`).
- `number_of_samples`: How many samples to take (for `Random Sampling`).
- `m`: Number of groups (for `Group Sampling`).
- `n`: Number of records per group (for `Group Sampling`).

### 🧪 `examples`
- `method`: How to choose examples.
- `number_of_examples`: How many examples to use.
- `random_seed`: Set a seed for reproducibility.
- `sample_size`: Size of the pool to sample from.
- `rows`: Specific row indices to use instead of random selection.

### 🧠 `dependency_finder`
- `method`: Method used for finding dependent attributes.
- `number_of_rules`: Maximum number of rules to extract (set to `-1` to remove the limit).
- `inner_threshold`: Controls the in-group condition (`q` in the paper).
- `outer_threshold`: Controls the across-group condition (`p` in the paper).

### 🔀 `dataset_partition`
- `train_ratio`: Ratio of training data.
- `number_of_test_rows`: Number of test rows to evaluate.
- `random_seed`: Seed for dataset shuffling.

### 📈 `evaluate`
- `methods`: Evaluation metrics to use (e.g., `exact match`, `BLEU`, `ROUGE`).

---

## ▶️ Installation & Running the Code

Before running the code, make sure to install the required dependencies listed in **requirements.txt** .
1. Install required packages:

<pre> pip install -r requirements.txt </pre>

2. Edit the `config.json` file located next to `main.py` to set your parameters. Save the file.

3. Run the program:

<pre> python main.py </pre>


---

## 📚 `Citation`
If you have used the codes in this repository, we would appreciate it if you cite the LDI paper:

> TBA
