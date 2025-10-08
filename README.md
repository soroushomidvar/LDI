# LDI: Localized Data Imputation

**LDI** is a method that uses LLMs to fill in missing values in tabular data, which improves both the **accuracy** and **explainability** of imputations by selecting only the attributes and tuples that can contribute to the prediction.

---

## ⚙️ How to Use the Config File

The config file is in JSON format. Below is a guide to each key:

### 🔧 Root-level Keys
- `task`: Type of task (e.g., data imputation).
- `model`: The LLM model to use (e.g., GPT, LLaMA).
- `na`: How to handle missing values during attribute selection.
- `repeat`: Number of times to repeat the experiment.
- `result_path`: Where to save the final results.
- `output_path`: Where to save logs.

### 📁 `dataset`
- `name`: Dataset name.
- `target_column`: The column that contains missing values (target attribute).

### 📊 `sampling`
- `method`: Sampling method for attribute selection.
- `number_of_samples`: How many samples to take.
- `m`: Number of groups.
- `n`: Number of records per group.

### 🧪 `examples`
- `method`: How to choose examples (e.g., random).
- `number_of_examples`: How many examples to use.
- `random_seed`: Set a seed for reproducibility.
- `sample_size`: Size of the pool to sample from.
- `rows`: Specific row indices to use instead of random selection.

### 🧠 `dependency_finder`
- `method`: Method used for finding dependent attributes.
- `number_of_rules`: Maximum number of rules to extract.
- `inner_threshold`: Controls the in-group condition (`q` in the paper).
- `outer_threshold`: Controls the across-group condition (`p` in the paper).

### 🔀 `dataset_partition`
- `train_ratio`: Ratio of training data.
- `number_of_test_rows`: Number of test rows to evaluate.
- `random_seed`: Seed for dataset shuffling.

### 📈 `evaluate`
- `methods`: Evaluation metrics to use (e.g., exact match, BLEU, ROUGE).

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

> TBD
