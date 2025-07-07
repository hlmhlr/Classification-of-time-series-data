# 🧠 Classification of Time-Series Data

## 📝 Overview
This repository provides an implementation of a deep learning pipeline for **time-series data classification**, specifically focused on detecting **electrical faults** in grid station data. It supports multiple model architectures — **TCN**, **GRU**, and **LSTM** — with configurable options through **command-line arguments**.

The project is built using **Keras**, with optional support for the **[TCN](https://github.com/philipperemy/keras-tcn)** library. It provides a clean, modular structure for training, testing, evaluation, and result visualization, making it easy to extend or adapt for similar time-series classification tasks.

---

## 📁 Project Structure
<pre lang="markdown">
.
├── main.py
├── utils
│   ├── data_generator.py
│   ├── model.py
│   └── test_plot.py
├── dataset
│   ├── test
│   ├── train
│   └── val
├── environment.yml
├── readme.md
└── weights
    └── best_model.h5 </pre>

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hlmhlr/Classification-of-time-series-data.git
cd Classification-of-time-series-data
```

### 2. Create Environment
Using conda:
```bash
conda env create -f environment.yml
conda activate time-class-env
```

## 📂 Dataset Structure
Your dataset should follow this structure:

/your_dataset_directory/
<pre lang="markdown">
├── train/
├── val/
└── test/  </pre>
Each folder should contain the input data files, supported by data_generator() in utils/data_generator.py.


```
<start_time>_<end_time>_<distance>_km_Rf_<Rf>_Rg_<Rg>_FIA_<angle>
```

Example:

```
0.2_0.3_150_km_Rf_0.001_Rg_0.01_FIA_30
```

Where:
- `0.2_0.3` — Fault duration in seconds  
- `150_km` — Distance from source  
- `Rf`, `Rg` — Fault and ground resistance values  
- `FIA` — Fault incidence angle in degrees  

These files are processed using the `data_generator.py` module under the `utils/` folder.



## 🚀 How to Use

### 🔧 Train the Model

```bash
# TCN Model
python main.py \
  --train \
  --model tcn \
  --dataset_dir /path/to/your_dataset_directory \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0005 \
  --optimizer adam \
  --loss sparse_categorical_crossentropy \
  --results_dir ./results_tcn
```
```bash
# GRU Model
python main.py \
  --train \
  --model gru \
  --dataset_dir /path/to/your_dataset_directory \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0005 \
  --optimizer adam \
  --loss sparse_categorical_crossentropy \
  --results_dir ./results_gru
```

```bash
# LSTM Model
python main.py \
  --train \
  --model lstm \
  --dataset_dir /path/to/your_dataset_directory \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0005 \
  --optimizer adam \
  --loss sparse_categorical_crossentropy \
  --results_dir ./results_lstm
  ```

🧪 Test the Model

```bash
# Test TCN Model
python main.py \
  --test \
  --model tcn \
  --dataset_dir /path/to/your_dataset_directory \
  --model_weights ./results_tcn/tcn_best_weights.h5 \
  --results_dir ./results_tcn
```

```bash
# Test GRU Model
python main.py \
  --test \
  --model gru \
  --dataset_dir /path/to/your_dataset_directory \
  --model_weights ./results_gru/gru_best_weights.h5 \
  --results_dir ./results_gru
```
```bash
# Test LSTM Model
python main.py \
  --test \
  --model lstm \
  --dataset_dir /path/to/your_dataset_directory \
  --model_weights ./results_lstm/lstm_best_weights.h5 \
  --results_dir ./results_lstm
```


## ⚙️ Command-Line Arguments
## ⚙️ Command-Line Arguments

| Argument           | Description                                                   | Default                          |
|--------------------|---------------------------------------------------------------|----------------------------------|
| `--train`          | Train the model                                               | `False`                          |
| `--test`           | Test the model                                                | `False`                          |
| `--dataset_dir`    | Path to the root dataset directory                            | **(Required)**                   |
| `--model_type`     | Type of model to use: `tcn`, `gru`, or `lstm`                 | `tcn`                            |
| `--model_weights`  | Path to `.h5` weights file for loading/saving model           | `None`                           |
| `--output_dir`     | Directory to save outputs (logs, plots, model weights, etc.)  | `results`                        |
| `--log_dir`        | TensorBoard log directory                                     | `logs_tcn`                       |
| `--epochs`         | Number of training epochs                                     | `30`                             |
| `--batch_size`     | Training batch size                                           | `64`                             |
| `--lr`             | Learning rate                                                 | `0.001`                          |
| `--optimizer`      | Optimizer to use: `adam`, `rmsprop`, etc.                     | `adam`                           |
| `--loss`           | Loss function for model compilation                           | `sparse_categorical_crossentropy` |
| `--num_classes`    | Number of output classes                                      | `12`                             |



## 📈 TensorBoard Logs
To visualize training:
```bash
tensorboard --logdir=logs_tcn
```

## 📊 Model Output: Confusion Matrix

After training or testing the model, performance is evaluated using class-wise accuracies and a **confusion matrix** to visualize prediction results across all classes.

This matrix provides insight into how well the model is classifying each fault type based on the true vs. predicted labels.

Example output:

![Confusion Matrix](results/confusion_matrix_example.png)

> You can find your confusion matrix plots and evaluation results saved in the `results/` directory after model testing.



## 🧪 Requirements
See environment.yml for all dependencies.

## 📬 Contact
For issues, suggestions, or collaboration, feel free to open an issue.

## 📄 License
To be updated. 