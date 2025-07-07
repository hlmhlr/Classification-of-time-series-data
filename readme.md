# 🧠 Classification of Time-Series Data

## 📝 Overview
This repository provides an implementation of a **TCN-based deep learning pipeline** for time-series data classification — particularly focused on detecting **electrical faults** in grid station data. The model is built using **Keras** and the **[TCN](https://github.com/philipperemy/keras-tcn)** library, with flexible support for both training and testing via CLI arguments.

---

## 📁 Project Structure
<pre lang="markdown">
.
├── dataset
│   ├── test
│   ├── train
│   └── val
├── environment.yml
├── main.py
├── readme.md
├── utils
│   ├── data_generator.py
│   ├── model.py
│   └── test_plot.py
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
| Argument         | Description                              | Default        |
|------------------|------------------------------------------|----------------|
| `--train`        | Train the model                          | `False`        |
| `--test`         | Test the model                           | `True`         |
| `--dataset_dir`  | Path to the root dataset directory       | *(Required)*   |
| `--model_weights`| Path to `.h5` weights file for testing   | `None`         |
| `--log_dir`      | TensorBoard log directory                | `logs_tcn`     |
| `--epochs`       | Number of training epochs                | `30`           |
| `--batch_size`   | Training batch size                      | `64`           |
| `--lr`           | Learning rate                            | `0.001`        |

## 📈 TensorBoard Logs
To visualize training:
```bash
tensorboard --logdir=logs_tcn
```

## 🧪 Requirements
See environment.yml for all dependencies.

## 📬 Contact
For issues, suggestions, or collaboration, feel free to open an issue.

## 📄 License
To be updated. 