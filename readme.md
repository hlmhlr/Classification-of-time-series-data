# 🧠 Classification of Time-Series Data

## 📝 Overview
This repository provides an implementation of a **TCN-based deep learning pipeline** for time-series data classification — particularly focused on detecting **electrical faults** in grid station data. The model is built using **Keras** and the **[TCN](https://github.com/philipperemy/keras-tcn)** library, with flexible support for both training and testing via CLI arguments.

---

## 📁 Project Structure
<pre lang="markdown">
├── main.py # Main training/testing script
├── utils/
│ ├── data_generator.py # Data loading and preprocessing
│ └── test_plot.py # Evaluation and visualization utilities
├── best_model_tcn.h5 # (Optional) Pretrained model weights
├── logs_tcn/ # TensorBoard logs
├── environment.yml # Conda environment file
└── README.md </pre>

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
🔹 Train the Model
```bash
python main.py \
  --train \
  --dataset_dir /path/to/your_dataset_directory \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0005
```
🔹 Test the Model
```bash
python main.py \
  --test \
  --dataset_dir /path/to/your_dataset_directory \
  --model_weights /path/to/best_model_tcn.h5
```

🔹 Train and Test Together
```bash
python main.py \
  --train \
  --test \
  --dataset_dir /path/to/your_dataset_directory \
  --model_weights /path/to/best_model_tcn.h5
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