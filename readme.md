# Classification of time-series data

This repository provides an implementation of a TCN-based deep learning pipeline for time-series data classification, in particular, the classification of electrical faults experienced during electric data transmissions. The model is built using Keras and the TCN libraries, with flexible support for training and testing through command-line arguments.


---

## 📁 Project Structure

├── main.py # Main training/testing script
├── utils_new7.py # Data loading and preprocessing functions
├── test_plot.py # Evaluation and visualization
├── best_model_tcn.h5 # (Optional) Pretrained model weights
├── logs_tcn/ # Default TensorBoard logs directory
├── README.md
└── ...

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hlmhlr/Classification-of-time-series-data.gitt
cd roi-extraction-tcn
2. Create Environment
Using conda:

bash
Copy
Edit
conda env create -f environment.yml
conda activate roi-extraction-env
OR using pip:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
📂 Dataset Structure
Provide your dataset in the following directory format:

bash
Copy
Edit
/your_dataset_directory/
├── train/
├── val/
└── test/
Each folder should contain the corresponding input data files in a format supported by data_generator() in utils_new7.py.

🚀 How to Use
🔹 Train the Model

python main.py \
  --train \
  --dataset_dir /path/to/your_dataset_directory \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0005
🔹 Test the Model

python main.py \
  --test \
  --dataset_dir /path/to/your_dataset_directory \
  --model_weights /path/to/best_model_tcn.h5
🔹 Train and Test Together

python main.py \
  --train \
  --test \
  --dataset_dir /path/to/your_dataset_directory \
  --model_weights /path/to/best_model_tcn.h5

⚙️ Command-Line Arguments

Argument	Description	Default
--train	Train the model	False
--test	Test the model	True
--dataset_dir	Path to the root dataset directory	(Required)
--model_weights	Path to .h5 weights file for testing	None
--log_dir	TensorBoard log directory	logs_tcn
--epochs	Number of training epochs	30
--batch_size	Training batch size	64
--lr	Learning rate	0.001

📈 TensorBoard Logs
After training, you can view logs by running:

tensorboard --logdir=logs_tcn

🧪 Requirements
See environment.yml

 Contact
For questions or collaboration, please open an issue. 

📄 License
MIT License. See LICENSE file for details.
---
