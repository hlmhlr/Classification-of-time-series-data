import argparse
from utils.data_generator import data_generator
from utils.test_plot import visualize_training_data, test_analysis
import matplotlib.pyplot as plt
from tcn import compiled_tcn, tcn_full_summary
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, SGD, Nadam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Bidirectional
import numpy as np
from keras import backend as K
import os

TF_ENABLE_ONEDNN_OPTS = 0

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def parse_args():
    parser = argparse.ArgumentParser(description="Train or test the TCN/GRU/LSTM model for time-series classification")

    parser.add_argument("--train", action="store_true", help="Flag to train the model")
    parser.add_argument("--test", action="store_true", default=True, help="Flag to test the model (default: True)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to root dataset directory")
    parser.add_argument("--log_dir", type=str, default="logs_tcn", help="TensorBoard log directory")
    parser.add_argument("--model_weights", type=str, help="Path to model weights for testing")
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory to save result plots and outputs.')

    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument('--num_classes', type=int, default=12, help='Number of output classes.')

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'Nadam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--loss', type=str, default='sparse_categorical_crossentropy', help='Loss function to use')
    parser.add_argument('--model', type=str, default='tcn', choices=['tcn', 'gru', 'lstm'], help='Model to use')

    return parser.parse_args()

def get_optimizer(opt_name, lr):
    if opt_name == 'adam':
        return Adam(learning_rate=lr)
    elif opt_name == 'sgd':
        return SGD(learning_rate=lr)
    elif opt_name == 'Nadam':
        return Nadam(learning_rate=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

def build_model(args, x_train):
    if args.model == 'tcn':
        model = compiled_tcn(
            return_sequences=False,
            num_feat=x_train.shape[2],
            num_classes=args.num_classes,
            nb_filters=40,
            kernel_size=8,
            dilations=[2 ** i for i in range(9)],
            nb_stacks=1,
            max_len=x_train.shape[1],
            use_weight_norm=True,
            use_skip_connections=True,
            dropout_rate=0.05,
            opt=args.optimizer,
            lr=args.lr
        )
    elif args.model == 'gru':
        model = Sequential()
        model.add(GRU(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))    
        model.add(GRU(100, return_sequences=True))
        model.add(GRU(100, return_sequences=False))
        model.add(Dropout(0.05))
        model.add(Dense(64))
        model.add(Dense(args.num_classes, activation='softmax'))
        model.compile(loss=args.loss, metrics=['accuracy'], optimizer=get_optimizer(args.optimizer, args.lr))

    elif args.model == 'lstm':
        model = Sequential()
        model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Bidirectional(LSTM(50, return_sequences=False)))
        model.add(Dropout(0.05))
        model.add(Dense(64))
        model.add(Dense(args.num_classes, activation='softmax'))
        model.compile(loss=args.loss, metrics=['accuracy'], optimizer=get_optimizer(args.optimizer, args.lr))

    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    train_path = os.path.join(args.dataset_dir, "val")
    val_path = os.path.join(args.dataset_dir, "val")
    test_path = os.path.join(args.dataset_dir, "val")

    (x_train, y_train) = data_generator(main_folder=train_path)
    (x_val, y_val) = data_generator(main_folder=val_path)
    (x_test, y_test) = data_generator(main_folder=test_path)

    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')
    print(f'x_test.shape = {x_test.shape}')
    print(f'y_test.shape = {y_test.shape}')

    model = build_model(args, x_train)
    model.summary()

    if args.model == 'tcn':
        tcn_full_summary(model)

    best_weights_path = os.path.join(args.results_dir, f"{args.model}_best_weights.h5")
    log_csv_path = os.path.join(args.results_dir, f"{args.model}_training_log.csv")

    callbacks = [
        keras.callbacks.ModelCheckpoint(best_weights_path, save_best_only=True, monitor="val_loss"),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=0.00001),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        TensorBoard(log_dir=args.log_dir, histogram_freq=1, write_images=True),
        keras.callbacks.CSVLogger(log_csv_path)
    ]

    if args.train:
        print("Training started...")
        history = model.fit(
            x_train, y_train.squeeze().argmax(axis=1),
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val.squeeze().argmax(axis=1)),
            callbacks=callbacks,
            verbose=1
        )
        visualize_training_data(history, output_dir=args.results_dir)
        test_analysis(model, x_test, y_test, output_dir=args.results_dir)

    if args.test:
        print("Testing started...")
        if not args.model_weights:
            raise ValueError("Model weights path must be provided for testing using --model_weights.")
        model.load_weights(args.model_weights)
        test_analysis(model, x_test, y_test, output_dir=args.results_dir)

if __name__ == '__main__':
    main()