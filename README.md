# Project Readme

## Project Overview

This project is designed for training a deep learning model using PyTorch for image classification tasks. It includes components for dataset loading, training and validation loops, logging, and checkpointing.

### Features
- Model architecture selection (default: DenseNet121).
- Customizable dataset paths and training parameters.
- Supports multiple devices: CUDA, MPS (for Apple Silicon), and CPU.
- Logging and visualization of performance metrics, including confusion matrices.
- Early stopping to prevent overfitting.

## Requirements

To run this project, the following libraries are required:

```bash
torch
torchvision
argparse
json
```

Install them using pip:
```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Arguments

You can pass various arguments to customize the training and evaluation process:

| Argument             | Type   | Default         | Description                                            |
|----------------------|--------|-----------------|--------------------------------------------------------|
| `--model`            | string | densenet121      | Name of the model architecture.                        |
| `--run_dir`          | string | ""              | Directory for saving checkpoints and logs.             |
| `--data_folder`      | string | None            | Path to the root dataset folder.                       |
| `--train_folder`     | string | train_balanced  | Folder name containing training data.                  |
| `--val_folder`       | string | val             | Folder name containing validation data.                |
| `--test_folder`      | string | test_upsampled  | Folder name containing test data.                      |
| `--image_size`       | int    | 224             | Size of input images (height x width).                 |
| `--shuffle`          | bool   | True            | Whether to shuffle the dataset during loading.         |
| `--batch_size`       | int    | 64              | Batch size for training and evaluation.                |
| `--num_workers`      | int    | 8               | Number of workers for data loading.                    |
| `--lr`               | float  | 0.001           | Learning rate for the optimizer.                       |
| `--epochs`           | int    | 100             | Number of epochs for training.                         |
| `--early_stopping`   | int    | 0               | Number of epochs for early stopping patience (0 = off).|

### Example Command
```bash
python main.py –data_folder /path/to/dataset –epochs 50 –lr 0.0001
```

## Key Components

### Dataset Handling

The datasets are expected to be structured in the following way:
```
data_folder/
  ├── train_balanced/
  │     ├── class1/
  │     ├── class2/
  │     └── ...
  ├── val/
  │     ├── class1/
  │     ├── class2/
  │     └── ...
  └── test_upsampled/
        ├── class1/
        ├── class2/
        └── ...
```
### Model Selection

The model is selected using the `--model` argument. The default model is DenseNet121. You can define your own model by adding it in the `model.py` file and specifying its name when running the script.

### Training and Validation

The `train_eval` function handles both training and validation:

- **Training**: The model is trained using the Adam optimizer with a cosine annealing warm restart scheduler.
- **Validation**: After each epoch, the model is evaluated on the validation set. If the validation accuracy improves, the model is saved.

### Logging

The training process is logged using a custom logger that stores metrics like training loss, accuracy, and confusion matrices. These are useful for visualizing the model's performance.

### Checkpointing

Checkpoints are saved when the validation accuracy improves, allowing you to restore the best model later using `load_latest_checkpoint`.

## Logging and Visualizations

- The logger records training and validation performance metrics.
- Confusion matrices are logged for both validation and test sets.
- The final test accuracy is reported after loading the best model from checkpoints.

## Device Support

The script supports multiple devices, automatically selecting the best available:

- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon)
- **CPU**

## Early Stopping

Early stopping is configured through the `--early_stopping` argument. If validation accuracy does not improve for the specified number of epochs, training is halted to prevent overfitting.

## Running Tests

After training, the model is evaluated on the test dataset. You can specify the test dataset folder using `--test_folder`. The results will be logged, including the test accuracy and confusion matrix.

## Saving and Loading Checkpoints

The model checkpoints are saved in the directory specified by `--run_dir`. The best model based on validation accuracy is saved, and you can reload it for further evaluation or testing using `load_latest_checkpoint`.

---