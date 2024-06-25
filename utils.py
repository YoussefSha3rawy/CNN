import socket
import time
import argparse
import yaml
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else "cpu")


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(epoch, model, model_name, optimizer):
    ckpt = {'epoch': epoch, 'model_weights': model.state_dict(
    ), 'optimizer_state': optimizer.state_dict()}
    file_name = f"{model_name}.pth"

    directory_name = 'weights'
    os.makedirs(directory_name, exist_ok=True)
    torch.save(ckpt, os.path.join(directory_name, file_name))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='configs/cnn_config.yaml',
                        help='Path to YAML configuration file')
    return parser.parse_args()


def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)

    hostname = socket.gethostname()

    if not hostname.endswith('local'):  # Example check for local machine names
        print("Running on Macbook locally")
    else:
        print(f"Running on remote server: {hostname}")
        settings['dataset']['data_folder'] = settings['dataset']['data_folder_hyperion']

    del settings['dataset']['data_folder_local'], settings['dataset']['data_folder_hyperion']
    return settings


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(
            f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result
    return wrapper
