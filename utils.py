import time
import argparse
import os
import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.
                      backends.mps.is_available() else "cpu")


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" %
          (optimizer.param_groups[0]['lr'], ))


def save_checkpoint(dir, epoch, model, optimizer, logger=None):
    ckpt = {
        'epoch': epoch,
        'model_weights': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }

    file_name = f"{model.__class__.__name__}_ckpt.pth"

    os.makedirs(dir, exist_ok=True)
    save_path = os.path.join(dir, file_name)
    torch.save(ckpt, save_path)
    if logger:
        artifact = wandb.Artifact(name=file_name, type="model")
        # Add dataset file to artifact
        artifact.add_file(local_path=save_path)
        logger.log_artifact(artifact)
    return save_path


def load_checkpoint(load_path, model, optimizer=None, device='cpu'):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at '{load_path}'")

    ckpt = torch.load(load_path, map_location=torch.device(device))

    model.load_state_dict(ckpt['model_weights'])

    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state'])

    epoch = ckpt['epoch']

    print(f"Checkpoint loaded from '{load_path}' at epoch {epoch}")

    return model, epoch, optimizer


def load_latest_checkpoint(dir, model, optimizer=None, device='cpu'):
    latest_ckpt = None
    latest_ckpt_epoch = 0
    latest_ckpt_path = ''
    for file_path in os.listdir(dir):
        if file_path.endswith(".pth"):
            ckpt = torch.load(os.path.join(dir, file_path),
                              map_location=torch.device(device))
            if ckpt['epoch'] > latest_ckpt_epoch:
                latest_ckpt = ckpt
                latest_ckpt_epoch = ckpt['epoch']
                latest_ckpt_path = os.path.join(dir, file_path)

    if latest_ckpt:
        model.load_state_dict(latest_ckpt['model_weights'])

        if optimizer:
            optimizer.load_state_dict(latest_ckpt['optimizer_state'])

        epoch = latest_ckpt['epoch']

        print(f"Checkpoint loaded from '{latest_ckpt_path}' at epoch {epoch}")
    else:
        print(f"No checkpoints found at {dir}")

    return model, epoch, optimizer


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Settings')

    # Model arguments
    parser.add_argument('--model',
                        type=str,
                        default='densenet121',
                        help='Model name')

    parser.add_argument('--run_dir',
                        default='',
                        type=str,
                        help='Run directory')

    # Dataset arguments
    parser.add_argument('--data_folder',
                        type=str,
                        help='Path to dataset folder')
    parser.add_argument('--train_folder',
                        type=str,
                        default='train_balanced',
                        help='Name of the train folder')
    parser.add_argument('--val_folder',
                        type=str,
                        default='val',
                        help='Name of the validation folder')
    parser.add_argument('--test_folder',
                        type=str,
                        default='test_upsampled',
                        help='Name of the test folder')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='Size of input images')

    # Dataloader arguments
    parser.add_argument('--shuffle',
                        type=bool,
                        default=True,
                        help='Shuffle the data')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size for training')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='Number of workers for data loading')

    # Training arguments
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of training epochs')
    parser.add_argument('--early_stopping',
                        type=int,
                        default=0,
                        help='Early stopping patience')

    args = parser.parse_args()

    create_new_run_dir(args)

    return args


def create_new_run_dir(args):
    if args.run_dir == '':
        runs_dir = './runs'
        latest_run = 0
        if os.path.exists(runs_dir):
            for dir in os.listdir(runs_dir):
                dir_path = os.path.join(runs_dir, dir)
                if os.path.isdir(dir_path):
                    try:
                        dir_int = int(dir)
                        if dir_int > latest_run:
                            latest_run = dir_int
                    except ValueError:
                        continue

        experiment_run = f'{latest_run + 1}'
        args.run_dir = os.path.join(runs_dir, experiment_run)

    os.makedirs(args.run_dir, exist_ok=True)


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


def get_attention_weights(model, x):
    # Extract the attention weights
    attentions = []

    def hook(module, input, output):
        attentions.append(output)

    # Register the hook to the attention layers (in this case, for all transformer blocks)
    handles = []
    for block in model.blocks:
        handle = block.attn.register_forward_hook(hook)
        handles.append(handle)

    # Forward pass through the model
    with torch.no_grad():
        _ = model(x)

    # Remove the hooks
    for handle in handles:
        handle.remove()

    # Return the attention weights from all layers
    return attentions
