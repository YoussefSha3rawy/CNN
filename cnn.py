
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import os
from utils import read_settings, save_checkpoint
from dataset import JustRAIGSDataset
from tqdm import tqdm
from dotenv import load_dotenv
from logger import Logger
import json


# Set device
device = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'Using device: {device}')

# Hyperparameter file path
settings_path = os.path.join('configs', 'cnn_config.yaml')


def main():

    load_dotenv()

    # Read settings from YAML file
    settings = read_settings(settings_path)

    print(json.dumps(settings, indent=4))

    dataset_settings, dataloader_settings, train_settings = settings[
        'dataset'], settings['dataloader'], settings['train']

    # Data preprocessing and augmentation
    transform = models.ResNet50_Weights.DEFAULT.transforms()

    # Load Dataset
    train_dataset = JustRAIGSDataset(
        **dataset_settings, stage='train', transforms=transform)
    train_loader = DataLoader(dataset=train_dataset,
                              **dataloader_settings)

    val_dataset = JustRAIGSDataset(
        **dataset_settings, stage='test', transforms=transform)
    val_loader = DataLoader(dataset=val_dataset,
                            **dataloader_settings)

    # Load pre-trained model and modify the final layer
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), train_settings['lr'])

    logger = Logger(
        settings, logger_name=model.__class__.__name__, project='FinalProject')
    # Training function
    # Main training loop
    best_val_acc = 0.0
    num_epochs = train_settings['epochs']
    for epoch in range(num_epochs):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        logger.log({'Train Loss': train_loss,
                    'Train Acc': train_acc,
                    'Val Loss': val_loss,
                    'Val Acc': val_acc})

        # Save the model if validation accuracy is the best
        epochs_since_improvement = 0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            save_checkpoint(epoch, model, optimizer, logger)
        else:
            epochs_since_improvement += 1

        if hasattr(train_settings, 'early_stopping') and train_settings['early_stopping'] != 0 and epochs_since_improvement >= train_settings['early_stopping']:
            print(
                f"Validation accuracy did not improve for {epochs_since_improvement} epochs. Stopping training.")
            break

    print(f'Best Validation Accuracy: {best_val_acc:.2f}%')


def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader, desc='Training batch'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    main()
