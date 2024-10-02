import torch
import torch.optim as optim
import torch.nn as nn
from utils import parse_arguments, save_checkpoint, load_latest_checkpoint, get_attention_weights
from logger import Logger
import json
from model import get_model
from train_eval import train_eval
from data import get_datasets, get_test_loader


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available(
    ) else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f'Using device: {device}')

    # Read arguments from command line
    args = parse_arguments()

    # Assign parsed arguments to variables
    model_name = args.model
    data_folder = args.data_folder
    train_folder = args.train_folder
    val_folder = args.val_folder
    test_folder = args.test_folder
    image_size = args.image_size
    shuffle = args.shuffle
    batch_size = args.batch_size
    num_workers = args.num_workers
    lr = args.lr
    num_epochs = args.epochs
    early_stopping = args.early_stopping
    run_dir = args.run_dir

    print(json.dumps(vars(args), indent=4))

    train_loader, val_loader = get_datasets(
        data_folder=data_folder,
        train_folder=train_folder,
        val_folder=val_folder,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    model = get_model(model_name, len(train_loader.dataset.classes))
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, eta_min=1e-5)

    logger = Logger(args,
                    logger_name=model.__class__.__name__,
                    project='FinalProject')
    # Training function
    # Main training loop
    best_val_acc = 0.0
    epochs_since_improvement = 0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, _, _ = train_eval(model=model,
                                                 dataloader=train_loader,
                                                 criterion=criterion,
                                                 optimizer=optimizer,
                                                 scheduler=optimizer_scheduler,
                                                 device=device)
        _, val_acc, all_val_labels, all_val_predictions = train_eval(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device)

        print(f'Epoch [{epoch}/{num_epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%'
              f'Val Acc: {val_acc:.2f}%')

        logger.log_dict({
            'Train Loss': train_loss,
            'Train Acc': train_acc,
            'Val Acc': val_acc
        })
        logger.log_confusion_matrix(all_val_labels, all_val_predictions)

        # Save the model if validation accuracy is the best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_since_improvement = 0
            save_checkpoint(run_dir, epoch, model, optimizer, logger)
        else:
            epochs_since_improvement += 1

        if hasattr(
                args, 'early_stopping'
        ) and early_stopping != 0 and epochs_since_improvement >= early_stopping:
            print(
                f"Validation accuracy did not improve for {epochs_since_improvement} epochs. Stopping training."
            )
            break

    print(f'Best Validation Accuracy: {best_val_acc:.4f}%')

    model, _, _ = load_latest_checkpoint(run_dir, model, device=device)

    test_loader = get_test_loader(
        data_folder=data_folder,
        test_folder=test_folder,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    _, test_acc, all_test_labels, all_test_predictions = train_eval(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device)

    print(f'Test Acc: {test_acc:.4f}%')

    logger.log_dict({'Test Acc': test_acc})
    logger.log_confusion_matrix(all_test_labels, all_test_predictions)


if __name__ == "__main__":
    main()
