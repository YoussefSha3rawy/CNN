import torch
from tqdm import tqdm


def train_eval(model,
               dataloader,
               criterion,
               optimizer=None,
               scheduler=None,
               device='cpu'):
    is_train = optimizer is not None
    if is_train:
        model.train()
        grad = torch.enable_grad
    else:
        model.eval()
        grad = torch.no_grad

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_predictions = [], []

    with grad():
        for inputs, labels in tqdm(
                dataloader,
                desc=f'{"Training" if is_train else "Evaluation"} batch'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_labels, all_predictions
