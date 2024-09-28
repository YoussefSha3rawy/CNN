from torch.utils.data import DataLoader
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
normalize = transforms.Normalize(mean=mean, std=std)


def get_datasets(data_folder, train_folder, val_folder, image_size, batch_size,
                 shuffle, num_workers):
    # Data preprocessing and augmentation
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=(-25, 25), shear=15),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    # Load Dataset
    # train_dataset = JustRAIGSDataset(
    #     **dataset_settings, stage='train', transforms=transform)
    train_dataset = ImageFolder(os.path.join(data_folder, train_folder),
                                transform=train_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # val_dataset = JustRAIGSDataset(
    #     **dataset_settings, stage='test', transforms=transform)

    val_dataset = ImageFolder(os.path.join(data_folder, val_folder),
                              transform=val_transform)
    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def get_test_loader(data_folder, test_folder, image_size, shuffle, batch_size,
                    num_workers):
    test_transform = transforms.Compose([
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = ImageFolder(os.path.join(data_folder, test_folder),
                               transform=test_transform)

    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return test_loader
