import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os

# Uncomment the second function for data_augmentation dataloader.
def get_data_loaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel image
        transforms.Resize((224, 224)),  # Resize images to match ResNet/MobileNet input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    trainset = ImageFolder(root=train_dir, transform=transform)
    valset = ImageFolder(root=val_dir, transform=transform)
    testset = ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

# def get_data_loaders(data_dir, batch_size):
#     # **Data Augmentation for Training Set**
#     train_transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
#         transforms.Resize((224, 224)),  # Resize to match model input
#         transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images
#         transforms.RandomRotation(10),  # Rotate images by up to ±10 degrees
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust colors
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly crop images
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Normalize
#     ])

#     # **No Augmentation for Validation & Test Set**
#     val_test_transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     train_dir = os.path.join(data_dir, 'train')
#     val_dir = os.path.join(data_dir, 'val')
#     test_dir = os.path.join(data_dir, 'test')

#     trainset = ImageFolder(root=train_dir, transform=train_transform)
#     valset = ImageFolder(root=val_dir, transform=val_test_transform)
#     testset = ImageFolder(root=test_dir, transform=val_test_transform)

#     train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
#     val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
#     test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

#     return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_dir = '/home/ashmal/Courses/MedImgComputing/Assignment_1/data/2/chest_xray/chest_xray'
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=16)

    for batch_idx, (data, target) in enumerate(val_loader):
        print(f"Batch {batch_idx}: Data shape {data.shape}, Target {target}")
        break
