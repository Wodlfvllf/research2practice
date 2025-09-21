# data/dataset.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional

def get_dataset(dataset_name: str, data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """
    Get train and test datasets
    
    Args:
        dataset_name: Name of dataset ("mnist", "cifar10", etc.)
        data_dir: Directory to store data
    
    Returns:
        train_dataset, test_dataset
    """
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
        
    elif dataset_name.lower() == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform_test
        )
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset
