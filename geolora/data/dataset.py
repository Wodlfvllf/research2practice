# data/dataset.py
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional
from .qa_dataset import QADataset

def get_dataset(dataset_name: str, data_dir: str = "./data", tokenizer=None, num_samples=100) -> Tuple[Dataset, Dataset]:
    """
    Get train and test datasets
    
    Args:
        dataset_name: Name of dataset ("mnist", "cifar10", "qa")
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

    elif dataset_name.lower() == "qa":
        train_dataset = QADataset(num_samples=num_samples)
        test_dataset = QADataset(num_samples=num_samples // 5)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, test_dataset

def get_dataloaders(dataset_name: str, 
                   batch_size: int = 128,
                   data_dir: str = "./data",
                   num_workers: int = 4,
                   tokenizer = None) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test dataloaders
    
    Args:
        dataset_name: Name of dataset
        batch_size: Batch size
        data_dir: Directory to store data
        num_workers: Number of worker processes
    
    Returns:
        train_loader, test_loader
    """
    if dataset_name == "qa":
        train_dataset, test_dataset = get_dataset(dataset_name, data_dir, tokenizer=tokenizer)

        def collate_fn(batch):
            questions = [item['question'] for item in batch]
            answers = [item['answer'] for item in batch]
            inputs = tokenizer(questions, answers, return_tensors='pt', padding=True, truncation=True)
            return inputs

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

    else:
        train_dataset, test_dataset = get_dataset(dataset_name, data_dir)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, test_loader
