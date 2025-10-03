# geolora_main.py
#!/usr/bin/env python3
"""
Main entry point for GeoLoRA training
"""

import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from config.geolora_config import GeoLoRAConfig
from models.geolora_model import GeoLoRAModel
from training.geolora_trainer import GeoLoRATrainer
from data.dataset import get_dataloaders

# Simple CNN for demonstration
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('geolora_training.log')
        ]
    )

def create_model(dataset: str, config: GeoLoRAConfig, device: str, tokenizer) -> GeoLoRAModel:
    """Create base model and wrap with GeoLoRA"""
    if dataset.lower() == "mnist":
        base_model = SimpleCNN(num_classes=10, input_channels=1)
    elif dataset.lower() == "cifar10":
        base_model = SimpleCNN(num_classes=10, input_channels=3)
    elif dataset.lower() == "qa":
        model_name = "Qwen/Qwen2-0.5B"
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Wrap with GeoLoRA
    geolora_model = GeoLoRAModel(
        base_model=base_model,
        config=config,
        target_modules=["Linear"]  # Apply GeoLoRA to linear layers
    )
    
    return geolora_model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Train model with GeoLoRA")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "qa"],
                       help="Dataset to use")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="Initial LoRA rank")
    parser.add_argument("--tau", type=float, default=1e-3, help="Truncation threshold")
    parser.add_argument("--alpha", type=float, default=1e-4, help="Initialization scale")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Save directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create config
    config = GeoLoRAConfig(
        rank_init=args.rank,
        learning_rate=args.lr,
        tau=args.tau,
        alpha=args.alpha,
        use_small_matrix_optimizer=True,
        truncation_strategy="local_threshold"
    )
    
    logger.info(f"GeoLoRA Config: {config}")

    # Create model
    logger.info("Creating model with GeoLoRA...")
    model, tokenizer = create_model(args.dataset, config, device, None)
    
    # Load data
    logger.info(f"Loading {args.dataset} dataset...")
    train_loader, test_loader = get_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        tokenizer=tokenizer
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"GeoLoRA layers: {list(model.geolora_layers.keys())}")
    
    # Create trainer
    trainer = GeoLoRATrainer(model, config, device)
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = trainer.train(train_loader, test_loader, args.epochs)
    
    # Save model
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history,
        'args': vars(args)
    }
    
    save_path = save_dir / f"geolora_{args.dataset}_rank{args.rank}_epoch{args.epochs}.pt"
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to {save_path}")
    
    # Print final results
    final_train_acc = history['train_accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
    
    logger.info(f"Training completed!")
    logger.info(f"Final train accuracy: {final_train_acc:.2f}%")
    if final_val_acc > 0:
        logger.info(f"Final validation accuracy: {final_val_acc:.2f}%")
    
    # Print rank evolution for each layer
    logger.info("Final layer ranks:")
    for name, layer in model.geolora_layers.items():
        logger.info(f"  {name}: {layer.current_rank}")

if __name__ == "__main__":
    main()