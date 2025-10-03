# training/geolora_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
from tqdm import tqdm
import logging

from ..models.geolora_model import GeoLoRAModel
from ..config.geolora_config import GeoLoRAConfig

class GeoLoRATrainer:
    """
    Trainer implementing GeoLoRA Algorithm 1
    """
    
    def __init__(self, 
                 model: GeoLoRAModel,
                 config: GeoLoRAConfig,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer for base parameters (if any are trainable)
        self.base_optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Hooks for capturing gradients
        self._setup_gradient_hooks()
    
    def _setup_gradient_hooks(self):
        """Setup hooks to capture ambient gradients"""
        def create_hook(layer_name):
            def hook_fn(grad):
                # Store ambient gradient for GeoLoRA step
                if layer_name in self.model.geolora_layers:
                    layer = self.model.geolora_layers[layer_name]
                    # Reconstruct ambient gradient from factor gradients
                    layer._ambient_grad = self._reconstruct_ambient_gradient(layer)
                return grad
            return hook_fn
        
        # Register hooks (simplified - would need proper implementation)
        for name, layer in self.model.geolora_layers.items():
            if hasattr(layer, 'U') and layer.U.grad is not None:
                layer.U.register_hook(create_hook(name))
    
    def _reconstruct_ambient_gradient(self, layer: 'GeoLoRALayer') -> torch.Tensor:
        """
        Reconstruct ambient gradient from factor gradients
        This is a simplified version - proper implementation would use autograd
        """
        if layer.U.grad is None or layer.V.grad is None or layer.s.grad is None:
            return torch.zeros(layer.out_features, layer.in_features, device=self.device)
        
        # G ≈ ∇U @ S @ V^T + U @ ∇S @ V^T + U @ S @ ∇V^T
        S_diag = torch.diag(layer.s)
        
        term1 = layer.U.grad @ S_diag @ layer.V.T
        term2 = layer.U @ torch.diag(layer.s.grad) @ layer.V.T
        term3 = layer.U @ S_diag @ layer.V.grad.T
        
        return term1 + term2 + term3
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train one epoch using GeoLoRA"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Standard forward-backward pass
            self.base_optimizer.zero_grad()
            
            outputs = self.model.base_model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Perform GeoLoRA steps for all layers
            self.model.perform_geolora_step(self.config.learning_rate)
            
            # Update base parameters
            self.base_optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}'
            })
        
        return {
            'train_loss': total_loss / len(train_loader)
        }
        
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model.base_model(**batch, labels=batch["input_ids"])
                loss = outputs.loss
                total_loss += loss.item()
        
        return {
            'val_loss': total_loss / len(val_loader)
        }
        
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
              num_epochs: int = 10) -> Dict[str, Any]:
        """Full training loop"""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                
                self.logger.info(
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}"
                )
            else:
                self.logger.info(
                    f"Train Loss: {train_metrics['train_loss']:.4f}"
                )
        
        return history