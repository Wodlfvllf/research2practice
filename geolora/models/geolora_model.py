# models/geolora_model.py
import torch
import torch.nn as nn
from typing import Dict, List
from .geolora_layer import GeoLoRALayer
from ..config.geolora_config import GeoLoRAConfig

class GeoLoRAModel(nn.Module):
    """
    Wrapper that adds GeoLoRA layers to a base model
    """
    
    def __init__(self, base_model: nn.Module, config: GeoLoRAConfig, 
                 target_modules: List[str] = None):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        self.geolora_layers: Dict[str, GeoLoRALayer] = {}
        
        # Default target modules (typically linear layers)
        if target_modules is None:
            target_modules = ["linear", "Linear"]
        
        self._add_geolora_layers(target_modules)
        
        # Track original forward pass
        self._wrap_forward()
    
    def _add_geolora_layers(self, target_modules: List[str]):
        """Add GeoLoRA layers to specified modules"""
        for name, module in self.base_model.named_modules():
            if any(target in module.__class__.__name__.lower() for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Create GeoLoRA layer
                    geolora_layer = GeoLoRALayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        config=self.config,
                        base_layer=module
                    )
                    
                    self.geolora_layers[name] = geolora_layer
                    self.add_module(f"geolora_{name.replace('.', '_')}", geolora_layer)

    def _wrap_forward(self):
        """Wrap base model forward to include GeoLoRA adaptations"""
        # This is a simplified version - in practice you'd need more sophisticated hooking
        pass
    
    def forward(self, *args, **kwargs):
        """Forward pass through base model with GeoLoRA adaptations"""
        # For simplicity, assuming we can directly call base model
        # In practice, you'd need to hook into intermediate layers
        return self.base_model(*args, **kwargs)
    
    def geolora_parameters(self):
        """Return GeoLoRA parameters for optimization"""
        params = []
        for layer in self.geolora_layers.values():
            params.extend([layer.U, layer.V, layer.s])
        return params
    
    def perform_geolora_step(self, learning_rate: float):
        """Perform GeoLoRA update for all layers"""
        for name, layer in self.geolora_layers.items():
            # Get ambient gradient for this layer
            # This would need to be computed during backward pass
            if hasattr(layer, '_ambient_grad') and layer._ambient_grad is not None:
                layer.geolora_step(layer._ambient_grad, learning_rate)
                layer._ambient_grad = None  # Clear after use