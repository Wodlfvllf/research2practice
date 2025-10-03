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
        
        self._add_geolora_layers(self.base_model, target_modules)
    
    def _add_geolora_layers(self, module: nn.Module, target_modules: List[str], parent_name=""):
        """Recursively add GeoLoRA layers to specified modules"""
        for name, child_module in module.named_children():
            full_name = f"{parent_name}.{name}" if parent_name else name

            if any(target in child_module.__class__.__name__ for target in target_modules) and isinstance(child_module, nn.Linear):
                # Create GeoLoRA layer
                geolora_layer = GeoLoRALayer(
                    in_features=child_module.in_features,
                    out_features=child_module.out_features,
                    config=self.config,
                    base_layer=child_module
                )
                
                self.geolora_layers[full_name] = geolora_layer
                setattr(module, name, geolora_layer)

            else:
                self._add_geolora_layers(child_module, target_modules, full_name)

    def forward(self, *args, **kwargs):
        """Forward pass through base model with GeoLoRA adaptations"""
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
