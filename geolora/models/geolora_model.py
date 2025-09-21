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
