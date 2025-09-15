#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
from .losses.perceptual_loss import VGGPerceptualLoss
from .losses.pixel_loss import ReconstructionLoss
from .models.vit_main import Model
from .data.imagenet_dataset import ImageNetDataset


class MAETrainer:
    """Modified MAE trainer for custom ImageNet dataset"""
    
    def __init__(self, model, device='cuda', extract_aux_features=True):
        self.model = model.to(device)
        self.perceptual_loss_fn = VGGPerceptualLoss().to(device)
        self.reconstruction_loss_fn = ReconstructionLoss().to(device)
        self.device = device
        self.extract_aux_features = extract_aux_features
    
    def compute_loss(self, target, reconstruction, auxiliary, mask):
        """Compute total loss given target and predictions"""
        # Main reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(reconstruction, target, self.model.patch_size, mask)
        
        total_loss = reconstruction_loss
        loss_dict = {'reconstruction': reconstruction_loss.item()}
        
        # Auxiliary losses if available
        if auxiliary and self.extract_aux_features:
            aux_loss_total = 0
            
            if 'hog_pred' in auxiliary and 'hog_targets' in auxiliary:
                hog_loss = nn.MSELoss()(auxiliary['hog_pred'], auxiliary['hog_targets'])
                aux_loss_total += hog_loss
                loss_dict['hog'] = hog_loss.item()
            
            if 'clip_pred' in auxiliary and 'clip_targets' in auxiliary:
                clip_loss = nn.MSELoss()(auxiliary['clip_pred'], auxiliary['clip_targets'])
                aux_loss_total += clip_loss
                loss_dict['clip'] = clip_loss.item()
            
            total_loss += aux_loss_total * 0.1
            loss_dict['auxiliary'] = aux_loss_total.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
    
    def train_one_epoch(self, dataloader, optimizer, epoch, mask_ratio=0.75, max_batches=0):
        """Train one epoch"""
        self.model.train()
        total_losses = {'total': 0, 'reconstruction': 0, 'auxiliary': 0}
        
        # Determine the number of batches to run
        num_batches = len(dataloader)
        if max_batches > 0:
            num_batches = min(num_batches, max_batches)

        for batch_idx, batch_data in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            if isinstance(batch_data, (list, tuple)):
                images, _ = batch_data
            else:
                images = batch_data
            
            images = images.to(self.device)
            
            reconstruction, mask, aux_outputs = self.model(images, mask_ratio=mask_ratio)
            
            loss, loss_dict = self.compute_loss(images, reconstruction, aux_outputs, mask)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            for key in loss_dict:
                if key in total_losses:
                    total_losses[key] += loss_dict[key]
            
            if batch_idx % 100 == 0:
                avg_loss = total_losses['total'] / (batch_idx + 1)
                print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}')
        
        avg_losses = {key: total / num_batches for key, total in total_losses.items()}
        print(f'Epoch {epoch} completed. Average Loss: {avg_losses["total"]:.4f}')
        return avg_losses['total']
    
    def validate(self, dataloader, mask_ratio=0.75):
        self.model.eval()
        total_losses = {'total': 0}
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, (list, tuple)):
                    images, _ = batch_data
                else:
                    images = batch_data
                
                images = images.to(self.device)
                reconstruction, mask, aux_outputs = self.model(images, mask_ratio=mask_ratio)
                loss, loss_dict = self.compute_loss(images, reconstruction, aux_outputs, mask)
                total_losses['total'] += loss_dict['total']
        
        avg_loss = total_losses['total'] / num_batches
        print(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss

    def save_checkpoint(self, epoch, optimizer, loss, filepath):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'extract_aux_features': self.extract_aux_features,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')

def create_data_loaders(dataset_class, data_dir, batch_size=64, img_size=256, num_workers=4):
    train_dataset = dataset_class(root_dir=os.path.join(data_dir, 'train'), split='train', img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    val_dataset = dataset_class(root_dir=os.path.join(data_dir, 'val'), split='val', img_size=img_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader

def train_mae_model(model_class, dataset_class, data_dir, num_epochs=100, 
                   batch_size=64, lr=1e-4, mask_ratio=0.75, save_every=10,
                   extract_aux_features=True, checkpoint_dir='checkpoints', max_batches=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    
    train_loader, val_loader = create_data_loaders(dataset_class, data_dir, batch_size=batch_size, img_size=256)
    
    model = model_class(
        img_size=256, patch_size=16, embed_dim=768,
        encoder_depth=12, encoder_heads=12,
        decoder_depth=8, decoder_heads=16,
        mlp_ratio=4.0,
        use_auxiliary_decoders=extract_aux_features,
    )
    
    trainer = MAETrainer(model, device=device, extract_aux_features=extract_aux_features)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\n--- Epoch {epoch+1}/{num_epochs} ---')
        
        train_loss = trainer.train_one_epoch(train_loader, optimizer, epoch, mask_ratio=mask_ratio, max_batches=max_batches)
        
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_loss = trainer.validate(val_loader, mask_ratio=mask_ratio)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(epoch, optimizer, val_loss, os.path.join(checkpoint_dir, 'mae_best_model.pth'))
        
        scheduler.step()
        
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            trainer.save_checkpoint(epoch, optimizer, train_loss, os.path.join(checkpoint_dir, f'mae_epoch_{epoch}.pth'))
        
        print(f'Epoch {epoch} - LR: {scheduler.get_last_lr()[0]:.6f}')
    
    print('Training completed!')
    return model, trainer

def parse_args():
    p = argparse.ArgumentParser(description='Train MAE model on custom ImageNet dataset')
    p.add_argument('--data-path', required=True, help='Path to ImageNet dataset')
    p.add_argument('--resolution', type=int, default=256, help='Input image resolution')
    p.add_argument('--batch-size', type=int, default=64, help='Batch size')
    p.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--mask-ratio', type=float, default=0.75, help='Masking ratio')
    p.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    p.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint directory')
    p.add_argument('--no-aux-features', action='store_true', help='Disable auxiliary features')
    p.add_argument('--max-batches', type=int, default=0, help='Limit number of batches per epoch for testing.')
    return p.parse_args()

def main():
    args = parse_args()
    
    print("MAE Training Script")
    print(f"Data path: {args.data_path}")
    
    train_mae_model(
        model_class=Model,
        dataset_class=ImageNetDataset,
        data_dir=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        save_every=args.save_every,
        extract_aux_features=not args.no_aux_features,
        checkpoint_dir=args.checkpoint_dir,
        max_batches=args.max_batches
    )

if __name__ == '__main__':
    main()
