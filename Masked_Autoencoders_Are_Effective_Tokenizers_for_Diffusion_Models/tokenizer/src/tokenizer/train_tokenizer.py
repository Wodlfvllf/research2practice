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
from tokenizer.losses.perceptual_loss import VGGPerceptualLoss
from tokenizer.losses.pixel_loss import ReconstructionLoss
from tokenizer.models.vit_main import Model
from tokenizer.data.imagenet_dataset import ImageNetDataset


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
        # Convert images to patches for loss computation
        target_patches = self.model.patchify(target)  # (B, num_patches, patch_features)
        
        # Main reconstruction loss
        reconstruction_loss = self.reconstruction_loss_fn(reconstruction, target_patches, mask)
        
        # Apply mask and compute mean loss over masked patches
        masked_loss = (reconstruction_loss * mask).sum() / mask.sum()
        
        total_loss = masked_loss
        loss_dict = {'reconstruction': masked_loss.item()}
        
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
            
            total_loss += aux_loss_total * 0.1  # Weight auxiliary losses
            loss_dict['auxiliary'] = aux_loss_total.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
    
    def train_one_epoch(self, dataloader, optimizer, epoch, mask_ratio=0.75):
        """Train one epoch"""
        self.model.train()
        total_losses = {'total': 0, 'reconstruction': 0, 'auxiliary': 0}
        num_batches = len(dataloader)
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Handle different batch formats from your custom dataset
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    images, _ = batch_data  # (images, labels)
                else:
                    images = batch_data[0]  # Just images
            else:
                images = batch_data
            
            images = images.to(self.device)
            
            # Forward pass through your MAE model
            reconstruction, mask, aux_outputs = self.model(
                images, 
                mask_ratio=mask_ratio
            )
            
            # Compute loss
            loss, loss_dict = self.compute_loss(images, reconstruction, aux_outputs, mask)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            for key in loss_dict:
                if key in total_losses:
                    total_losses[key] += loss_dict[key]
            
            # Log progress
            if batch_idx % 100 == 0:
                avg_loss = total_losses['total'] / (batch_idx + 1)
                print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                      f'Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}')
                
                # Log detailed losses
                if len(loss_dict) > 2:
                    details = ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items() if k != 'total'])
                    print(f'  Details - {details}')
        
        # Compute average losses
        avg_losses = {key: total / num_batches for key, total in total_losses.items()}
        
        print(f'Epoch {epoch} completed. Average Loss: {avg_losses["total"]:.4f}')
        if 'reconstruction' in avg_losses:
            print(f'  Reconstruction: {avg_losses["reconstruction"]:.4f}')
        if 'auxiliary' in avg_losses and avg_losses['auxiliary'] > 0:
            print(f'  Auxiliary: {avg_losses["auxiliary"]:.4f}')
            
        return avg_losses['total']
    
    def validate(self, dataloader, mask_ratio=0.75):
        """Validate the model"""
        self.model.eval()
        total_losses = {'total': 0, 'reconstruction': 0, 'auxiliary': 0}
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Handle different batch formats
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 2:
                        images, _ = batch_data
                    else:
                        images = batch_data[0]
                else:
                    images = batch_data
                
                images = images.to(self.device)
                
                # Forward pass
                reconstruction, mask, aux_outputs = self.model(
                    images, 
                    mask_ratio=mask_ratio
                )
                
                # Compute loss
                loss, loss_dict = self.compute_loss(images, reconstruction, aux_outputs, mask)
                
                # Accumulate losses
                for key in loss_dict:
                    if key in total_losses:
                        total_losses[key] += loss_dict[key]
        
        # Compute average losses
        avg_losses = {key: total / num_batches for key, total in total_losses.items()}
        
        print(f'Validation Loss: {avg_losses["total"]:.4f}')
        if 'reconstruction' in avg_losses:
            print(f'  Reconstruction: {avg_losses["reconstruction"]:.4f}')
        if 'auxiliary' in avg_losses and avg_losses['auxiliary'] > 0:
            print(f'  Auxiliary: {avg_losses["auxiliary"]:.4f}')
            
        return avg_losses['total']
    
    def visualize_reconstruction(self, dataloader, num_samples=4, save_path=None):
        """Visualize reconstructions"""
        self.model.eval()
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Handle different batch formats
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) == 2:
                        images, _ = batch_data
                    else:
                        images = batch_data[0]
                else:
                    images = batch_data
                
                images = images.to(self.device)
                
                # Forward pass
                reconstruction, mask, aux_outputs = self.model(images, mask_ratio=0.75)
                
                # Convert reconstruction to image format if needed
                if hasattr(self.model, 'unpatchify'):
                    reconstructed = self.model.unpatchify(reconstruction)
                else:
                    reconstructed = reconstruction
                
                # Only show first few samples
                images = images[:num_samples]
                reconstructed = reconstructed[:num_samples]
                mask = mask[:num_samples]
                
                # Convert to numpy for visualization
                images_np = images.cpu().numpy()
                reconstructed_np = reconstructed.cpu().numpy()
                mask_np = mask.cpu().numpy()
                
                # Create visualization
                fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*3, 9))
                
                for i in range(num_samples):
                    # Original image
                    img = np.transpose(images_np[i], (1, 2, 0))
                    img = np.clip((img * 0.5) + 0.5, 0, 1)  # Denormalize
                    axes[0, i].imshow(img)
                    axes[0, i].set_title('Original')
                    axes[0, i].axis('off')
                    
                    # Reconstructed image
                    rec_img = np.transpose(reconstructed_np[i], (1, 2, 0))
                    rec_img = np.clip((rec_img * 0.5) + 0.5, 0, 1)  # Denormalize
                    axes[1, i].imshow(rec_img)
                    axes[1, i].set_title('Reconstructed')
                    axes[1, i].axis('off')
                    
                    # Mask visualization
                    patch_size = self.model.patch_size
                    img_size = images.shape[-1]
                    mask_img = self.visualize_mask(mask_np[i], img_size, patch_size)
                    axes[2, i].imshow(mask_img, cmap='gray')
                    axes[2, i].set_title('Mask')
                    axes[2, i].axis('off')
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f'Visualization saved to {save_path}')
                
                return images, reconstructed, mask
    
    def visualize_mask(self, mask, img_size, patch_size):
        """Convert patch mask to image mask"""
        num_patches_per_side = img_size // patch_size
        mask_2d = mask.reshape(num_patches_per_side, num_patches_per_side)
        
        # Upsample mask to image size
        mask_img = np.repeat(mask_2d, patch_size, axis=0)
        mask_img = np.repeat(mask_img, patch_size, axis=1)
        
        return mask_img
    
    def save_checkpoint(self, epoch, optimizer, loss, filepath):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'extract_aux_features': self.extract_aux_features,
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')
    
    def load_checkpoint(self, filepath, optimizer=None):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        # Load auxiliary feature flag if available
        if 'extract_aux_features' in checkpoint:
            self.extract_aux_features = checkpoint['extract_aux_features']
        
        print(f'Checkpoint loaded: {filepath}, Epoch: {epoch}, Loss: {loss}')
        return epoch, loss


def create_data_loaders(dataset_class, data_dir, batch_size=64, img_size=256, num_workers=4):
    """Create train and validation data loaders using your custom ImageNet dataset"""
    
    # Training dataset - assumes your custom dataset takes these parameters
    train_dataset = dataset_class(
        root_dir=data_dir,
        split='train',
        img_size=img_size,
        # Add any other parameters your custom dataset needs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Validation dataset
    val_dataset = dataset_class(
        root_dir=data_dir,
        split='val',
        img_size=img_size,
        # Add any other parameters your custom dataset needs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def train_mae_model(model_class, dataset_class, data_dir, num_epochs=100, 
                   batch_size=64, lr=1e-4, mask_ratio=0.75, save_every=10,
                   extract_aux_features=True, checkpoint_dir='checkpoints'):
    """Complete training pipeline"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    
    # Create data loaders
    print('Creating data loaders...')
    train_loader, val_loader = create_data_loaders(
        dataset_class, data_dir, batch_size=batch_size, img_size=256
    )
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    
    # Create model - you'll need to pass your specific model class
    print('Creating MAE model...')
    model = model_class(
        img_size=256,
        patch_size=16,
        embed_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_depth=8,
        decoder_heads=16,
        use_auxiliary_decoders=extract_aux_features,
    )
    
    # Create trainer
    trainer = MAETrainer(model, device=device, extract_aux_features=extract_aux_features)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.05,
        betas=(0.9, 0.95)
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    print('Starting training...')
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\n--- Epoch {epoch+1}/{num_epochs} ---')
        
        # Train
        train_loss = trainer.train_one_epoch(
            train_loader, optimizer, epoch, mask_ratio=mask_ratio
        )
        
        # Validate
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_loss = trainer.validate(val_loader, mask_ratio=mask_ratio)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(
                    epoch, optimizer, val_loss, 
                    os.path.join(checkpoint_dir, 'mae_best_model.pth')
                )
            
            # Visualize reconstructions occasionally
            if epoch % 20 == 0:
                viz_path = os.path.join(checkpoint_dir, f'reconstruction_epoch_{epoch}.png')
                trainer.visualize_reconstruction(val_loader, num_samples=4, save_path=viz_path)
        
        # Step scheduler
        scheduler.step()
        
        # Save checkpoint
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            trainer.save_checkpoint(
                epoch, optimizer, train_loss, 
                os.path.join(checkpoint_dir, f'mae_epoch_{epoch}.pth')
            )
        
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch} - LR: {current_lr:.6f}')
        print('-' * 50)
    
    print('Training completed!')
    return model, trainer


def parse_args():
    """Parse command line arguments"""
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
    return p.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    print("MAE Training Script")
    print(f"Data path: {args.data_path}")
    print(f"Resolution: {args.resolution}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Mask ratio: {args.mask_ratio}")
    
    model, trainer = train_mae_model(
        model_class=Model,
        dataset_class=ImageNetDataset,
        data_dir=args.data_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        save_every=args.save_every,
        extract_aux_features=not args.no_aux_features,
        checkpoint_dir=args.checkpoint_dir
    )


if __name__ == '__main__':
    main()