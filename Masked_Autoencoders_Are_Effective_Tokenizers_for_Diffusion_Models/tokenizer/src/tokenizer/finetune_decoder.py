#!/usr/bin/env python3
import argparse
import torch
import torch.optim as optim
import os
from tokenizer.models.vit_main import Model
from tokenizer.data.imagenet_dataset import ImageNetDataset
from tokenizer.losses.adversarial_loss import PatchDiscriminator
from tokenizer.train_tokenizer import MAETrainer, create_data_loaders

class DecoderFinetuner(MAETrainer):
    """
    A trainer specifically for fine-tuning the MAE decoder.
    It freezes the encoder and trains the decoder and discriminator.
    """
    def __init__(self, model, device='cuda', extract_aux_features=True):
        super().__init__(model, device, extract_aux_features)
        
        # Freeze the encoder parameters
        print("Freezing encoder parameters...")
        for name, param in self.model.named_parameters():
            if 'encoder' in name:
                param.requires_grad = False
        
        # Initialize the discriminator for adversarial loss
        self.discriminator = PatchDiscriminator(
            in_channels=model.in_channels,
            patch_size=model.patch_size
        ).to(device)

    def compute_loss(self, target, reconstruction, auxiliary, mask):
        """
        Override the loss computation to include adversarial loss.
        """
        # 1. Reconstruction and Perceptual Loss (from parent class)
        recon_loss, loss_dict = super().compute_loss(target, reconstruction, auxiliary, mask)
        
        # 2. Adversarial Loss
        # We need to reshape the patches for the discriminator
        recon_patches_img = self.model.unpatchify(reconstruction)
        
        # Generator loss
        g_loss = self.adversarial_loss_fn(recon_patches_img, target, mask, 'generator')
        
        # Discriminator loss
        d_loss = self.adversarial_loss_fn(recon_patches_img, target, mask, 'discriminator')
        
        # Combine losses
        # The paper suggests a weighted sum, e.g., lambda_adv * g_loss
        total_loss = recon_loss + 0.1 * g_loss
        
        loss_dict['generator_adv'] = g_loss.item()
        loss_dict['discriminator_adv'] = d_loss.item()
        loss_dict['total'] = total_loss.item()
        
        # We need to return the discriminator loss separately to update its optimizer
        return total_loss, d_loss, loss_dict

    def train_one_epoch(self, dataloader, optim_g, optim_d, epoch, mask_ratio=0.75):
        """
        Modified training loop for two optimizers (generator and discriminator).
        """
        self.model.train()
        self.discriminator.train()
        
        total_losses = {'total': 0, 'reconstruction': 0, 'generator_adv': 0, 'discriminator_adv': 0}
        num_batches = len(dataloader)
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(self.device)
            
            # --- Train Discriminator ---
            optim_d.zero_grad()
            
            with torch.no_grad():
                reconstruction, mask, _ = self.model(images, mask_ratio=mask_ratio)
            
            _, d_loss, _ = self.compute_loss(images, reconstruction, None, mask)
            
            d_loss.backward()
            optim_d.step()
            
            # --- Train Generator (Decoder) ---
            optim_g.zero_grad()
            
            reconstruction, mask, aux_outputs = self.model(images, mask_ratio=mask_ratio)
            g_loss, _, loss_dict = self.compute_loss(images, reconstruction, aux_outputs, mask)
            
            g_loss.backward()
            optim_g.step()
            
            # Accumulate and log
            for key in loss_dict:
                if key in total_losses:
                    total_losses[key] += loss_dict[key]
            
            if batch_idx % 100 == 0:
                avg_loss = total_losses['total'] / (batch_idx + 1)
                print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, G_Loss: {g_loss.item():.4f}, D_Loss: {d_loss.item():.4f}')
        
        avg_losses = {key: total / num_batches for key, total in total_losses.items()}
        print(f'Epoch {epoch} completed. Avg G_Loss: {avg_losses["total"]:.4f}, Avg D_Loss: {avg_losses["discriminator_adv"]:.4f}')
        
        return avg_losses['total']


def finetune_decoder(model_class, dataset_class, data_dir, pretrained_checkpoint, num_epochs=50, 
                     batch_size=64, lr=5e-5, mask_ratio=0.75, save_every=5,
                     extract_aux_features=True, checkpoint_dir='checkpoints/finetuned'):
    """
    Main pipeline for fine-tuning the decoder.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Fine-tuning on device: {device}')
    
    train_loader, val_loader = create_data_loaders(dataset_class, data_dir, batch_size=batch_size)
    
    # Load the pre-trained model
    model = model_class(use_auxiliary_decoders=extract_aux_features)
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pre-trained model from {pretrained_checkpoint}")
    
    # Create the fine-tuner
    finetuner = DecoderFinetuner(model, device=device, extract_aux_features=extract_aux_features)
    
    # Optimizers: one for the generator (decoder) and one for the discriminator
    optim_g = optim.AdamW(
        filter(lambda p: p.requires_grad, finetuner.model.parameters()), 
        lr=lr, weight_decay=0.05
    )
    optim_d = optim.AdamW(
        finetuner.discriminator.parameters(), 
        lr=lr * 2, weight_decay=0.05 # Discriminator often uses a slightly higher LR
    )
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'\n--- Finetune Epoch {epoch+1}/{num_epochs} ---')
        
        train_loss = finetuner.train_one_epoch(
            train_loader, optim_g, optim_d, epoch, mask_ratio=mask_ratio
        )
        
        if epoch % save_every == 0 or epoch == num_epochs - 1:
            finetuner.save_checkpoint(
                epoch, optim_g, train_loss, 
                os.path.join(checkpoint_dir, f'finetuned_decoder_epoch_{epoch}.pth')
            )
            # Also save discriminator
            torch.save(finetuner.discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))

    print('Decoder fine-tuning completed!')

def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune the MAE decoder.')
    p.add_argument('--data-path', required=True, help='Path to ImageNet dataset')
    p.add_argument('--pretrained-checkpoint', required=True, help='Path to the pre-trained MAE model checkpoint.')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--mask-ratio', type=float, default=0.75)
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--checkpoint-dir', default='checkpoints/finetuned')
    p.add_argument('--no-aux-features', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    finetune_decoder(
        model_class=Model,
        dataset_class=ImageNetDataset,
        data_dir=args.data_path,
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        save_every=args.save_every,
        extract_aux_features=not args.no_aux_features,
        checkpoint_dir=args.checkpoint_dir
    )
