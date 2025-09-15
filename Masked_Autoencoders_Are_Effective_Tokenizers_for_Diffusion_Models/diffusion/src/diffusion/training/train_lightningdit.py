import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os

# Import the model and tokenizer
from diffusion.src.diffusion.models.lightningdit_model import LightningDiT
from tokenizer.src.tokenizer.models.vit_main import Model as MAETokenizer

# Import the dataset and transforms from the tokenizer module
from tokenizer.src.tokenizer.data.imagenet_dataset import ImageNetDataset
from tokenizer.src.tokenizer.data.transforms import DataTransforms

def get_alpha_schedule(timesteps=1000, beta_start=0.0001, beta_end=0.02):
    """
    Creates a linear beta schedule and computes alphas.
    """
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas_cumprod

def q_sample(x_start, t, alphas_cumprod, noise=None):
    """
    Forward diffusion process.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(x_start.shape[0], 1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t]).view(x_start.shape[0], 1, 1)

    noisy_latents = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_latents

def main():
    parser = argparse.ArgumentParser(description='Train LightningDiT model.')
    parser.add_argument('--data-path', required=True, help='Path to ImageNet dataset')
    parser.add_argument('--tokenizer-path', required=True, help='Path to pre-trained MAE tokenizer checkpoint.')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--checkpoint-dir', default='diffusion/checkpoints/lightningdit_model', help='Directory to save checkpoints.')
    # Add model parameters as arguments
    parser.add_argument('--hidden-size', type=int, default=1152, help='Hidden size of the DiT model.')
    parser.add_argument('--depth', type=int, default=28, help='Depth of the DiT model.')
    parser.add_argument('--num-heads', type=int, default=16, help='Number of attention heads.')
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of classes for conditioning.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # --- Dataset and DataLoader ---
    # Using the actual dataset loader now
    dataset = ImageNetDataset(root_dir=args.data_path, split='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(f"Loaded dataset from {args.data_path}. Found {len(dataset)} images.")

    # --- Load Tokenizer ---
    # The tokenizer architecture needs to be known. This is a simplification.
    tokenizer = MAETokenizer(
        img_size=256, patch_size=16, embed_dim=768, # These should match the saved model
        encoder_depth=12, encoder_heads=12,
        decoder_depth=8, decoder_heads=16,
        mlp_ratio=4.0,
        use_auxiliary_decoders=True, # Changed to True
        hidden_token_length=16 # Pass latent dim here
    )
    print(f"Loading tokenizer from {args.tokenizer_path}")
    # The fine-tuned checkpoint saves the whole model, not just the state_dict
    # We need to load the state_dict from the pre-trained model instead.
    pretrained_tokenizer_path = args.tokenizer_path.replace('maetok_finetuned/finetuned_decoder_epoch_0.pth', 'maetok_pretrained/mae_best_model.pth')
    tokenizer_checkpoint = torch.load(pretrained_tokenizer_path)
    tokenizer.load_state_dict(tokenizer_checkpoint['model_state_dict'])
    tokenizer.to(device)
    tokenizer.eval()
    print("Tokenizer loaded successfully.")

    # --- Diffusion Model ---
    model = LightningDiT(
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        num_classes=args.num_classes
    ).to(device)
    print(f"Created LightningDiT model with {sum(p.numel() for p in model.parameters())} parameters.")

    # --- Training Setup ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    alphas_cumprod = get_alpha_schedule().to(device)

    # --- Training Loop ---
    print("Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # 1. Get latents from tokenizer
            with torch.no_grad():
                _, latents, _, _ = tokenizer(images, mask_ratio=0.0, extract_aux_features=False)

            # 2. Create noisy latents (forward diffusion)
            t = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = q_sample(latents, t, alphas_cumprod, noise)

            # 3. Get model prediction
            predicted_noise = model(noisy_latents, t, labels)

            # 4. Calculate loss
            loss = loss_fn(predicted_noise, noise)

            # 5. Backpropagate
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 1 == 0: # Print every batch for this small dataset
                print(f"Epoch {epoch}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")

        # --- Save Checkpoint ---
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.checkpoint_dir, f'lightningdit_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("Training finished.")

if __name__ == '__main__':
    main()
