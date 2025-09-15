import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusion.src.diffusion.models.sit_model import SiT
from tokenizer.src.tokenizer.models.vit_main import Model as MAETokenizer
import argparse
import os

class DiffusionTrainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer.to(device)
        self.device = device
        self.loss_fn = nn.MSELoss()

    def train_one_epoch(self, dataloader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)

        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()

            t = torch.randint(0, 1000, (images.shape[0],), device=self.device).long()
            noise = torch.randn_like(images)
            x_t = self.q_sample(images, t, noise)
            
            predicted_noise = self.model(x_t, t, labels)
            
            loss = self.loss_fn(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        return avg_loss

    def q_sample(self, x_0, t, noise):
        # Forward diffusion process
        # These values would typically be pre-computed
        alpha_t = self.get_alpha(t)
        alpha_bar_t = self.get_alpha_bar(alpha_t)
        
        mean = alpha_bar_t.sqrt().view(-1, 1, 1, 1) * x_0
        std_dev = (1 - alpha_bar_t).sqrt().view(-1, 1, 1, 1)
        
        return mean + std_dev * noise

    def get_alpha(self, t):
        # Dummy alpha schedule
        return 1.0 - 0.02 * (t / 1000.0)

    def get_alpha_bar(self, alpha):
        # Dummy alpha_bar calculation
        return alpha 

def main():
    parser = argparse.ArgumentParser(description='Train SiT model.')
    parser.add_argument('--data-path', required=True, help='Path to ImageNet dataset')
    parser.add_argument('--tokenizer-path', required=True, help='Path to pre-trained MAE tokenizer checkpoint.')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint-dir', default='diffusion/checkpoints/sit_model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = MAETokenizer()
    tokenizer.load_state_dict(torch.load(args.tokenizer_path)['model_state_dict'])
    tokenizer.eval()

    # Create model
    model = SiT(tokenizer=tokenizer)

    # Create trainer
    trainer = DiffusionTrainer(model, tokenizer, device=device)

    # Create data loader (using a dummy dataset for now)
    # Replace with your actual ImageNet dataset
    from torch.utils.data import TensorDataset
    dummy_images = torch.randn(128, 3, 256, 256)
    dummy_labels = torch.randint(0, 1000, (128,))
    dataset = TensorDataset(dummy_images, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        trainer.train_one_epoch(dataloader, optimizer, epoch)
        if epoch % 10 == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'sit_epoch_{epoch}.pth'))

if __name__ == '__main__':
    main()
