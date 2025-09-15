import torch
import argparse
from diffusion.src.diffusion.sampling.sample import p_sample_loop
from diffusion.src.diffusion.models.sit_model import SiT
from tokenizer.src.tokenizer.models.vit_main import Model as MAETokenizer
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

def generate_images_for_evaluation(model, tokenizer, num_images, class_id, device):
    model.eval()
    images = []
    batch_size = 16 # Adjust based on GPU memory
    num_batches = (num_images + batch_size - 1) // batch_size

    for _ in tqdm(range(num_batches), desc="Generating images"):
        x = torch.randn(batch_size, 3, 256, 256, device=device)
        y = torch.full((batch_size,), class_id, device=device, dtype=torch.long)

        for i in range(999, -1, -1):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            x = p_sample_loop(model, x, t, y)
        
        img = tokenizer.decode(x)
        images.append((img.cpu() * 255).to(torch.uint8))

    return torch.cat(images, dim=0)[:num_images]

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained SiT model.')
    parser.add_argument('--model-path', required=True, help='Path to the trained SiT model checkpoint.')
    parser.add_argument('--tokenizer-path', required=True, help='Path to the pre-trained MAE tokenizer checkpoint.')
    parser.add_argument('--real-images-path', required=True, help='Path to the real ImageNet validation set.')
    parser.add_argument('--num-images', type=int, default=1000, help='Number of images to generate for evaluation.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    tokenizer = MAETokenizer()
    tokenizer.load_state_dict(torch.load(args.tokenizer_path)['model_state_dict'])
    tokenizer.eval().to(device)

    model = SiT(tokenizer=tokenizer)
    model.load_state_dict(torch.load(args.model_path))
    model.eval().to(device)

    # Metrics
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    # Generate fake images
    fake_images = generate_images_for_evaluation(model, tokenizer, args.num_images, 0, device)
    
    # Load real images
    dataset = ImageFolder(args.real-images-path, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=32)
    real_images = []
    for img, _ in dataloader:
        real_images.append((img * 255).to(torch.uint8))
        if len(real_images) * 32 >= args.num_images:
            break
    real_images = torch.cat(real_images, dim=0)[:args.num_images]

    # Update metrics
    fid.update(real_images.to(device), real=True)
    fid.update(fake_images.to(device), real=False)
    inception.update(fake_images.to(device))

    # Compute and print results
    print(f'FID: {fid.compute()}')
    mean, std = inception.compute()
    print(f'Inception Score: {mean} ± {std}')

if __name__ == '__main__':
    main()
