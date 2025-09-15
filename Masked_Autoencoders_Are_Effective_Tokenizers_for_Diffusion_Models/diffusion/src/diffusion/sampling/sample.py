import torch
import argparse
from diffusion.src.diffusion.models.sit_model import SiT
from tokenizer.src.tokenizer.models.vit_main import Model as MAETokenizer
from torchvision.utils import save_image
import os

@torch.no_grad()
def p_sample_loop(model, x, t, y):
    # Reverse diffusion process for one step
    model.eval()
    t_in = t.unsqueeze(0)
    pred_noise = model(x, t_in, y)
    
    # These would be based on a proper noise schedule
    alpha = 1.0 - 0.02 * (t / 1000.0)
    alpha_bar = alpha
    sigma = (1 - alpha).sqrt()
    
    x = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - alpha_bar).sqrt()) * pred_noise) + sigma * torch.randn_like(x)
    return x

def main():
    parser = argparse.ArgumentParser(description='Sample images from a trained SiT model.')
    parser.add_argument('--model-path', required=True, help='Path to the trained SiT model checkpoint.')
    parser.add_argument('--tokenizer-path', required=True, help='Path to the pre-trained MAE tokenizer checkpoint.')
    parser.add_argument('--num-samples', type=int, default=4, help='Number of images to generate.')
    parser.add_argument('--class-id', type=int, default=0, help='Class ID to generate.')
    parser.add_argument('--output-dir', default='generated_images', help='Directory to save the generated images.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = MAETokenizer()
    tokenizer.load_state_dict(torch.load(args.tokenizer_path)['model_state_dict'])
    tokenizer.eval().to(device)

    # Load model
    model = SiT(tokenizer=tokenizer)
    model.load_state_dict(torch.load(args.model_path))
    model.eval().to(device)

    # Sampling loop
    x = torch.randn(args.num_samples, 3, 256, 256, device=device)
    y = torch.full((args.num_samples,), args.class_id, device=device, dtype=torch.long)

    for i in range(999, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        x = p_sample_loop(model, x, t, y)
        if i % 100 == 0:
            print(f'Sampling step {i}')

    # Decode the final image
    x = tokenizer.decode(x)

    # Save images
    for i in range(args.num_samples):
        save_image(x[i], os.path.join(args.output_dir, f'sample_{i}_class_{args.class_id}.png'), normalize=True)

    print(f'Generated {args.num_samples} images in {args.output_dir}')

if __name__ == '__main__':
    main()
