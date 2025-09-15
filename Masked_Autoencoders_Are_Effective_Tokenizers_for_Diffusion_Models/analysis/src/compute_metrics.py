import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.fid import FrechetInceptionDistance

def compute_psnr(img1, img2):
    """Computes the PSNR between two images."""
    return psnr(img1, img2, data_range=1.0)

def compute_ssim(img1, img2):
    """Computes the SSIM between two images."""
    return ssim(img1, img2, data_range=1.0, multichannel=True, channel_axis=-1)

def compute_fid(real_images, fake_images, device):
    """Computes the FID between two sets of images."""
    fid = FrechetInceptionDistance(normalize=True).to(device)
    # torchmetrics expects images to be in range [0, 255] and of type uint8
    real_images = (real_images * 255).to(torch.uint8)
    fake_images = (fake_images * 255).to(torch.uint8)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()
