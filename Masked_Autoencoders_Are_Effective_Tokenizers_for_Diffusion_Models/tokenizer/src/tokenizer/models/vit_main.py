import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *
from .rope_embeddings import *
from .vit_encoder import Encoder
from .vit_decoder import Decoder

class Model(nn.Module):
    def __init__(self, 
                img_size=64, 
                patch_size=4,
                embed_dim=128,
                encoder_depth=4,
                decoder_depth=4,
                hidden_token_length=16,
                mlp_ratio=4.0,
                encoder_heads=4,
                decoder_heads=4,
                in_channels=3,
                ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.hidden_token_length = hidden_token_length
        self.mlp_ratio = mlp_ratio
        self.encoder_heads = encoder_heads
        self.decoder_heads = decoder_heads
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        
        # Encoder
        self.encoder = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            encoder_depth=encoder_depth,
            hidden_token_length=hidden_token_length,
            encoder_heads=encoder_heads,
            in_channels=in_channels,
            mlp_ratio=mlp_ratio
        )
        
        # Decoder
        self.decoder = Decoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            decoder_depth=decoder_depth,
            hidden_token_length=hidden_token_length,
            decoder_heads=decoder_heads,
            in_channels=in_channels,
            mlp_ratio=mlp_ratio
        )
        
    def mask_patches(self, x, mask_ratio):
        """
        Mask patches for MAE training.
        x: (B, num_patches, embed_dim)
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def forward(self, x, mask_ratio=0.75):
        # x: (B, C, H, W)
        # Get patch embeddings
        patches = self.encoder.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Mask patches
        patches_masked, mask, ids_restore = self.mask_patches(patches, mask_ratio)
        
        # Add latent tokens and encode
        bs = x.shape[0]
        latent_tokens = self.encoder.latent_tokens.expand(bs, -1, -1)
        encoder_input = torch.cat([latent_tokens, patches_masked], dim=1)
        
        # Forward through encoder blocks
        height = width = self.img_size // self.patch_size
        for block in self.encoder.blocks:
            encoder_input = block(encoder_input, height=height, width=width)
        encoder_input = self.encoder.norm_layer(encoder_input)
        
        # Decode
        reconstruction = self.decoder(encoder_input, ids_restore)
        
        return reconstruction, mask