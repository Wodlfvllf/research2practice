import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *
from .rope_embeddings import *
from .vit_encoder import Encoder
from .vit_decoder import Decoder
from .aux_decoder import AuxiliaryDecoder, FeatureExtractors

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
                use_auxiliary_decoders=True,
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
        self.use_auxiliary_decoders = use_auxiliary_decoders
        
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
        
        # Auxiliary decoders with same architecture as your main decoder
        if use_auxiliary_decoders:
            self.hog_decoder = AuxiliaryDecoder(
                embed_dim=embed_dim,
                output_dim=576,  # Updated HOG feature dimension for 4x4 patches
                decoder_depth=decoder_depth,
                decoder_heads=decoder_heads,
                hidden_token_length=hidden_token_length,
                mlp_ratio=mlp_ratio,
                img_size=img_size,
                patch_size=patch_size
            )
            
            self.clip_decoder = AuxiliaryDecoder(
                embed_dim=embed_dim,
                output_dim=1024,  # CLIP feature dimension
                decoder_depth=decoder_depth,
                decoder_heads=decoder_heads,
                hidden_token_length=hidden_token_length,
                mlp_ratio=mlp_ratio,
                img_size=img_size,
                patch_size=patch_size
            )
            
            self.feature_extractors = FeatureExtractors()
            
    def patchify(self, imgs):
        """Convert images to patches"""
        B, C, H, W = imgs.shape
        assert H == self.img_size and W == self.img_size
        
        x = self.encoder.patch_embed(imgs)  # B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x
    
    def unpatchify(self, x):
        """Convert patches to images"""
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_channels, h * p, h * p))
        return imgs
    
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
    
    def forward(self, x, mask_ratio=0.75, extract_aux_features=True):
        # x: (B, C, H, W)
        # Get patch embeddings
        patches = self.encoder.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Forward through encoder blocks
        height = width = self.img_size // self.patch_size
        for block in self.encoder.blocks:
            patches = block(patches, height=height, width=width)
        
        # Mask patches
        patches_masked, mask, ids_restore = self.mask_patches(patches, mask_ratio)
        encoder_input = patches_masked
        
        # Decode
        reconstruction = self.decoder(encoder_input, ids_restore)

        # Auxiliary predictions
        aux_outputs = {}
        if self.use_auxiliary_decoders and extract_aux_features:
            # Extract target features
            with torch.no_grad():
                hog_targets = self.feature_extractors.extract_hog_features(x, self.patch_size)
                clip_targets = self.feature_extractors.extract_clip_features(x, self.patch_size)
            
            # Predict auxiliary features using same encoder output
            hog_pred = self.hog_decoder(encoder_input, ids_restore)
            clip_pred = self.clip_decoder(encoder_input, ids_restore)
            
            aux_outputs = {
                'hog_pred': hog_pred,
                'clip_pred': clip_pred,
                'hog_targets': hog_targets,
                'clip_targets': clip_targets,
            }
        
        return reconstruction, mask, aux_outputs