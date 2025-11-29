import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.pos_embed import build_position_encoding

class SlotDecoder(nn.Module):
    def __init__(self, slot_dim, out_channels=3, resolution=(14, 14), upsample_steps=4):
        """
        Simple decoder that upsamples slots back to image.
        upsample_steps: how many times to x2 upsample. 
        If feature map is 14x14 and image is 224x224, we need 16x upsample (4 steps).
        """
        super().__init__()
        self.resolution = resolution
        self.slot_dim = slot_dim
        
        # Positional encoding for the decoder grid
        self.register_buffer(
            "pos_embed", 
            build_position_encoding(resolution[0], slot_dim).unsqueeze(0) # (1, H*W, D)
        )
        
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(slot_dim, slot_dim, 5, stride=2, padding=2, output_padding=1), # 14->28
            nn.ReLU(),
            nn.ConvTranspose2d(slot_dim, slot_dim, 5, stride=2, padding=2, output_padding=1), # 28->56
            nn.ReLU(),
            nn.ConvTranspose2d(slot_dim, slot_dim, 5, stride=2, padding=2, output_padding=1), # 56->112
            nn.ReLU(),
            nn.ConvTranspose2d(slot_dim, slot_dim, 5, stride=2, padding=2, output_padding=1), # 112->224
            nn.ReLU(),
            nn.Conv2d(slot_dim, out_channels + 1, 3, stride=1, padding=1) # RGB + Alpha
        )

    def forward(self, slots):
        """
        slots: (B, K, D)
        """
        b, k, d = slots.shape
        h, w = self.resolution
        
        # Spatial Broadcast
        # (B, K, D) -> (B*K, D, 1, 1) -> (B*K, D, H, W)
        slots = slots.reshape(b * k, d, 1, 1).expand(-1, -1, h, w)
        
        # Add Pos Embed
        # pos_embed: (1, H*W, D) -> (1, D, H, W)
        pos = self.pos_embed.permute(0, 2, 1).view(1, d, h, w)
        slots = slots + pos
        
        # Decode
        out = self.conv_layers(slots) # (B*K, 4, H_out, W_out)
        
        # Split RGB and Alpha
        out = out.view(b, k, 4, out.shape[2], out.shape[3])
        rgb = out[:, :, :3, :, :]
        alpha = out[:, :, 3:, :, :]
        
        return rgb, alpha

class PartAutoEncoder(nn.Module):
    def __init__(self, backbone, slot_attn, decoder):
        super().__init__()
        self.backbone = backbone
        self.slot_attn = slot_attn
        self.decoder = decoder
        
    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
        # 1. Extract features
        features = self.backbone(x) # (B, C, 14, 14)
        
        # 2. Slot Attention
        slots, attn_masks = self.slot_attn(features) # slots: (B, K, D)
        
        # 3. Decode
        # rgb: (B, K, 3, H_im, W_im), alpha: (B, K, 1, H_im, W_im)
        rgb, alpha = self.decoder(slots)
        
        # 4. Recombine
        alpha = F.softmax(alpha, dim=1)
        recon = (rgb * alpha).sum(dim=1) # (B, 3, H_im, W_im)
        
        return recon, slots, attn_masks, alpha
