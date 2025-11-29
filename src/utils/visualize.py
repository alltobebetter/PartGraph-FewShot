import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from src.data.cub_dataset import get_cub_inverse_transform

def visualize_slots(image, recon, attn_masks, save_path=None):
    """
    image: (3, H, W) normalized tensor
    recon: (3, H, W) normalized tensor
    attn_masks: (K, H_feat, W_feat) tensor, range [0, 1]
    """
    inv_transform = get_cub_inverse_transform()
    
    # Un-normalize images
    image = inv_transform(image).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    recon = inv_transform(recon).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    
    # Process attention masks
    K = attn_masks.shape[0]
    attn_masks = attn_masks.detach().cpu().numpy() # (K, H, W)
    
    # Create figure
    # Layout: Original | Reconstruction | Slot 1 | Slot 2 | ... | Slot K
    fig, axes = plt.subplots(1, 2 + K, figsize=(4 * (2 + K), 4))
    
    # 1. Original
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # 2. Reconstruction
    axes[1].imshow(recon)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    
    # 3. Slots
    for k in range(K):
        ax = axes[2 + k]
        # Upsample attention to image size for better visualization
        # Simple resize using imshow extent or interpolation
        ax.imshow(image, alpha=0.5) # Show faint original
        ax.imshow(attn_masks[k], cmap='jet', alpha=0.5, extent=[0, image.shape[1], image.shape[0], 0])
        ax.set_title(f"Slot {k}")
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
