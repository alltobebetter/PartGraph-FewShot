import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn.functional as F
from src.model.backbone import ResNetBackbone
from src.model.slot_attention import PartAwareSlotAttention
from src.model.part_autoencoder import PartAutoEncoder, SlotDecoder

def main():
    print("Initializing MVP Model...")
    
    # 1. Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    resolution = (14, 14) # Feature map size
    slot_dim = 64 # Keep it small for MVP
    num_slots = 5
    
    # 2. Build Model
    backbone = ResNetBackbone(pretrained=False, return_layer='layer3') # Output 14x14
    # Note: ResNet layer3 output channels is 256
    
    slot_attn = PartAwareSlotAttention(
        num_slots=num_slots,
        in_channels=256,
        slot_dim=slot_dim,
        iters=3,
        resolution=resolution
    )
    
    decoder = SlotDecoder(
        slot_dim=slot_dim,
        resolution=resolution,
        upsample_steps=4 # 14 -> 224
    )
    
    model = PartAutoEncoder(backbone, slot_attn, decoder).to(device)
    print("Model built successfully.")
    
    # 3. Dummy Data
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"Input shape: {dummy_img.shape}")
    
    # 4. Forward Pass Check
    print("Running forward pass...")
    recon, slots, attn, alpha = model(dummy_img)
    
    print(f"Reconstruction shape: {recon.shape}") # Should be (B, 3, 224, 224)
    print(f"Slots shape: {slots.shape}") # (B, K, D)
    print(f"Attention shape: {attn.shape}") # (B, K, 14, 14)
    
    # 5. Backward Pass Check
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss = F.mse_loss(recon, dummy_img)
    print(f"Initial Loss: {loss.item()}")
    
    loss.backward()
    optimizer.step()
    print("Backward pass successful.")
    
    print("\nMVP Test Passed! The architecture is valid.")
    print("Next steps: Load CUB-200 dataset and train for real.")

if __name__ == "__main__":
    main()
