import sys
import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.backbone import ResNetBackbone
from src.model.slot_attention import PartAwareSlotAttention
from src.model.part_autoencoder import PartAutoEncoder, SlotDecoder
from src.data.cub_dataset import CUB200Dataset, get_cub_transforms
from src.utils.visualize import visualize_slots

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data
    print("Loading CUB-200 Dataset...")
    transform = get_cub_transforms()
    # Assuming CUB dataset is at args.data_root/CUB_200_2011
    dataset_root = os.path.join(args.data_root, 'CUB_200_2011')
    
    try:
        train_dataset = CUB200Dataset(dataset_root, train=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        print(f"Dataset loaded. {len(train_dataset)} training images.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure you have downloaded and extracted CUB_200_2011.tgz")
        return

    # 2. Model
    backbone = ResNetBackbone(pretrained=True, return_layer='layer3')
    slot_attn = PartAwareSlotAttention(
        num_slots=args.num_slots,
        in_channels=256, # ResNet layer3 output
        slot_dim=args.slot_dim,
        iters=3
    )
    decoder = SlotDecoder(
        slot_dim=args.slot_dim,
        resolution=(14, 14),
        upsample_steps=4
    )
    
    model = PartAutoEncoder(backbone, slot_attn, decoder).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, images in enumerate(pbar):
            images = images.to(device)
            
            # Forward
            recon, slots, attn, alpha = model(images)
            
            # Loss: MSE Reconstruction
            loss = F.mse_loss(recon, images)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Visualize first batch
            if i == 0:
                vis_path = os.path.join(args.output_dir, f"epoch_{epoch+1}_vis.png")
                visualize_slots(images[0], recon[0], attn[0], save_path=vis_path)
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data', help='Path to folder containing CUB_200_2011')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Where to save logs and models')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_slots', type=int, default=5)
    parser.add_argument('--slot_dim', type=int, default=64)
    
    args = parser.parse_args()
    train(args)
