"""
改进版训练脚本 - 多任务学习 (重建 + 分类)
核心改动：
1. 增加 slot 容量 (5→8, 64→256)
2. 加入分类损失，让 slot 学习判别性特征
3. 加入多样性损失，防止 slot collapse
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.backbone import ResNetBackbone
from src.model.slot_attention import PartAwareSlotAttention
from src.model.part_autoencoder import PartAutoEncoder, SlotDecoder
from src.data.cub_dataset import CUB200Dataset, get_cub_transforms
from src.utils.visualize import visualize_slots


class CUB200WithLabels(CUB200Dataset):
    """扩展数据集，返回图像和类别标签"""
    def __init__(self, root_dir, train=True, transform=None):
        super().__init__(root_dir, train, transform)
        # 从路径提取类别 ID (001.xxx -> 0)
        self.labels = [int(p.split('.')[0]) - 1 for p in self.image_paths]
    
    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        label = self.labels[idx]
        return image, label


class SlotClassifier(nn.Module):
    """基于 slot 特征的分类头"""
    def __init__(self, slot_dim, num_slots, num_classes):
        super().__init__()
        # 方法1: 平均所有 slot 后分类
        self.fc = nn.Linear(slot_dim, num_classes)
        # 方法2: 注意力加权
        self.attn_weight = nn.Linear(slot_dim, 1)
        
    def forward(self, slots):
        """
        slots: (B, K, D)
        """
        # 注意力加权聚合
        weights = F.softmax(self.attn_weight(slots), dim=1)  # (B, K, 1)
        pooled = (slots * weights).sum(dim=1)  # (B, D)
        return self.fc(pooled)


def diversity_loss(attn):
    """
    鼓励不同 slot 关注不同区域
    attn: (B, K, H, W)
    """
    B, K, H, W = attn.shape
    attn_flat = attn.view(B, K, -1)  # (B, K, H*W)
    
    # 计算 slot 之间的重叠 (余弦相似度)
    attn_norm = F.normalize(attn_flat, dim=-1)
    similarity = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # (B, K, K)
    
    # 去掉对角线 (自己和自己)
    mask = 1 - torch.eye(K, device=attn.device).unsqueeze(0)
    overlap = (similarity * mask).sum() / (B * K * (K - 1))
    
    return overlap


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: num_slots={args.num_slots}, slot_dim={args.slot_dim}")
    
    # 1. Data
    transform = get_cub_transforms()
    dataset_root = os.path.join(args.data_root, 'CUB_200_2011')
    
    try:
        train_dataset = CUB200WithLabels(dataset_root, train=True, transform=transform)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=2,
            drop_last=True
        )
        print(f"Dataset loaded. {len(train_dataset)} training images, 200 classes.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Model - 增加容量
    backbone = ResNetBackbone(pretrained=True, return_layer='layer3')
    
    slot_attn = PartAwareSlotAttention(
        num_slots=args.num_slots,      # 8 (原来5)
        in_channels=256,
        slot_dim=args.slot_dim,        # 256 (原来64)
        iters=args.slot_iters          # 3
    )
    
    decoder = SlotDecoder(
        slot_dim=args.slot_dim,
        resolution=(14, 14),
        upsample_steps=4
    )
    
    model = PartAutoEncoder(backbone, slot_attn, decoder).to(device)
    classifier = SlotClassifier(args.slot_dim, args.num_slots, 200).to(device)
    
    # 3. Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. Training Loop
    os.makedirs(args.output_dir, exist_ok=True)
    
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        classifier.train()
        
        total_loss = 0
        total_recon = 0
        total_cls = 0
        total_div = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            recon, slots, attn, alpha = model(images)
            logits = classifier(slots)
            
            # Losses
            loss_recon = F.mse_loss(recon, images)
            loss_cls = F.cross_entropy(logits, labels)
            loss_div = diversity_loss(attn)
            
            # 总损失 - 分类为主，重建为辅
            loss = (
                args.lambda_cls * loss_cls + 
                args.lambda_recon * loss_recon + 
                args.lambda_div * loss_div
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_cls += loss_cls.item()
            total_div += loss_div.item()
            
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'cls': f'{loss_cls.item():.3f}',
                'acc': f'{100.*correct/total:.1f}%'
            })
            
            # Visualize
            if i == 0:
                vis_path = os.path.join(args.output_dir, f"epoch_{epoch+1}_vis.png")
                visualize_slots(images[0], recon[0], attn[0], save_path=vis_path)
        
        scheduler.step()
        
        # Epoch stats
        n = len(train_loader)
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/n:.4f}, "
              f"Recon={total_recon/n:.4f}, Cls={total_cls/n:.4f}, "
              f"Div={total_div/n:.4f}, Acc={acc:.2f}%")
        
        # Save best
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'epoch': epoch,
                'acc': acc
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Regular checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
            }, os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))
    
    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_improved')
    parser.add_argument('--batch_size', type=int, default=16)  # 减小以节省显存
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # 关键改动
    parser.add_argument('--num_slots', type=int, default=8)      # 原来 5
    parser.add_argument('--slot_dim', type=int, default=256)     # 原来 64
    parser.add_argument('--slot_iters', type=int, default=3)
    
    # 损失权重
    parser.add_argument('--lambda_cls', type=float, default=1.0)    # 分类损失
    parser.add_argument('--lambda_recon', type=float, default=0.1)  # 重建损失 (降低权重)
    parser.add_argument('--lambda_div', type=float, default=0.01)   # 多样性损失
    
    args = parser.parse_args()
    train(args)
