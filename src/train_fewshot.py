"""
真正的 Few-Shot 训练脚本

关键改动：
- 只在 base classes (1-100) 上训练
- 测试在 novel classes (101-200) 上进行
- 这样才是真正的 few-shot 评估
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.backbone import ResNetBackbone
from src.model.gnn_slot_attention import PartAwareGNNSlotAttention
from src.model.slot_attention import PartAwareSlotAttention
from src.model.part_autoencoder import SlotDecoder
from src.data.cub_dataset import CUB200FewShot, get_cub_transforms
from src.utils.visualize import visualize_slots


class PartAutoEncoder(nn.Module):
    """Part-based AutoEncoder"""
    def __init__(self, backbone, slot_attn, decoder):
        super().__init__()
        self.backbone = backbone
        self.slot_attn = slot_attn
        self.decoder = decoder
        
    def forward(self, x):
        features = self.backbone(x)
        slots, attn_masks = self.slot_attn(features)
        rgb, alpha = self.decoder(slots)
        alpha = F.softmax(alpha, dim=1)
        recon = (rgb * alpha).sum(dim=1)
        return recon, slots, attn_masks, alpha


class SlotClassifier(nn.Module):
    """基于 slot 特征的分类头"""
    def __init__(self, slot_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(slot_dim, num_classes)
        self.attn_weight = nn.Linear(slot_dim, 1)
        
    def forward(self, slots):
        # 注意力加权聚合
        weights = F.softmax(self.attn_weight(slots), dim=1)
        pooled = (slots * weights).sum(dim=1)
        return self.fc(pooled)


def diversity_loss(attn):
    """鼓励不同 slot 关注不同区域"""
    B, K, H, W = attn.shape
    attn_flat = attn.view(B, K, -1)
    attn_norm = F.normalize(attn_flat, dim=-1)
    similarity = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
    mask = 1 - torch.eye(K, device=attn.device).unsqueeze(0)
    overlap = (similarity * mask).sum() / (B * K * (K - 1))
    return overlap


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"=== Few-Shot Training (Base Classes Only) ===")
    print(f"num_slots={args.num_slots}, slot_dim={args.slot_dim}, use_gnn={args.use_gnn}")
    
    # 1. Data - 只用 base classes (1-100)
    transform_train = get_cub_transforms(augment=True)
    dataset_root = os.path.join(args.data_root, 'CUB_200_2011')
    
    try:
        train_dataset = CUB200FewShot(
            dataset_root, 
            split='base',  # 只用前 100 类！
            transform=transform_train,
            use_all_images=True  # 用该类的所有图片
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=2,
            drop_last=True
        )
        num_classes = train_dataset.num_classes
        print(f"Training on {num_classes} base classes, {len(train_dataset)} images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Model
    backbone = ResNetBackbone(pretrained=True, return_layer='layer3')
    
    if args.use_gnn:
        slot_attn = PartAwareGNNSlotAttention(
            num_slots=args.num_slots,
            in_channels=256,
            slot_dim=args.slot_dim,
            iters=args.slot_iters,
            gnn_start_iter=args.gnn_start_iter
        )
    else:
        slot_attn = PartAwareSlotAttention(
            num_slots=args.num_slots,
            in_channels=256,
            slot_dim=args.slot_dim,
            iters=args.slot_iters
        )
    
    decoder = SlotDecoder(
        slot_dim=args.slot_dim,
        resolution=(14, 14),
        upsample_steps=4
    )
    
    model = PartAutoEncoder(backbone, slot_attn, decoder).to(device)
    classifier = SlotClassifier(args.slot_dim, num_classes).to(device)
    
    # 3. Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. Training
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
                'acc': acc,
                'args': vars(args)
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        # Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model': model.state_dict(),
                'classifier': classifier.state_dict(),
                'args': vars(args)
            }, os.path.join(args.output_dir, f'model_epoch_{epoch+1}.pth'))
    
    print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
    print(f"Note: This accuracy is on BASE classes. Run eval_fewshot_novel.py for true few-shot evaluation on NOVEL classes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_fewshot')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Model
    parser.add_argument('--use_gnn', action='store_true', help='Use GNN-in-the-Loop')
    parser.add_argument('--num_slots', type=int, default=8)
    parser.add_argument('--slot_dim', type=int, default=256)
    parser.add_argument('--slot_iters', type=int, default=3)
    parser.add_argument('--gnn_start_iter', type=int, default=1)
    
    # Loss weights
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_recon', type=float, default=0.1)
    parser.add_argument('--lambda_div', type=float, default=0.01)
    
    args = parser.parse_args()
    train(args)
