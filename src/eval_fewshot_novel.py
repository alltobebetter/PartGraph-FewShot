"""
真正的 Few-Shot 评估脚本

关键：在 NOVEL classes (101-200) 上测试
这些类别在训练时从未见过！
"""
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.backbone import ResNetBackbone
from src.model.gnn_slot_attention import PartAwareGNNSlotAttention
from src.model.slot_attention import PartAwareSlotAttention
from src.model.part_autoencoder import SlotDecoder
from src.data.cub_dataset import CUB200FewShot, get_cub_transforms


class PartAutoEncoder(nn.Module):
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


def extract_features(model, images, method='slot'):
    """提取特征"""
    with torch.no_grad():
        if method == 'backbone':
            features = model.backbone(images)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        else:
            _, slots, _, _ = model(images)
            features = slots.mean(dim=1)  # 平均所有 slot
    return features


def proto_classify(support_features, support_labels, query_features):
    """
    Prototypical Network 分类
    
    Args:
        support_features: (N*K, D) - N-way K-shot 的 support 特征
        support_labels: (N*K,) - support 标签 (0 到 N-1)
        query_features: (Q, D) - query 特征
    
    Returns:
        predictions: (Q,) - 预测标签
    """
    # 计算每个类的原型
    unique_labels = torch.unique(support_labels)
    prototypes = []
    
    for label in unique_labels:
        mask = support_labels == label
        class_features = support_features[mask]
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)  # (N, D)
    
    # 计算距离并分类
    dists = torch.cdist(query_features, prototypes)  # (Q, N)
    predictions = dists.argmin(dim=1)
    
    return predictions


def run_episode(model, dataset, n_way=5, k_shot=5, n_query=15, method='slot', device='cuda'):
    """
    运行一个 few-shot episode
    
    从 dataset 中随机选 n_way 个类，每类 k_shot 个 support + n_query 个 query
    """
    # 获取所有类别
    all_labels = list(set(dataset.labels))
    
    if len(all_labels) < n_way:
        raise ValueError(f"Dataset has only {len(all_labels)} classes, but n_way={n_way}")
    
    # 随机选 n_way 个类
    selected_classes = random.sample(all_labels, n_way)
    
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    
    for new_label, orig_label in enumerate(selected_classes):
        # 找到该类的所有样本
        indices = [i for i, l in enumerate(dataset.labels) if l == orig_label]
        
        # 采样
        if len(indices) < k_shot + n_query:
            # 样本不够，重复采样
            selected = random.choices(indices, k=k_shot + n_query)
        else:
            selected = random.sample(indices, k_shot + n_query)
        
        support_idx = selected[:k_shot]
        query_idx = selected[k_shot:k_shot + n_query]
        
        for idx in support_idx:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(new_label)
        
        for idx in query_idx:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(new_label)
    
    # 转为 tensor
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_images = torch.stack(query_images).to(device)
    query_labels = torch.tensor(query_labels).to(device)
    
    # 提取特征
    support_features = extract_features(model, support_images, method)
    query_features = extract_features(model, query_images, method)
    
    # 分类
    predictions = proto_classify(support_features, support_labels, query_features)
    
    # 计算准确率
    correct = (predictions == query_labels).sum().item()
    total = query_labels.size(0)
    
    return correct, total


def compute_slot_metrics(model, dataset, device, n_samples=100):
    """计算 slot 质量指标"""
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    overlaps = []
    coverages = []
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            img, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)
            
            _, _, attn, _ = model(img)
            
            # Overlap: slot 之间的重叠
            B, K, H, W = attn.shape
            attn_flat = attn.view(B, K, -1)
            attn_norm = F.normalize(attn_flat, dim=-1)
            similarity = torch.bmm(attn_norm, attn_norm.transpose(1, 2))
            mask = 1 - torch.eye(K, device=device).unsqueeze(0)
            overlap = (similarity * mask).sum() / (K * (K - 1))
            overlaps.append(overlap.item())
            
            # Coverage: 图像被覆盖的比例
            max_attn, _ = attn.max(dim=1)
            covered = (max_attn > 0.1).float().mean()
            coverages.append(covered.item())
    
    return {
        'overlap': np.mean(overlaps),
        'coverage': np.mean(coverages)
    }


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载 NOVEL classes 数据集
    transform = get_cub_transforms(augment=False)
    dataset_root = os.path.join(args.data_root, 'CUB_200_2011')
    
    novel_dataset = CUB200FewShot(
        dataset_root, 
        split='novel',  # 类别 101-200，训练时从未见过！
        transform=transform,
        use_all_images=True
    )
    print(f"\n=== Evaluating on NOVEL classes (never seen during training!) ===")
    print(f"Novel dataset: {novel_dataset.num_classes} classes, {len(novel_dataset)} images")
    
    # 2. 构建模型
    backbone = ResNetBackbone(pretrained=True, return_layer='layer3')
    
    if args.use_gnn:
        print("Model: GNN-in-the-Loop Slot Attention")
        slot_attn = PartAwareGNNSlotAttention(
            num_slots=args.num_slots,
            in_channels=256,
            slot_dim=args.slot_dim,
            iters=args.slot_iters,
            gnn_start_iter=args.gnn_start_iter
        )
    else:
        print("Model: Baseline Slot Attention")
        slot_attn = PartAwareSlotAttention(
            num_slots=args.num_slots,
            in_channels=256,
            slot_dim=args.slot_dim,
            iters=args.slot_iters
        )
    
    decoder = SlotDecoder(slot_dim=args.slot_dim, resolution=(14, 14))
    model = PartAutoEncoder(backbone, slot_attn, decoder).to(device)
    
    # 3. 加载 checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
    else:
        print("WARNING: No checkpoint loaded! Using random/pretrained weights.")
    
    model.eval()
    
    # 4. Slot 质量指标
    print("\n--- Slot Quality Metrics (on novel classes) ---")
    metrics = compute_slot_metrics(model, novel_dataset, device)
    print(f"  Overlap:  {metrics['overlap']:.4f} (lower is better)")
    print(f"  Coverage: {metrics['coverage']:.4f} (higher is better)")
    
    # 5. Few-shot 评估
    print(f"\n--- {args.n_way}-way {args.k_shot}-shot Evaluation ---")
    print(f"Running {args.n_episodes} episodes...")
    
    results = {'slot': [], 'backbone': []}
    
    for method in ['backbone', 'slot']:
        correct_total = 0
        sample_total = 0
        
        for _ in tqdm(range(args.n_episodes), desc=f"{method}"):
            correct, total = run_episode(
                model, novel_dataset,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_query=args.n_query,
                method=method,
                device=device
            )
            correct_total += correct
            sample_total += total
            results[method].append(correct / total)
        
        acc = 100. * correct_total / sample_total
        std = 100. * np.std(results[method])
        ci95 = 1.96 * std / np.sqrt(args.n_episodes)
        
        print(f"\n{method.upper()}: {acc:.2f}% ± {ci95:.2f}%")
    
    # 6. 总结
    print("\n" + "="*60)
    print("SUMMARY (on NOVEL classes - true few-shot evaluation)")
    print("="*60)
    print(f"  Backbone only:     {100.*np.mean(results['backbone']):.2f}%")
    print(f"  Slot features:     {100.*np.mean(results['slot']):.2f}%")
    improvement = np.mean(results['slot']) - np.mean(results['backbone'])
    print(f"  Improvement:       {100.*improvement:+.2f}%")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--use_gnn', action='store_true')
    
    # Model params (should match training)
    parser.add_argument('--num_slots', type=int, default=8)
    parser.add_argument('--slot_dim', type=int, default=256)
    parser.add_argument('--slot_iters', type=int, default=3)
    parser.add_argument('--gnn_start_iter', type=int, default=1)
    
    # Evaluation params
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--n_episodes', type=int, default=600)
    
    args = parser.parse_args()
    evaluate(args)
