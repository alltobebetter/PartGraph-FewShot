"""
GNN-in-the-Loop Slot Attention Few-Shot 评估脚本

评估指标：
1. Few-shot 准确率 (5-way K-shot)
2. Slot 重叠率 (衡量 slot 是否分工明确)
3. Slot 覆盖率 (衡量 slot 是否覆盖整个图像)
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.backbone import ResNetBackbone
from src.model.gnn_slot_attention import PartAwareGNNSlotAttention
from src.model.slot_attention import PartAwareSlotAttention
from src.model.part_autoencoder import SlotDecoder
from src.data.cub_dataset import CUB200Dataset, get_cub_transforms


class CUB200WithLabels(CUB200Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        super().__init__(root_dir, train, transform)
        self.labels = [int(p.split('.')[0]) - 1 for p in self.image_paths]
    
    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        label = self.labels[idx]
        return image, label


class GNNPartAutoEncoder(torch.nn.Module):
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


class BaselineAutoEncoder(torch.nn.Module):
    """原版 Slot Attention (无 GNN) 用于对比"""
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


def compute_slot_overlap(attn):
    """
    计算 slot 之间的重叠率
    attn: (B, K, H, W)
    返回: 平均重叠率 (越低越好，说明 slot 分工明确)
    """
    B, K, H, W = attn.shape
    attn_flat = attn.view(B, K, -1)  # (B, K, N)
    
    # 归一化
    attn_norm = F.normalize(attn_flat, dim=-1)
    
    # 计算 slot 之间的余弦相似度
    similarity = torch.bmm(attn_norm, attn_norm.transpose(1, 2))  # (B, K, K)
    
    # 去掉对角线
    mask = 1 - torch.eye(K, device=attn.device).unsqueeze(0)
    overlap = (similarity * mask).sum() / (B * K * (K - 1))
    
    return overlap.item()


def compute_slot_coverage(attn):
    """
    计算 slot 对图像的覆盖率
    attn: (B, K, H, W)
    返回: 平均覆盖率 (越高越好)
    """
    # 所有 slot 的最大 attention
    max_attn, _ = attn.max(dim=1)  # (B, H, W)
    
    # 有多少位置被至少一个 slot 关注 (attention > threshold)
    threshold = 0.1
    covered = (max_attn > threshold).float().mean()
    
    return covered.item()


def compute_slot_entropy(attn):
    """
    计算 slot attention 的熵 (越低说明越集中)
    attn: (B, K, H, W)
    """
    B, K, H, W = attn.shape
    attn_flat = attn.view(B, K, -1)
    attn_prob = attn_flat / (attn_flat.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(attn_prob * torch.log(attn_prob + 1e-8)).sum(dim=-1)
    return entropy.mean().item()


def extract_features(model, images, method='slot'):
    """提取特征"""
    with torch.no_grad():
        if method == 'backbone':
            features = model.backbone(images)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        else:
            _, slots, _, _ = model(images)
            features = slots.mean(dim=1)
    return features


def proto_classify(support_features, support_labels, query_features, n_way):
    """Prototypical Network 分类"""
    prototypes = []
    unique_labels = torch.unique(support_labels)
    
    for label in unique_labels:
        mask = support_labels == label
        class_features = support_features[mask]
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)
    dists = torch.cdist(query_features, prototypes)
    predictions = dists.argmin(dim=1)
    
    return predictions


def run_episode(model, dataset, n_way=5, k_shot=5, n_query=15, method='slot', device='cuda'):
    """运行一个 few-shot episode"""
    all_labels = list(set(dataset.labels))
    selected_classes = random.sample(all_labels, n_way)
    
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    
    for new_label, orig_label in enumerate(selected_classes):
        indices = [i for i, l in enumerate(dataset.labels) if l == orig_label]
        
        if len(indices) < k_shot + n_query:
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
    
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_images = torch.stack(query_images).to(device)
    query_labels = torch.tensor(query_labels).to(device)
    
    support_features = extract_features(model, support_images, method)
    query_features = extract_features(model, query_images, method)
    
    predictions = proto_classify(support_features, support_labels, query_features, n_way)
    
    correct = (predictions == query_labels).sum().item()
    total = query_labels.size(0)
    
    return correct, total


def evaluate_slot_quality(model, dataset, device, n_samples=100):
    """评估 slot 质量指标"""
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    
    overlaps = []
    coverages = []
    entropies = []
    
    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating slot quality"):
            img, _ = dataset[idx]
            img = img.unsqueeze(0).to(device)
            
            _, _, attn, _ = model(img)
            
            overlaps.append(compute_slot_overlap(attn))
            coverages.append(compute_slot_coverage(attn))
            entropies.append(compute_slot_entropy(attn))
    
    return {
        'overlap': np.mean(overlaps),
        'coverage': np.mean(coverages),
        'entropy': np.mean(entropies)
    }


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    transform = get_cub_transforms()
    dataset_root = os.path.join(args.data_root, 'CUB_200_2011')
    test_dataset = CUB200WithLabels(dataset_root, train=False, transform=transform)
    print(f"Test dataset: {len(test_dataset)} images")
    
    # Build GNN model
    backbone = ResNetBackbone(pretrained=True, return_layer='layer3')
    
    if args.use_gnn:
        print("\n=== GNN-in-the-Loop Slot Attention ===")
        slot_attn = PartAwareGNNSlotAttention(
            num_slots=args.num_slots,
            in_channels=256,
            slot_dim=args.slot_dim,
            iters=args.slot_iters,
            gnn_start_iter=args.gnn_start_iter
        )
        model = GNNPartAutoEncoder(backbone, slot_attn, 
                                   SlotDecoder(slot_dim=args.slot_dim, resolution=(14,14))).to(device)
    else:
        print("\n=== Baseline Slot Attention (no GNN) ===")
        slot_attn = PartAwareSlotAttention(
            num_slots=args.num_slots,
            in_channels=256,
            slot_dim=args.slot_dim,
            iters=args.slot_iters
        )
        model = BaselineAutoEncoder(backbone, slot_attn,
                                    SlotDecoder(slot_dim=args.slot_dim, resolution=(14,14))).to(device)
    
    # Load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
    
    model.eval()
    
    # 1. Slot quality metrics
    print("\n--- Slot Quality Metrics ---")
    quality = evaluate_slot_quality(model, test_dataset, device)
    print(f"  Overlap:  {quality['overlap']:.4f} (lower is better)")
    print(f"  Coverage: {quality['coverage']:.4f} (higher is better)")
    print(f"  Entropy:  {quality['entropy']:.4f} (lower is better)")
    
    # 2. Few-shot accuracy
    print(f"\n--- {args.n_way}-way {args.k_shot}-shot Evaluation ---")
    
    results = {'slot': [], 'backbone': []}
    
    for method in ['backbone', 'slot']:
        correct_total = 0
        sample_total = 0
        
        for _ in tqdm(range(args.n_episodes), desc=f"{method}"):
            correct, total = run_episode(
                model, test_dataset,
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
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"  Backbone: {100.*np.mean(results['backbone']):.2f}%")
    print(f"  Slot:     {100.*np.mean(results['slot']):.2f}%")
    improvement = np.mean(results['slot']) - np.mean(results['backbone'])
    print(f"  Delta:    {100.*improvement:+.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--use_gnn', action='store_true', help='Use GNN-in-the-Loop')
    parser.add_argument('--num_slots', type=int, default=8)
    parser.add_argument('--slot_dim', type=int, default=256)
    parser.add_argument('--slot_iters', type=int, default=3)
    parser.add_argument('--gnn_start_iter', type=int, default=1)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--n_episodes', type=int, default=100)
    
    args = parser.parse_args()
    evaluate(args)
