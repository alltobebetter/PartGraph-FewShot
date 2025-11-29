"""
Few-Shot 评估脚本
测试训练好的模型在 5-way K-shot 任务上的表现
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.backbone import ResNetBackbone
from src.model.slot_attention import PartAwareSlotAttention
from src.model.part_autoencoder import PartAutoEncoder, SlotDecoder
from src.data.cub_dataset import CUB200Dataset, get_cub_transforms


class CUB200WithLabels(CUB200Dataset):
    """扩展数据集，返回图像和类别标签"""
    def __init__(self, root_dir, train=True, transform=None):
        super().__init__(root_dir, train, transform)
        self.labels = [int(p.split('.')[0]) - 1 for p in self.image_paths]
    
    def __getitem__(self, idx):
        image = super().__getitem__(idx)
        label = self.labels[idx]
        return image, label


def extract_features(model, images, method='slot'):
    """
    提取特征用于 few-shot 分类
    method: 'slot' - 使用 slot 特征, 'backbone' - 使用 backbone 全局特征
    """
    with torch.no_grad():
        if method == 'backbone':
            # Baseline: 直接用 backbone 特征
            features = model.backbone(images)  # (B, C, H, W)
            features = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # (B, C)
        else:
            # 我们的方法: 用 slot 特征
            _, slots, _, _ = model(images)  # slots: (B, K, D)
            # 平均所有 slot
            features = slots.mean(dim=1)  # (B, D)
    return features


def proto_classify(support_features, support_labels, query_features, n_way):
    """
    Prototypical Network 分类
    """
    # 计算每个类的原型 (类内平均)
    prototypes = []
    unique_labels = torch.unique(support_labels)
    
    for label in unique_labels:
        mask = support_labels == label
        class_features = support_features[mask]
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)
    
    prototypes = torch.stack(prototypes)  # (N, D)
    
    # 计算查询样本到每个原型的距离
    # 使用负欧氏距离作为相似度
    dists = torch.cdist(query_features, prototypes)  # (Q, N)
    predictions = dists.argmin(dim=1)
    
    return predictions


def run_episode(model, dataset, n_way=5, k_shot=5, n_query=15, method='slot', device='cuda'):
    """
    运行一个 few-shot episode
    """
    # 1. 随机选择 n_way 个类
    all_labels = list(set(dataset.labels))
    selected_classes = random.sample(all_labels, n_way)
    
    # 2. 为每个类采样 k_shot + n_query 个样本
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []
    
    for new_label, orig_label in enumerate(selected_classes):
        # 找到该类的所有样本索引
        indices = [i for i, l in enumerate(dataset.labels) if l == orig_label]
        
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
    
    # 3. 转换为 tensor
    support_images = torch.stack(support_images).to(device)
    support_labels = torch.tensor(support_labels).to(device)
    query_images = torch.stack(query_images).to(device)
    query_labels = torch.tensor(query_labels).to(device)
    
    # 4. 提取特征
    support_features = extract_features(model, support_images, method)
    query_features = extract_features(model, query_images, method)
    
    # 5. 分类
    predictions = proto_classify(support_features, support_labels, query_features, n_way)
    
    # 6. 计算准确率
    correct = (predictions == query_labels).sum().item()
    total = query_labels.size(0)
    
    return correct, total


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load dataset (test set)
    transform = get_cub_transforms()
    dataset_root = os.path.join(args.data_root, 'CUB_200_2011')
    
    test_dataset = CUB200WithLabels(dataset_root, train=False, transform=transform)
    print(f"Test dataset: {len(test_dataset)} images")
    
    # 2. Build model
    backbone = ResNetBackbone(pretrained=True, return_layer='layer3')
    slot_attn = PartAwareSlotAttention(
        num_slots=args.num_slots,
        in_channels=256,
        slot_dim=args.slot_dim,
        iters=3
    )
    decoder = SlotDecoder(slot_dim=args.slot_dim, resolution=(14, 14), upsample_steps=4)
    model = PartAutoEncoder(backbone, slot_attn, decoder).to(device)
    
    # 3. Load checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
    else:
        print("No checkpoint loaded, using random/pretrained weights")
    
    model.eval()
    
    # 4. Run episodes
    print(f"\nEvaluating {args.n_way}-way {args.k_shot}-shot...")
    print(f"Running {args.n_episodes} episodes...\n")
    
    results = {'slot': [], 'backbone': []}
    
    for method in ['backbone', 'slot']:
        correct_total = 0
        sample_total = 0
        
        for _ in tqdm(range(args.n_episodes), desc=f"Method: {method}"):
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
        
        print(f"\n{method.upper()} Method:")
        print(f"  Accuracy: {acc:.2f}% ± {ci95:.2f}%")
    
    # 5. Compare
    print("\n" + "="*50)
    print("COMPARISON:")
    print(f"  Backbone (baseline): {100.*np.mean(results['backbone']):.2f}%")
    print(f"  Slot (ours):         {100.*np.mean(results['slot']):.2f}%")
    
    improvement = np.mean(results['slot']) - np.mean(results['backbone'])
    print(f"  Improvement:         {100.*improvement:+.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_slots', type=int, default=8)
    parser.add_argument('--slot_dim', type=int, default=256)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--n_episodes', type=int, default=100)
    
    args = parser.parse_args()
    evaluate(args)
