# PartGraph: Part-Aware Few-Shot Learning via Slot Attention

基于部件级表征的少样本学习框架

---

## 核心成果

在 CUB-200-2011 数据集上的 5-way 5-shot 评估（Novel Classes）：

| 方法 | 准确率 | 提升 |
|------|--------|------|
| ResNet Backbone | 47.43% | - |
| **Slot Attention (Ours)** | **79.24%** | **+31.81%** |
| Slot Attention + GNN | 79.94% | +32.51% |

---

## 项目结构

```
PartGraph-FewShot/
├── src/
│   ├── model/
│   │   ├── backbone.py              # ResNet-18 特征提取
│   │   ├── slot_attention.py        # Slot Attention 部件发现
│   │   ├── gnn_slot_attention.py    # GNN-in-the-Loop 版本
│   │   └── part_autoencoder.py      # 解码器（重建损失）
│   ├── data/
│   │   └── cub_dataset.py           # CUB-200 数据集
│   ├── utils/
│   │   ├── visualize.py             # 可视化工具
│   │   └── pos_embed.py             # 位置编码
│   ├── train_fewshot.py             # 训练脚本
│   └── eval_fewshot_novel.py        # 评估脚本
├── docs/
│   ├── PROJECT_MILESTONE.md         # 项目里程碑报告
│   ├── GLOSSARY.md                  # 专业术语详解
│   ├── technical_design.md          # 技术设计文档
│   └── GNN_IN_LOOP_DESIGN.md        # GNN 设计文档
├── train_fewshot.ipynb              # Colab 训练 notebook
├── train_gnn_kaggle.ipynb           # Kaggle GNN 训练 notebook
└── requirements.txt                 # 依赖
```

---

## 快速开始

### 环境配置

```bash
pip install -r requirements.txt
```

### 数据集准备

下载 CUB-200-2011 数据集：
```bash
mkdir -p data && cd data
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
cd ..
```

### 训练

```bash
# Baseline (Slot Attention)
python src/train_fewshot.py \
    --data_root ./data \
    --output_dir ./checkpoints \
    --num_slots 8 \
    --slot_dim 256 \
    --epochs 50

# GNN-in-the-Loop 版本
python src/train_fewshot.py \
    --data_root ./data \
    --output_dir ./checkpoints_gnn \
    --use_gnn \
    --num_slots 8 \
    --slot_dim 256 \
    --epochs 50
```

### 评估

```bash
python src/eval_fewshot_novel.py \
    --data_root ./data \
    --checkpoint ./checkpoints/best_model.pth \
    --n_way 5 \
    --k_shot 5 \
    --n_episodes 600
```

---

## 一键运行

- **Colab**: 上传 `train_fewshot.ipynb`
- **Kaggle**: 上传 `train_gnn_kaggle.ipynb`

---

## 方法概述

1. **部件发现**: 使用 Slot Attention 将图像分解为 K 个部件表征
2. **关系建模**: (可选) 使用 GNN 建模部件间的空间关系
3. **分类**: 使用 Prototypical Network 进行 Few-Shot 分类

详细技术文档见 `docs/` 目录。

---

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@misc{partgraph2024,
  title={PartGraph: Part-Aware Few-Shot Learning via Slot Attention},
  year={2024}
}
```

---

## License

MIT License
