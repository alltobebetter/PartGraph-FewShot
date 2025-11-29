# GNN-in-the-Loop Slot Attention

## 核心创新

**问题**：原版 Slot Attention 中，每个 slot 独立竞争像素，完全不知道其他 slot 在干嘛。

**后果**：
- Slot 可能重叠（两个 slot 都抢"头"）
- Slot 可能遗漏（没人管"尾巴"）
- 部件间的结构约束被忽略（"翅膀应该在身体两侧"）

**我们的解法**：在 Slot Attention 的每次迭代中插入 GNN 消息传递，让 slot 之间交换信息。

```
原版 Slot Attention (堆叠，无创新):
    slots = slot_attention(features)  # 迭代 T 次
    graph = gnn(slots)                # 再过 GNN

我们的方法 (GNN-in-the-Loop):
    for t in range(T):
        slots = slot_attention_step(features, slots)
        slots = gnn_message_passing(slots)  # 每次迭代都让 slot 交互
```

## 技术细节

### 1. 动态图构建

每次迭代后，基于 slot 的 attention map 计算空间位置（质心），构建动态图：

```python
positions = compute_slot_centers(attn)  # (B, K, 2)
edge_attr = encode_spatial_relation(positions)  # 相对位置、距离、角度
```

### 2. 边特征

| 特征 | 维度 | 含义 |
|------|------|------|
| delta | 2 | 相对位置 (dx, dy) |
| dist | 1 | 欧氏距离 |
| angle | 1 | 方向角 |
| similarity | 1 | slot 语义相似度 |
| is_near | 1 | 是否邻近 (dist < 0.3) |
| is_very_near | 1 | 是否很近 (dist < 0.15) |

### 3. GNN 消息传递

使用 GAT 风格的多头注意力，边特征作为 attention bias：

```python
attn = Q @ K.T / sqrt(d) + edge_encoder(edge_attr)
out = softmax(attn) @ V
slots = LayerNorm(slots + out)
```

## 实验设计

### 消融实验

| 配置 | 说明 |
|------|------|
| Baseline | 原版 Slot Attention，无 GNN |
| GNN-after | Slot Attention 后接 GNN（堆叠） |
| GNN-in-Loop (ours) | 每次迭代都有 GNN |

### 评估指标

1. **Few-shot 准确率**：5-way K-shot
2. **Slot 重叠率**：slot attention 之间的余弦相似度（越低越好）
3. **Slot 覆盖率**：图像被 slot 覆盖的比例（越高越好）
4. **Slot 熵**：attention 的集中程度（越低越好）

## 文件结构

```
src/model/
├── gnn_slot_attention.py   # 核心实现
│   ├── SlotGNNLayer        # GNN 消息传递层
│   ├── GNNInLoopSlotAttention  # 带 GNN 的 Slot Attention
│   └── PartAwareGNNSlotAttention  # 完整模块
├── slot_attention.py       # 原版 (baseline)
└── ...

src/
├── train_gnn_slot.py       # 训练脚本
└── eval_fewshot_gnn.py     # 评估脚本
```

## 运行命令

```bash
# 训练
python src/train_gnn_slot.py \
    --num_slots 8 \
    --slot_dim 256 \
    --slot_iters 3 \
    --gnn_start_iter 1 \
    --epochs 50

# 评估
python src/eval_fewshot_gnn.py \
    --checkpoint ./checkpoints_gnn_slot/best_model.pth \
    --use_gnn \
    --n_episodes 600
```

## 预期结果

如果方法有效，应该看到：
1. Slot 重叠率下降（slot 分工更明确）
2. Slot 覆盖率上升（不遗漏部件）
3. Few-shot 准确率提升

## 论文故事线

> "Slot Attention 的独立竞争机制忽略了部件间的结构约束。我们提出 GNN-in-the-Loop，在每次迭代中让 slot 通过消息传递感知彼此，实现协作式部件发现。实验表明，这种设计显著降低了 slot 重叠，提升了 few-shot 分类性能。"
