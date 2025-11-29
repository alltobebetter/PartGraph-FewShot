# 技术设计文档

## 1. 整体架构

```
┌────────────────────────────────────────────────────────────────────┐
│                         PartGraph 架构图                            │
│                                                                     │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────┐ │
│  │  Input  │ →  │   Backbone   │ →  │    Slot     │ →  │  GNN   │ │
│  │  Image  │    │  (ResNet18)  │    │  Attention  │    │        │ │
│  └─────────┘    └──────────────┘    └─────────────┘    └────────┘ │
│                        ↓                   ↓               ↓       │
│                   Feature Map         K Slots          Part Graph  │
│                   (14×14×512)        (K×256)          (Nodes+Edges)│
│                                                            ↓       │
│                                                     ┌────────────┐ │
│                                                     │   Graph    │ │
│                                                     │  Matching  │ │
│                                                     └────────────┘ │
│                                                            ↓       │
│                                                      Classification│
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块详细设计

### 2.1 Backbone (特征提取器)

**选择**：ResNet-18 (预训练于ImageNet)

**输出**：去掉最后的全局池化和分类层，保留特征图

```python
# 输入: (B, 3, 224, 224)
# 输出: (B, 512, 7, 7) 或 (B, 256, 14, 14) 取决于截取位置
```

**为什么选ResNet-18**：
- 参数量小，适合few-shot场景
- 和大多数baseline一致，便于公平对比
- 可以换成其他backbone（ViT等）

---

### 2.2 Part-Aware Slot Attention

#### 核心公式

```
输入: 
  - F ∈ R^(B×H×W×D): 特征图
  - slots ∈ R^(B×K×D): 初始槽位

迭代 T 次:
  1. 计算 attention:
     attn = softmax(k(F) · q(slots)^T / √D, dim=slots)
     # attn ∈ R^(B×H×W×K)
     # 注意: softmax在slots维度，实现竞争
  
  2. 加权聚合:
     updates = attn^T · v(F)  # (B×K×D)
  
  3. 更新slots:
     slots = GRU(slots, updates)

输出:
  - slots: 最终的部件表征
  - attn: attention map，用于可视化和位置计算
```

#### 我们的改进

**改进1: 位置编码**
```python
# 给特征图加入2D位置编码
pos_enc = get_2d_sincos_pos_embed(D, H, W)
F = F + pos_enc
```

**改进2: 局部性损失**
```python
# 鼓励每个slot的attention是空间连续的
def locality_loss(attn):
    # attn: (B, K, H, W)
    # 计算每个slot的attention的空间方差
    # 方差越小说明越集中
    coords = get_coord_grid(H, W)  # (H, W, 2)
    
    for k in range(K):
        weights = attn[:, k]  # (B, H, W)
        center = (weights.unsqueeze(-1) * coords).sum(dim=[1,2])
        variance = (weights.unsqueeze(-1) * (coords - center)**2).sum()
    
    return variance  # 最小化这个
```

**改进3: 多尺度融合**
```python
# 从backbone的不同层提取特征
F1 = backbone.layer2(x)  # 大尺度，捕捉大部件
F2 = backbone.layer3(x)  # 中尺度
F3 = backbone.layer4(x)  # 小尺度，捕捉细节

# 分别做slot attention，然后融合
slots1 = slot_attn(F1, slots[:, :K1])
slots2 = slot_attn(F2, slots[:, K1:K2])
slots3 = slot_attn(F3, slots[:, K2:])
```

---

### 2.3 Relation Graph Module

#### 图的构建

```python
def build_graph(slots, attention_maps):
    """
    slots: (B, K, D) - 部件表征
    attention_maps: (B, K, H, W) - 每个部件的attention
    
    返回:
    - nodes: (B, K, D) - 节点特征（就是slots）
    - edges: (B, K, K, E) - 边特征
    - adj: (B, K, K) - 邻接矩阵（可选，全连接或基于距离）
    """
    
    # 1. 计算每个部件的中心位置
    positions = compute_slot_centers(attention_maps)  # (B, K, 2)
    
    # 2. 计算边特征
    edges = []
    for i in range(K):
        for j in range(K):
            if i != j:
                # 空间关系
                dx = positions[:, j, 0] - positions[:, i, 0]
                dy = positions[:, j, 1] - positions[:, i, 1]
                dist = torch.sqrt(dx**2 + dy**2)
                angle = torch.atan2(dy, dx)
                
                # 语义关系（可学习）
                semantic = MLP(torch.cat([slots[:, i], slots[:, j]], dim=-1))
                
                edge_feat = torch.stack([dx, dy, dist, angle, semantic], dim=-1)
                edges.append(edge_feat)
    
    edges = torch.stack(edges, dim=1).reshape(B, K, K, -1)
    
    return slots, edges
```

#### Graph Attention Network

```python
class PartGAT(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads=4):
        self.W_q = nn.Linear(node_dim, node_dim)
        self.W_k = nn.Linear(node_dim, node_dim)
        self.W_v = nn.Linear(node_dim, node_dim)
        self.W_e = nn.Linear(edge_dim, num_heads)  # 边特征影响attention
        
    def forward(self, nodes, edges):
        # nodes: (B, K, D)
        # edges: (B, K, K, E)
        
        Q = self.W_q(nodes)  # (B, K, D)
        K = self.W_k(nodes)  # (B, K, D)
        V = self.W_v(nodes)  # (B, K, D)
        
        # 计算attention，融入边特征
        attn = torch.einsum('bqd,bkd->bqk', Q, K) / sqrt(D)
        edge_bias = self.W_e(edges).squeeze(-1)  # (B, K, K)
        attn = attn + edge_bias
        attn = F.softmax(attn, dim=-1)
        
        # 更新节点
        out = torch.einsum('bqk,bkd->bqd', attn, V)
        
        return nodes + out  # 残差连接
```

---

### 2.4 Graph Matching Classifier

#### Few-shot 场景下的图匹配

```python
def graph_matching_classify(query_graph, support_graphs, support_labels):
    """
    query_graph: 查询样本的部件图
    support_graphs: 支持集中所有样本的部件图
    support_labels: 支持集的标签
    
    返回: 查询样本的预测类别
    """
    
    scores = []
    for class_id in unique(support_labels):
        # 获取该类别的所有支持样本图
        class_graphs = [g for g, l in zip(support_graphs, support_labels) if l == class_id]
        
        # 计算查询图和每个支持图的匹配分数
        class_scores = [graph_similarity(query_graph, g) for g in class_graphs]
        
        # 取平均（或最大）作为该类别的分数
        scores.append(mean(class_scores))
    
    return argmax(scores)
```

#### 图相似度计算

```python
def graph_similarity(G1, G2):
    """
    G1, G2: 两个部件图，每个包含 (nodes, edges)
    """
    nodes1, edges1 = G1
    nodes2, edges2 = G2
    
    # 方法1: 简单的节点匹配 + 结构匹配
    
    # 节点相似度矩阵
    node_sim = cosine_similarity(nodes1, nodes2)  # (K1, K2)
    
    # 最优匹配（匈牙利算法或软匹配）
    # 这里用软匹配（可微分）
    soft_assign = sinkhorn(node_sim)  # (K1, K2)
    
    # 节点匹配分数
    node_score = (soft_assign * node_sim).sum()
    
    # 结构匹配分数（边的一致性）
    # 如果节点i匹配到节点i'，节点j匹配到节点j'
    # 那么边(i,j)应该和边(i',j')相似
    edge_score = compute_edge_consistency(edges1, edges2, soft_assign)
    
    return node_score + lambda * edge_score
```

---

## 3. 训练细节

### 3.1 损失函数

```python
total_loss = (
    L_cls +                    # 分类损失
    lambda1 * L_recon +        # 重建损失（确保部件有意义）
    lambda2 * L_diversity +    # 多样性损失（避免slot collapse）
    lambda3 * L_locality       # 局部性损失（部件应该是局部的）
)
```

**分类损失**：
```python
L_cls = CrossEntropy(predictions, labels)
```

**重建损失**（可选，帮助部件发现）：
```python
# 从slots重建原图
reconstructed = decoder(slots)
L_recon = MSE(reconstructed, original_image)
```

**多样性损失**：
```python
# 不同slot的attention应该关注不同区域
def diversity_loss(attn):
    # attn: (B, K, H*W)
    attn_flat = attn.reshape(B, K, -1)
    # 计算不同slot之间的attention重叠
    overlap = torch.einsum('bki,bji->bkj', attn_flat, attn_flat)
    # 对角线是自己和自己，不算
    mask = 1 - torch.eye(K)
    return (overlap * mask).mean()
```

### 3.2 训练流程

```python
for episode in range(num_episodes):
    # 1. 采样 N-way K-shot episode
    support_images, support_labels, query_images, query_labels = sample_episode()
    
    # 2. 提取部件图
    support_graphs = [extract_part_graph(img) for img in support_images]
    query_graphs = [extract_part_graph(img) for img in query_images]
    
    # 3. 分类
    predictions = []
    for q_graph in query_graphs:
        pred = graph_matching_classify(q_graph, support_graphs, support_labels)
        predictions.append(pred)
    
    # 4. 计算损失
    loss = compute_loss(predictions, query_labels, ...)
    
    # 5. 反向传播
    loss.backward()
    optimizer.step()
```

---

## 4. 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_slots (K) | 8 | 最大部件数量 |
| slot_dim | 256 | 部件表征维度 |
| slot_iterations | 3 | Slot Attention迭代次数 |
| gnn_layers | 2 | 图神经网络层数 |
| lambda1 (recon) | 0.1 | 重建损失权重 |
| lambda2 (diversity) | 0.01 | 多样性损失权重 |
| lambda3 (locality) | 0.01 | 局部性损失权重 |
| learning_rate | 1e-4 | 学习率 |
| episodes | 100000 | 训练episode数 |

---

## 5. 实现注意事项

### 5.1 显存优化

Slot Attention 和 Graph Matching 都比较吃显存，需要注意：

1. **梯度检查点**：对Slot Attention的迭代使用gradient checkpointing
2. **混合精度**：使用FP16训练
3. **小batch**：few-shot本身batch就小，问题不大

### 5.2 训练稳定性

1. **预热**：先冻结backbone，只训练slot attention
2. **渐进式**：先用简单的分类头，再换成graph matching
3. **正则化**：对slots加L2正则，防止爆炸

### 5.3 可视化调试

训练过程中定期可视化：
1. 每个slot的attention map
2. 部件图的结构
3. 图匹配的对应关系

这对调试非常重要！
