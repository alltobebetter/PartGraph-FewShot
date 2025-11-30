# PartGraph 项目里程碑报告

**日期**: 2024年11月30日  
**状态**: 核心验证完成，准备进入下一阶段

---

## 一、项目概述

### 1.1 研究背景与动机

少样本学习（Few-Shot Learning）是机器学习领域的重要研究方向，旨在解决深度学习模型对大规模标注数据的依赖问题。人类能够通过少量样本快速学习新概念，而当前的深度学习模型通常需要成千上万的标注样本才能达到良好的性能。

认知科学研究表明，人类在识别物体时会自动将其分解为有意义的部件（如动物的头部、躯干、四肢等），并通过理解部件之间的空间和语义关系来进行识别。这种"部件级"的认知方式具有天然的组合泛化能力——已知部件的新组合可以快速被识别为新类别。

基于上述观察，本项目提出了一个核心假设：**如果让模型学习部件级表征而非整体特征，将显著提升其在少样本场景下的泛化能力**。

### 1.2 研究目标

本项目的研究目标包括：

1. **验证部件表征假设**：通过 Slot Attention 机制实现无监督的部件发现，验证部件级表征在 Few-Shot Learning 中的有效性
2. **探索部件关系建模**：设计 GNN-in-the-Loop 机制，在部件发现过程中建模部件间的空间和语义关系
3. **构建端到端框架**：建立从部件发现到关系建模再到分类的完整 Few-Shot Learning 框架

### 1.3 技术路线

本项目采用的技术路线如下：

```
输入图像 → 特征提取(ResNet) → 部件发现(Slot Attention) → 关系建模(GNN) → 分类(Prototypical Network)
```

---

## 二、实验结果与分析

### 2.1 实验设置

**数据集**: CUB-200-2011（Caltech-UCSD Birds 200）
- 总计 200 个鸟类类别，约 12,000 张图像
- Base Classes（训练）: 类别 1-100，约 5,864 张图像
- Novel Classes（测试）: 类别 101-200，约 5,924 张图像

**评估协议**: 5-way 5-shot
- 每个 episode 随机选择 5 个类别
- 每个类别提供 5 张支持样本（support set）
- 每个类别测试 15 张查询样本（query set）
- 共进行 600 个 episode，报告平均准确率和 95% 置信区间

**实现细节**:
- Backbone: ResNet-18（ImageNet 预训练）
- Slot 数量: 8
- Slot 维度: 256
- Slot Attention 迭代次数: 3
- 训练轮数: 约 35-40 轮（自动早停）
- 优化器: AdamW，学习率 1e-4（Baseline）/ 5e-5（GNN）

### 2.2 主要结果

| 方法 | Novel Classes 准确率 | 95% 置信区间 | 相对 Backbone 提升 |
|------|---------------------|-------------|-------------------|
| ResNet Backbone (Baseline) | 47.43% | ±0.73% | - |
| ResNet Backbone (GNN 实验) | 53.76% | - | +6.33% |
| Slot Attention | 79.24% | ±0.79% | +31.81% |
| Slot Attention + GNN-in-Loop | 79.94% | ±0.74% | +32.51% |

### 2.3 结果分析

**核心发现 1: 部件表征显著优于全局表征**

Slot Attention 提取的部件级特征相比 ResNet 全局特征，在 Novel Classes 上的准确率提升了 31.81 个百分点。这一结果有力地验证了我们的核心假设：部件级表征具有更强的跨类别泛化能力。

从认知角度解释，不同鸟类虽然整体外观差异较大，但共享许多相似的部件（喙、翅膀、爪子等）。模型在 Base Classes 上学习到的部件表征可以直接迁移到 Novel Classes，因为新类别只是已知部件的新组合。

**核心发现 2: GNN-in-the-Loop 效果有限**

GNN 版本相比 Baseline Slot Attention 仅提升了 0.70 个百分点（79.24% → 79.94%），未达到预期效果。我们分析可能的原因如下：

1. **分类器限制**: 当前使用的 Prototypical Network 会对所有 slot 特征取平均，这一操作丢失了 GNN 学习到的结构信息
2. **数据集特性**: CUB-200 中鸟类的部件空间关系相对固定（头在上、翅膀在两侧），GNN 建模的关系信息可能冗余
3. **训练策略**: GNN 增加了模型复杂度，可能需要更精细的训练策略

这一负面结果本身具有研究价值，为后续改进指明了方向。

**核心发现 3: 评估协议的重要性**

在早期实验中，我们在全部 200 类上进行训练和测试（仅划分不同图像），获得了 88% 的准确率。然而，这种设置测试的是"泛化到新图像"的能力，而非"泛化到新类别"的能力。

采用标准的 Base/Novel 类别划分后，准确率下降到约 80%，但这才是真正有意义的 Few-Shot 评估。这一经验提醒我们实验设置的严谨性对于研究结论的可靠性至关重要。

---

## 三、相关工作与创新性分析

### 3.1 相关工作

**Slot Attention 相关**:
- Locatello et al. (NeurIPS 2020) 提出 Slot Attention 用于无监督物体发现
- 后续工作主要集中在视频理解、场景分解等领域
- 将 Slot Attention 用于 Few-Shot Learning 的工作较少

**Few-Shot Learning 相关**:
- Prototypical Networks (Snell et al., NeurIPS 2017) 是经典的度量学习方法
- 近年来出现了基于 GNN 的 Few-Shot 方法，但主要用于样本间关系建模
- 部件级表征在 Few-Shot 中的应用尚未被充分探索

**最相关工作**:
- SAFF (Ródenas et al., CVPR 2025 Workshop): 使用 Slot Attention 进行特征过滤，但未涉及部件关系建模
- Slot Structured World Models (Collu et al., arXiv 2024): 结合 Slot Attention 和 GNN，但用于世界模型而非 Few-Shot

### 3.2 本项目的创新点

1. **系统性地将 Slot Attention 的部件发现能力应用于 Few-Shot Learning**
   - 不同于简单的特征过滤，我们构建了完整的部件级表征框架
   - 实验证明部件表征带来 31% 的显著提升

2. **提出 GNN-in-the-Loop 机制**
   - 在 Slot Attention 的迭代过程中嵌入 GNN 消息传递
   - 使 slot 之间能够交换信息，实现协作式部件发现
   - 虽然当前效果有限，但提供了新的研究方向

3. **严格的实验验证**
   - 采用标准的 Base/Novel 类别划分
   - 提供完整的消融实验
   - 开源全部代码和实验配置

---

## 四、当前水平定位

### 4.1 与现有方法的比较

在 CUB-200-2011 数据集的 5-way 5-shot 设置下：

| 方法 | 准确率 | 发表venue |
|------|--------|----------|
| ProtoNet | 65-70% | NeurIPS 2017 |
| FEAT | 75-78% | CVPR 2020 |
| DeepEMD | 78-82% | CVPR 2020 |
| **本项目** | **~80%** | - |
| 当前 SOTA | 85%+ | 2023-2024 |

本项目的结果处于**中上水平**，与 2020 年的顶会方法相当，但与最新 SOTA 仍有差距。

### 4.2 优势与局限

**优势**:
- 方法简洁，概念清晰，易于理解和复现
- 部件表征提供了良好的可解释性（可视化 slot attention map）
- 无需额外的部件级标注，完全无监督学习

**局限**:
- 当前分类器（Prototypical Network）未能充分利用部件间的结构信息
- Backbone 较弱（ResNet-18），限制了特征提取能力
- 仅在 CUB-200 单一数据集上验证

---

## 五、未来研究方向

### 5.1 短期改进（1-2 周）

| 改进方向 | 预期提升 | 实现难度 |
|---------|---------|---------|
| 升级 Backbone 至 ResNet-50 | 2-5% | 低 |
| 增强数据增强策略 | 1-3% | 低 |
| 超参数优化 | 1-2% | 中 |

### 5.2 中期研究（1-2 月）

| 研究方向 | 预期提升 | 重要性 |
|---------|---------|--------|
| **实现 Graph Matching 分类器** | 3-5% | 最高 |
| 在 miniImageNet 上验证 | - | 高 |
| 在 tieredImageNet 上验证 | - | 高 |

**Graph Matching 分类器是最关键的改进方向**。当前 Prototypical Network 通过平均 slot 特征进行分类，丢失了部件间的结构信息。Graph Matching 可以直接比较两个部件图的结构相似度，从而充分利用 GNN 学习到的关系信息。

### 5.3 长期愿景

构建完整的 PartGraph 框架：

```
Input → Backbone → Slot Attention → GNN → Graph Matching → Classification
              ↓           ↓           ↓
          特征提取     部件发现    关系建模    结构匹配
```

每个模块承担明确的功能，形成端到端的"部件思维"Few-Shot Learning 系统。

---

## 六、资源需求评估

### 6.1 当前资源状况

- 计算资源: Google Colab 免费版 / Kaggle 免费 GPU
- GPU 类型: NVIDIA T4 (15GB) / P100 (16GB)
- 单次训练时间: 约 1-2 小时

### 6.2 后续研究所需资源

| 资源类型 | 用途 | 优先级 |
|---------|------|--------|
| 更多 GPU 时间 | 大规模实验、超参数搜索 | 高 |
| 高性能 GPU (A100) | 更大 Backbone、更多数据集 | 中 |
| 存储空间 | miniImageNet、tieredImageNet 数据集 | 中 |

---

## 七、论文撰写规划

### 7.1 目标发表venue

- 短期目标: CVPR/ICCV/ECCV Workshop
- 中期目标: AAAI/IJCAI
- 长期目标: CVPR/ICCV/ECCV 主会议（需要 Graph Matching 等改进）

### 7.2 论文结构规划

```
Title: Part-Aware Few-Shot Learning via Slot Attention

Abstract

1. Introduction
   1.1 Few-Shot Learning 的挑战
   1.2 部件表征的认知动机
   1.3 本文贡献

2. Related Work
   2.1 Few-Shot Learning
   2.2 Object-Centric Learning
   2.3 Graph Neural Networks for Few-Shot

3. Method
   3.1 Problem Formulation
   3.2 Part Discovery via Slot Attention
   3.3 GNN-in-the-Loop (Optional)
   3.4 Classification

4. Experiments
   4.1 Experimental Setup
   4.2 Main Results
   4.3 Ablation Studies
   4.4 Visualization and Analysis

5. Conclusion and Future Work

References
```

### 7.3 核心贡献总结

1. 首次系统性地将 Slot Attention 应用于 Few-Shot Learning，实现 31% 的性能提升
2. 提出 GNN-in-the-Loop 机制，探索部件协作发现
3. 提供严格的实验验证和深入的分析

---

## 八、总结与展望

### 8.1 阶段性成果

本项目已完成以下工作：

1. 验证了部件表征在 Few-Shot Learning 中的有效性，相比全局特征提升 31%
2. 实现了完整的训练和评估框架，支持标准的 Base/Novel 类别划分
3. 探索了 GNN-in-the-Loop 机制，虽然效果有限但提供了有价值的负面结果
4. 开源了全部代码，支持一键复现

### 8.2 里程碑意义

本报告标志着项目的**第一个重要里程碑**：

- 核心假设得到实验验证
- 具备了可展示的研究成果
- 明确了后续改进方向

### 8.3 下一步行动

最优先事项：**实现 Graph Matching 分类器**

这是让 GNN 真正发挥作用的关键，也是进一步提升性能的最大潜力点。

---

## 附录：项目文件结构

```
PartGraph-FewShot/
├── src/
│   ├── model/
│   │   ├── slot_attention.py        # Slot Attention 实现
│   │   ├── gnn_slot_attention.py    # GNN-in-the-Loop 实现
│   │   ├── backbone.py              # ResNet Backbone
│   │   └── part_autoencoder.py      # 解码器（用于重建损失）
│   ├── data/
│   │   └── cub_dataset.py           # CUB-200 数据集加载
│   ├── utils/
│   │   ├── visualize.py             # 可视化工具
│   │   └── pos_embed.py             # 位置编码
│   ├── train_fewshot.py             # 训练脚本
│   └── eval_fewshot_novel.py        # 评估脚本
├── docs/
│   ├── PROJECT_MILESTONE.md         # 本报告
│   ├── technical_design.md          # 技术设计文档
│   └── GNN_IN_LOOP_DESIGN.md        # GNN 设计文档
├── train_fewshot.ipynb              # Colab 训练 notebook
├── train_gnn_kaggle.ipynb           # Kaggle 训练 notebook
└── README.md                        # 项目说明
```

---

*报告完成日期: 2024年11月30日*
