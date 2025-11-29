# V2 优化文档

## 背景

V1 实验结果：
- Baseline (ResNet): **88.67%**
- Slot 方法: **26.77%** (几乎等于随机猜测 20%)

核心问题：**重建任务学到的特征对分类没用**

---

## 优化内容

### 1. 训练目标改变（核心改动）

| 版本 | 损失函数 | 问题 |
|------|---------|------|
| V1 | `loss = MSE(重建, 原图)` | 学像素复制，不学判别 |
| V2 | `loss = 1.0*分类 + 0.1*重建 + 0.01*多样性` | 直接学分类 |

**原理**：
- MSE 重建鼓励输出"平均颜色"（灰色图像）
- 分类损失直接告诉模型"区分 200 种鸟"
- 训练目标 = 测试目标，不再错位

### 2. 增加 Slot 容量

| 参数 | V1 | V2 |
|------|----|----|
| num_slots | 5 | 8 |
| slot_dim | 64 | 256 |

**原理**：
- 5 个 slot 只能分"前景/背景"，太粗
- 8 个 slot 可以分出头、嘴、翅膀、尾巴等部件
- 64 维信息量不够，256 维可以编码更复杂特征

### 3. 多样性损失

```python
def diversity_loss(attn):
    # 惩罚不同 slot 关注相同区域
    similarity = cosine_sim(slot_i_attn, slot_j_attn)
    return similarity.mean()
```

**原理**：
- V1 多个 slot 可能 collapse 到同一区域（都看整只鸟）
- 多样性损失强迫每个 slot 关注不同部位
- 实现真正的"部件分解"

### 4. 改进 Slot 初始化

| 版本 | 初始化方式 |
|------|-----------|
| V1 | 所有 slot 共享一个 μ 和 σ，随机采样 |
| V2 | 每个 slot 独立可学习的 μ 和 σ |

**原理**：
- V1 所有 slot 初始几乎相同，需要很久才能分化
- V2 从一开始就让不同 slot 有不同"职责倾向"

---

## 代码改动

| 文件 | 改动 |
|------|------|
| `src/train_improved.py` | 新增，多任务训练脚本 |
| `src/eval_fewshot.py` | 新增，few-shot 评估脚本 |
| `src/model/slot_attention.py` | 修改初始化方式 |
| `PartGraph_Train_Improved.ipynb` | 新增，Colab 训练 notebook |

---

## 预期结果

| 方法 | 准确率 |
|------|--------|
| 随机猜测 | 20% |
| V1 Slot (重建) | 26.77% |
| **V2 Slot (多任务)** | **50-70%** (预期) |
| Baseline (ResNet) | 88.67% |

### 各优化贡献预估

| 优化项 | 预期提升 |
|--------|---------|
| 分类损失 | +20-30% |
| 增加容量 | +5-10% |
| 多样性损失 | +3-5% |
| 改进初始化 | 加速收敛 |

---

## 运行命令

```bash
# 训练
python src/train_improved.py \
    --data_root ./data \
    --epochs 30 \
    --num_slots 8 \
    --slot_dim 256

# 评估
python src/eval_fewshot.py \
    --checkpoint ./checkpoints_improved/best_model.pth \
    --n_episodes 100
```

---

## 如果 V2 失败

如果准确率仍 < 40%，说明 Slot Attention + 重建框架本身有问题，需要：

1. **方案 B**：纯对比学习，完全去掉重建
2. **方案 C**：利用 CUB 关键点做弱监督
3. **方案 D**：换成 DINO/MAE 等自监督预训练

---

*文档日期: 2025-11-29*
