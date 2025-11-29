# PartGraph-FewShot Project Status Report

Date: 2025-11-29  
Stage: MVP Completed → **Improvement V1 Ready**  
Status: Major Improvements Needed → **Testing New Approach**

---

## 1. Project Goal

### Core Idea
Simulate human cognition through unsupervised part discovery for few-shot learning.

### Core Hypothesis
- Humans learn new objects from 3 images by decomposing them into reusable parts
- If machines learn this decomposition ability, few-shot learning efficiency improves

---

## 2. Implementation History

### V1 (Original MVP)
- ResNet-18 Backbone
- Slot Attention (5 slots, dim=64)
- Reconstruction-only training (20 epochs)
- **Result: 26.77% (Failed)**

### V2 (Current - Improved)
- Slot Attention (8 slots, dim=256)
- **Multi-task learning**: Classification + Reconstruction
- **Diversity loss**: Prevent slot collapse
- **Improved initialization**: Per-slot learnable init
- **Status: Ready to test**

---

## 3. Experimental Results

### 3.1 V1 Results (Original)

| Method | 5-way 5-shot Accuracy |
|--------|----------------------|
| Random Guess | 20.00% |
| Baseline (ResNet) | **88.67%** |
| V1 Slot (recon only) | **26.77%** ❌ |

### 3.2 V2 Expected Results

| Method | Expected Accuracy |
|--------|------------------|
| V2 Slot (multi-task) | **50-70%** (target) |

---

## 4. Problem Diagnosis (V1)

### Core Issue
**Reconstruction Task != Classification Task**

MSE reconstruction learns:
- ❌ Average colors (gray output)
- ❌ Rough positions only
- ❌ No discriminative features

### Root Causes
1. Slot dim too small (64 → 256)
2. Too few slots (5 → 8)
3. No classification signal
4. Slots collapse to similar patterns

---

## 5. V2 Improvements

### Code Changes
1. `src/train_improved.py` - Multi-task training script
2. `src/eval_fewshot.py` - Few-shot evaluation script
3. `src/model/slot_attention.py` - Improved initialization
4. `PartGraph_Train_Improved.ipynb` - Colab notebook

### Key Differences

| Aspect | V1 | V2 |
|--------|----|----|
| num_slots | 5 | 8 |
| slot_dim | 64 | 256 |
| Loss | MSE only | MSE + CE + Diversity |
| λ_cls | 0 | 1.0 |
| λ_recon | 1.0 | 0.1 |
| λ_div | 0 | 0.01 |

---

## 6. Next Steps

### Immediate (Today)
```bash
# Run improved training
python src/train_improved.py --epochs 30 --num_slots 8 --slot_dim 256

# Evaluate
python src/eval_fewshot.py --checkpoint ./checkpoints_improved/best_model.pth
```

### If V2 Works (>50%)
- Fine-tune hyperparameters
- Add Graph Neural Network module
- Implement graph matching

### If V2 Fails (<40%)
- Try contrastive learning (Plan 2)
- Try CUB keypoint supervision (Plan 3)

---

*Updated: 2025-11-29*
