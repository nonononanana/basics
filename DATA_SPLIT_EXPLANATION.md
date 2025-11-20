# 数据划分详细说明

## 📊 你的需求 vs 实现

### 你想要的数据划分

```
Known Classes (已知类别):
┌────────────────────────────────────────────────────────────┐
│ 每个 known class 的全部数据 (例如：1000 samples)           │
├────────────────┬──────────┬──────────┐                      │
│   Training     │   Val    │   Test   │                      │
│     80%        │   10%    │   10%    │                      │
│   (800)        │  (100)   │  (100)   │                      │
└────────────────┴──────────┴──────────┘                      │
                                                               │
Unknown Classes (未知类别):                                     │
┌────────────────────────────────────────────────────────────┤
│ 每个 unknown class 的全部数据 (例如：1000 samples)         │
├────────────────┬──────────────────────┬───────────────────┤ │
│   Training     │        Val           │       Test        │ │
│      0%        │        50%           │       50%         │ │
│     (0)        │       (500)          │      (500)        │ │
└────────────────┴──────────────────────┴───────────────────┘ │
```

### ✅ 现在的实现

代码现在**完全按照你的需求**实现：

```python
# meanflow/data/rml_dataset.py (Line 185-221)

if not is_unk:
    # Known class: 80% train, 10% val, 10% test
    n_train = int(n_samples * 0.8)   # 800
    n_val = int(n_samples * 0.1)     # 100
    # test gets remaining                100
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
else:
    # Unknown class: 0% train, 50% val, 50% test
    n_val = int(n_samples * 0.5)     # 500
    # test gets remaining                500
    
    train_indices = []  # No training
    val_indices = indices[:n_val]
    test_indices = indices[n_val:]
```

## 📈 实际数据量示例

### Experiment Setting 1 (9 known, 2 unknown classes)

假设每个 (modulation, SNR) 有 1000 个样本，21 个 SNR 级别：

#### Known Classes (9 个类别)
| 类别 | Training | Validation | Test | Total |
|------|----------|------------|------|-------|
| 每个类别 | 800 × 21 SNR<br>= 16,800 | 100 × 21 SNR<br>= 2,100 | 100 × 21 SNR<br>= 2,100 | 21,000 |
| **9个类别总计** | **151,200** | **18,900** | **18,900** | **189,000** |

#### Unknown Classes (2 个类别)
| 类别 | Training | Validation | Test | Total |
|------|----------|------------|------|-------|
| 每个类别 | 0 × 21 SNR<br>= 0 | 500 × 21 SNR<br>= 10,500 | 500 × 21 SNR<br>= 10,500 | 21,000 |
| **2个类别总计** | **0** | **21,000** | **21,000** | **42,000** |

#### 汇总
| Split | Known | Unknown | Total | Known:Unknown 比例 |
|-------|-------|---------|-------|--------------------|
| **Training** | 151,200 | 0 | 151,200 | 100:0 |
| **Validation** | 18,900 | 21,000 | **39,900** | **47:53** ✨ |
| **Test** | 18,900 | 21,000 | **39,900** | **47:53** ✨ |

### ✨ 关键优势

1. **Validation 和 Test 中 known 和 unknown 样本数接近平衡**
   - Known: 18,900 samples (47%)
   - Unknown: 21,000 samples (53%)
   
2. **充足的 unknown 样本用于 OOD 评估**
   - 每个 unknown class 有 10,500 个样本用于 validation
   - 可靠的 AUROC/AUPR 计算

3. **训练集完全不泄露 unknown 信息**
   - Training 只包含 known classes
   - Unknown classes 完全保留用于评估

## 🔍 为什么 Unknown 用 50/50 而不是 10/10？

### ❌ 如果 Unknown 也用 10/10 (旧方案)
```
Known:   80% train + 10% val + 10% test = 100%
Unknown:  0% train + 10% val + 10% test = 20% (浪费 80%!)

结果：
- Validation: 18,900 known + 4,200 unknown = 23,100 total
- 比例: 82% known, 18% unknown (不平衡!)
```

### ✅ Unknown 用 50/50 (新方案)
```
Known:   80% train + 10% val + 10% test = 100%
Unknown:  0% train + 50% val + 50% test = 100% (全部利用!)

结果：
- Validation: 18,900 known + 21,000 unknown = 39,900 total
- 比例: 47% known, 53% unknown (更平衡!)
```

## 🎯 对你的影响

### Training 阶段
```python
# 只看到 known classes
for batch in train_loader:
    pos_samples, neg_samples, labels, info = batch
    # labels: 0-8 (9 个 known classes)
    # info['is_unknown']: 全部为 False
```

### Validation 阶段
```python
# 看到 known + unknown (平衡的比例)
for batch in val_loader:
    pos_samples, neg_samples, labels, info = batch
    # labels: 0-8 for known, -1 for unknown
    # info['is_unknown']: ~53% 为 True
    
    # 可以计算有意义的 AUROC:
    val_auroc = compute_auroc(energies, is_unknown)
    # 因为有充足的 unknown 样本！
```

### Test 阶段
```python
# 和 validation 相同的分布
for batch in test_loader:
    pos_samples, neg_samples, labels, info = batch
    # 同样 ~47% known, ~53% unknown
```

## 📊 可视化对比

### 每个 Known Class 的数据流向
```
1000 samples (per class, per SNR)
│
├─ 80% (800) ──→ Training   ✅ 用于学习
├─ 10% (100) ──→ Validation ✅ 用于模型选择
└─ 10% (100) ──→ Test       ✅ 用于最终评估
```

### 每个 Unknown Class 的数据流向
```
1000 samples (per class, per SNR)
│
├─  0% (0)   ──→ Training   ❌ 不使用（防止泄露）
├─ 50% (500) ──→ Validation ✅ 用于 OOD 检测评估
└─ 50% (500) ──→ Test       ✅ 用于最终 OOD 评估
```

## ✅ 验证

运行测试确认数据分布：
```bash
python test_validation_ood.py
```

预期输出：
```
Validation Set Analysis:
Total samples: ~39,900
Known samples: ~18,900 (47%)
Unknown samples: ~21,000 (53%)

✅ PASS: Validation set correctly includes unknown classes
   Known/Unknown ratio: 18900:21000
   (Known uses 10% per class, Unknown uses 50% per class)
```

## 🚀 使用建议

1. **监控 `val/auroc`** - 现在有充足的 unknown 样本，AUROC 更可靠
2. **检查 energy 分离** - `val/mean_energy_known` vs `val/mean_energy_unknown`
3. **Per-class 分析** - 每个 unknown class 都有 10,500 个样本，可以单独评估
4. **Early stopping** - 基于 `val/auroc` 而不是 `val/closed_set_accuracy`

---

**总结：现在的实现完全符合你的需求！** ✨
- Known: 80% 训练，10% 验证，10% 测试
- Unknown: 0% 训练，50% 验证，50% 测试
- Validation 和 Test 中 known 和 unknown 样本数接近平衡

