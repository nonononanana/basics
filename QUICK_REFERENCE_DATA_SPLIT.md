# 🎯 数据划分快速参考

## 核心规则

```
Known Classes:    80% train  +  10% val  +  10% test  = 100%
Unknown Classes:   0% train  +  50% val  +  50% test  = 100%
```

## 示例：Experiment Setting 1

**9 known classes + 2 unknown classes**

每个类别每个 SNR 有 1000 个样本，共 21 个 SNR 级别：

| 数据集 | Known 样本数 | Unknown 样本数 | 总计 | 比例 |
|--------|--------------|----------------|------|------|
| **Training** | 151,200 | 0 | 151,200 | 100:0 |
| **Validation** | 18,900 | 21,000 | 39,900 | **47:53** ✨ |
| **Test** | 18,900 | 21,000 | 39,900 | **47:53** ✨ |

## 为什么这样设计？

### ✅ 优点

1. **训练不泄露 unknown**
   - Training 完全不包含 unknown classes
   - 模型在训练时只学习 known classes

2. **Validation 和 Test 平衡**
   - Known 和 Unknown 样本数接近 (47% vs 53%)
   - 更可靠的 AUROC/AUPR 计算
   - 充足的 unknown 样本用于 OOD 评估

3. **充分利用数据**
   - Unknown classes 的数据不浪费
   - 全部用于 validation 和 test (50/50 分配)

### ❌ 如果 Unknown 也用 10/10 (不推荐)

```
Validation: 18,900 known + 4,200 unknown = 23,100
比例: 82% known, 18% unknown (不平衡！)
浪费 80% 的 unknown 数据！
```

## 训练命令

```bash
# 无需任何特殊参数！
python meanflow/train_modulation.py \
    --experiment_setting 1 \
    --epochs 100 \
    --batch_size 256 \
    --data_path data/RML2016.10a_dict.pkl
```

## 关键指标

### Training
```
train/total_loss
train/reconstruction_loss
train/pos_energy_loss
train/neg_energy_loss
```

### Validation (⭐ 主要关注)
```
val/auroc                    # > 0.85 优秀
val/OA                       # Open-set accuracy
val/closed_set_accuracy      # Known classes 准确率
val/mean_energy_known        # 应该更负（更低）
val/mean_energy_unknown      # 应该更高（接近0）
```

### Test (最终评估)
```
test/auroc
test/OA
test/closed_set_accuracy
```

## 验证数据分布

```bash
# 快速验证逻辑
python test_val_logic.py

# 完整数据验证（需要数据文件）
python test_validation_ood.py

# 一键验证所有
./verify_changes.sh
```

## 常见问题

### Q: Validation 为什么有 53% unknown？
A: Unknown classes 不用于训练，所以 50/50 分配给 val/test，确保充足样本用于 OOD 评估。

### Q: 会不会泄露 unknown 信息？
A: 不会！Training 完全不包含 unknown。只在 validation 和 test 中使用 unknown 进行评估。

### Q: 为什么不是 80/10/10 for all？
A: 那样会浪费 80% 的 unknown 数据，且 val/test 中 unknown 样本太少，AUROC 不可靠。

## 文档

- **详细说明**: [DATA_SPLIT_EXPLANATION.md](DATA_SPLIT_EXPLANATION.md)
- **完整更新**: [VALIDATION_OOD_UPDATE.md](VALIDATION_OOD_UPDATE.md)
- **中文总结**: [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
- **主要 README**: [README_VALIDATION_CHANGES.md](README_VALIDATION_CHANGES.md)

---

**最后更新**: 2024-11-20  
**状态**: ✅ Tested and Verified

