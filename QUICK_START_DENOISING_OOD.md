# 去噪网络OOD检测 - 快速开始

## 🚀 快速开始

### 1. 训练去噪模型

```bash
python meanflow/train_denoising.py \
    --data_path data/RML2016.10a_dict_clean.pkl \
    --experiment_setting 1 \
    --epochs 500 \
    --batch_size 1024 \
    --eval_batch_size 2048 \
    --eval_freq 5
```

**训练期间会自动评估OOD检测性能**：
- 每5个epoch在验证集上评估
- 输出AUROC, AUPR, FPR@95等指标
- 按SNR和调制类型细分结果

### 2. 评估OOD检测性能

```bash
python meanflow/evaluate_ood_denoising.py \
    --checkpoint outputs/denoising/best_model.pth \
    --data_path data/RML2016.10a_dict_clean.pkl \
    --experiment_setting 1 \
    --method min_error \
    --batch_size 2048
```

## 📊 输出示例

### 训练时的评估输出

```
============================================================
Denoising & OOD Detection Evaluation Results
============================================================

Denoising Metrics:
  Overall MSE: 0.003456
  SNR Improvement: 3.45 dB
  Correlation: 0.9234
  PSNR: 28.45 dB

Classification Metrics:
  Closed-set Accuracy: 0.8765

OOD Detection Metrics:
  AUROC: 0.9234
  AUPR: 0.8976
  FPR@95: 0.1234
  Mean OOD Score (Known): 0.002345
  Mean OOD Score (Unknown): 0.008765

Per-Unknown-Class OOD Detection:
  AM-SSB:
    AUROC: 0.9156
    AUPR: 0.8845
    Mean Score: 0.008234
  GFSK:
    AUROC: 0.9312
    AUPR: 0.9107
    Mean Score: 0.009296

Per-SNR Results:
  SNR -20 dB: MSE=0.005678, AUROC=0.8567
  SNR  -10 dB: MSE=0.004123, AUROC=0.8890
  SNR    0 dB: MSE=0.002789, AUROC=0.9123
  SNR   10 dB: MSE=0.001567, AUROC=0.9456
  SNR   20 dB: MSE=0.001234, AUROC=0.9567
============================================================
```

## 🔍 核心原理

### OOD检测机制

```python
# 对每个测试样本
for sample in test_data:
    # 尝试用每个已知类别去噪
    min_error = infinity
    for class_id in known_classes:
        denoised = model.denoise(sample, class_id)
        error = mse(denoised, clean_signal)
        min_error = min(min_error, error)
    
    # 使用最小误差作为OOD分数
    if min_error > threshold:
        predict_as_OOD()
    else:
        predict_as_known_class()
```

### 为什么有效？

| 类型 | 去噪效果 | MSE | OOD分数 |
|------|---------|-----|---------|
| **Known类别** | 模型学习过，去噪效果好 | 低 | 低（In-Distribution） |
| **Unknown类别** | 模型未见过，去噪效果差 | 高 | 高（Out-of-Distribution） |

## 📈 指标解读

### AUROC (Area Under ROC Curve)
- **范围**: 0-1，越高越好
- **含义**: 随机选一个known和一个unknown样本，模型正确排序的概率
- **基准**: > 0.9 优秀，> 0.8 良好，> 0.7 可接受

### AUPR (Area Under Precision-Recall)
- **范围**: 0-1，越高越好
- **含义**: 考虑类别不平衡的综合指标
- **优势**: 在unknown样本较少时比AUROC更稳定

### FPR@95 (False Positive Rate at 95% TPR)
- **范围**: 0-1，越低越好
- **含义**: 检测出95%的unknown时，误报known为unknown的比例
- **目标**: < 0.1 优秀，< 0.2 良好

## 🎯 与Modulation任务对比

| 特性 | Denoising | Modulation |
|------|-----------|------------|
| **主要任务** | 信号去噪 | 调制分类 |
| **OOD机制** | 重建误差 | Energy score |
| **训练损失** | MSE loss | Energy-based loss |
| **推理策略** | 尝试所有类别，选最小MSE | 尝试所有类别，选最小energy |
| **OOD分数** | 最小MSE（高=OOD） | Energy score（高=OOD） |
| **计算开销** | K次推理 | 1次推理（per-class energy） |
| **优势** | 无需负样本训练 | 显式OOD训练 |

## 🔧 实际应用

### 在代码中使用

```python
import torch
from meanflow.models.meanflow_denoising import MeanFlowDenoising

# 加载训练好的模型
checkpoint = torch.load('outputs/denoising/best_model.pth')
model = create_model(args)
model.load_state_dict(checkpoint['ema_state_dict'])  # 使用EMA权重
model.eval()

# 对新样本进行OOD检测
with torch.no_grad():
    noisy_signal = ...  # [batch, 2, 128]
    clean_signal = ...  # [batch, 2, 128] (如果有ground truth)
    
    # 尝试所有类别，找最小MSE
    min_errors = []
    for class_id in range(num_classes):
        labels = torch.full((batch_size,), class_id)
        denoised = model.denoise(noisy_signal, labels, num_steps=1)
        error = F.mse_loss(denoised, clean_signal, reduction='none').mean(dim=(1,2))
        min_errors.append(error)
    
    ood_scores = torch.stack(min_errors).min(dim=0)[0]
    
    # 设置阈值（可通过验证集调优）
    threshold = 0.005  # 示例值
    is_ood = ood_scores > threshold
    
    print(f"OOD detection results: {is_ood}")
```

### 阈值选择

1. **基于验证集FPR**：
   ```python
   # 在验证集的known样本上设置5% FPR
   threshold = np.percentile(known_scores, 95)
   ```

2. **基于F1-score**：
   ```python
   # 在验证集上找最大化F1的阈值
   best_threshold = find_threshold_max_f1(val_scores, val_labels)
   ```

3. **SNR自适应**：
   ```python
   # 不同SNR使用不同阈值
   thresholds = {
       -20: 0.010,
       -10: 0.007,
         0: 0.005,
        10: 0.003,
        20: 0.002
   }
   ```

## 📝 实验设置说明

### Experiment Setting 1
- **Known (9类)**: 8PSK, AM-DSB, BPSK, CPFSK, PAM4, QAM16, QAM64, QPSK, WBFM
- **Unknown (2类)**: AM-SSB, GFSK

### 数据划分
- **训练集**: 80% known类别（144,000样本）
- **验证集**: 10% known + 50% unknown（38,000样本）
- **测试集**: 10% known + 50% unknown（38,000样本）

### SNR范围
- 默认: -20 dB to +20 dB
- 步长: 2 dB
- 共21个SNR点

## ⚡ 性能优化建议

### 1. 批量并行（推荐）
```python
# 不要循环，而是一次处理所有类别
batch_size, num_classes = signals.shape[0], 9
signals_expanded = signals.unsqueeze(1).expand(-1, num_classes, -1, -1)
labels_all = torch.arange(num_classes).unsqueeze(0).expand(batch_size, -1)
# 批量去噪并计算误差
```

### 2. 早停策略
```python
# 如果已经找到MSE < 某个很低的值，可以停止尝试其他类别
if min_error < 0.001:
    break
```

### 3. 只在推理时使用
- 训练阶段：使用真实标签，单次推理
- 验证/测试：尝试所有类别，K次推理

## 🎓 相关文件

- `meanflow/train_denoising.py`: 训练脚本（包含OOD评估）
- `meanflow/evaluate_ood_denoising.py`: 独立OOD评估脚本
- `OOD_DETECTION_WITH_DENOISING.md`: 详细原理说明
- `DENOISING_OOD_IMPROVEMENTS.md`: 改进总结
- `meanflow/models/meanflow_denoising.py`: 去噪模型实现
- `meanflow/data/rml_dataset.py`: 数据集加载器

## ❓ FAQ

**Q: 为什么不直接用真实标签去噪？**
A: 评估时模拟真实场景：不知道样本属于哪个类别（或是否OOD），所以尝试所有可能的类别。

**Q: 计算开销会不会太大？**
A: 验证/测试时确实需要K次推理，但可以通过批量并行优化。训练时仍使用真实标签，单次推理。

**Q: 能否加速推理？**
A: 可以！将模型改为一次输出所有类别的去噪结果，类似modulation的per-class energy。

**Q: OOD检测准确率如何？**
A: 取决于数据和模型质量。通常AUROC > 0.85表示效果不错。可以参考论文中的baseline。

**Q: 能否结合energy-based方法？**
A: 可以！将MSE score和energy score融合可能获得更好的OOD检测性能。



