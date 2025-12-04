# 使用去噪网络进行OOD检测

## 核心思想

去噪网络通过学习已知类别的信号特征，可以有效地进行OOD（Out-of-Distribution）检测：

1. **已知类别（Known Classes）**: 去噪误差低
   - 模型在训练时学习了这些调制类型的特征
   - 能够准确地从噪声中恢复出干净信号
   - 重建误差（MSE）较小

2. **未知类别（Unknown Classes）**: 去噪误差高
   - 模型从未见过这些调制类型
   - 无法准确恢复信号，重建结果不理想
   - 重建误差（MSE）较大

## OOD检测方法

### 方法1: 最小重建误差（推荐）

对于一个测试样本：
1. 用每个已知类别标签尝试去噪
2. 计算去噪后与原始噪声信号的重建误差
3. 取所有类别中的最小误差作为OOD分数
4. 如果最小误差 > 阈值，则判定为OOD

```python
def ood_score_min_reconstruction_error(model, noisy_signal, num_classes):
    """
    计算OOD分数（最小重建误差法）
    
    Args:
        model: 训练好的去噪模型
        noisy_signal: 噪声信号 [batch, 2, 128]
        num_classes: 已知类别数量
    
    Returns:
        ood_scores: OOD分数（越高越可能是OOD） [batch]
    """
    batch_size = noisy_signal.shape[0]
    min_errors = torch.full((batch_size,), float('inf'), device=noisy_signal.device)
    
    # 尝试每个已知类别
    for class_id in range(num_classes):
        # 使用当前类别标签去噪
        class_labels = torch.full((batch_size,), class_id, device=noisy_signal.device)
        denoised = model.denoise(noisy_signal, class_labels, num_steps=1)
        
        # 计算重建误差（per-sample MSE）
        errors = F.mse_loss(denoised, noisy_signal, reduction='none')
        errors = errors.mean(dim=(1, 2))  # [batch]
        
        # 更新最小误差
        min_errors = torch.minimum(min_errors, errors)
    
    return min_errors
```

### 方法2: 平均重建误差

对于一个测试样本：
1. 用所有已知类别标签去噪，取平均
2. 计算平均去噪结果与原始信号的误差
3. 误差越大，越可能是OOD

```python
def ood_score_avg_reconstruction_error(model, noisy_signal, num_classes):
    """
    计算OOD分数（平均重建误差法）
    
    Args:
        model: 训练好的去噪模型
        noisy_signal: 噪声信号 [batch, 2, 128]
        num_classes: 已知类别数量
    
    Returns:
        ood_scores: OOD分数（越高越可能是OOD） [batch]
    """
    batch_size = noisy_signal.shape[0]
    denoised_sum = torch.zeros_like(noisy_signal)
    
    # 对所有已知类别求平均
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, device=noisy_signal.device)
        denoised = model.denoise(noisy_signal, class_labels, num_steps=1)
        denoised_sum += denoised
    
    denoised_avg = denoised_sum / num_classes
    
    # 计算重建误差
    errors = F.mse_loss(denoised_avg, noisy_signal, reduction='none')
    errors = errors.mean(dim=(1, 2))
    
    return errors
```

### 方法3: 去噪改善度

比较去噪前后的信号质量改善：
- 已知类别：去噪应该能降低噪声，改善信号质量
- 未知类别：去噪可能无效甚至恶化信号

```python
def ood_score_improvement(model, noisy_signal, clean_signal, num_classes):
    """
    基于去噪改善度的OOD分数
    注意：这个方法需要知道真实的干净信号（仅用于验证/测试）
    
    Args:
        model: 训练好的去噪模型
        noisy_signal: 噪声信号 [batch, 2, 128]
        clean_signal: 干净信号 [batch, 2, 128]
        num_classes: 已知类别数量
    
    Returns:
        ood_scores: OOD分数（越高越可能是OOD） [batch]
    """
    batch_size = noisy_signal.shape[0]
    
    # 噪声前的误差
    noise_error = F.mse_loss(noisy_signal, clean_signal, reduction='none')
    noise_error = noise_error.mean(dim=(1, 2))
    
    # 去噪后的最小误差
    min_denoised_error = torch.full((batch_size,), float('inf'), device=noisy_signal.device)
    for class_id in range(num_classes):
        class_labels = torch.full((batch_size,), class_id, device=noisy_signal.device)
        denoised = model.denoise(noisy_signal, class_labels, num_steps=1)
        
        denoised_error = F.mse_loss(denoised, clean_signal, reduction='none')
        denoised_error = denoised_error.mean(dim=(1, 2))
        
        min_denoised_error = torch.minimum(min_denoised_error, denoised_error)
    
    # 改善度：负值表示去噪恶化了信号（很可能是OOD）
    improvement = noise_error - min_denoised_error
    
    # OOD分数：改善度越小（越负），OOD可能性越大
    ood_scores = -improvement
    
    return ood_scores
```

## 实际使用流程

### 1. 训练阶段
```bash
# 只在已知类别上训练去噪模型
python meanflow/train_denoising.py \
    --data_path data/RML2016.10a_dict_clean.pkl \
    --experiment_setting 1 \
    --epochs 500 \
    --batch_size 1024 \
    --eval_batch_size 2048
```

### 2. OOD检测阶段
```python
# 加载训练好的模型
model = load_checkpoint('outputs/denoising/best_model.pth')
model.eval()

# 对测试样本计算OOD分数
with torch.no_grad():
    ood_scores = ood_score_min_reconstruction_error(
        model, 
        test_noisy_signals, 
        num_classes=9  # experiment_setting 1 有9个已知类别
    )
    
    # 设置阈值（可以通过验证集调优）
    threshold = 0.01  # 示例阈值
    
    # OOD检测
    is_ood = ood_scores > threshold
```

### 3. 评估性能

使用标准OOD检测指标：
- **AUROC**: Area Under ROC Curve
- **AUPR**: Area Under Precision-Recall Curve  
- **FPR@95**: False Positive Rate at 95% True Positive Rate

```python
from sklearn.metrics import roc_auc_score, average_precision_score

# ground_truth: 0=known, 1=unknown
auroc = roc_auc_score(ground_truth, ood_scores)
aupr = average_precision_score(ground_truth, ood_scores)
```

## 优势

1. **无需额外训练**: 去噪网络本身就可以用于OOD检测
2. **可解释性强**: 基于重建误差，直观易懂
3. **不需要负样本**: 训练时只需要已知类别的干净/噪声对
4. **对噪声鲁棒**: 本身就是针对噪声信号设计的

## 注意事项

1. **阈值选择**: 需要通过验证集调优OOD检测阈值
2. **SNR影响**: 不同SNR下的重建误差范围不同，可能需要SNR自适应阈值
3. **计算开销**: 方法1需要对每个类别都推理一次，可以通过批处理优化

## 运行OOD评估脚本

```bash
python meanflow/evaluate_ood_denoising.py \
    --checkpoint outputs/denoising/best_model.pth \
    --data_path data/RML2016.10a_dict_clean.pkl \
    --experiment_setting 1 \
    --method min_error  # 可选: min_error, avg_error, improvement
```


