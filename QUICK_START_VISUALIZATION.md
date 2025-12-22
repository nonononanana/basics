# 快速开始：去噪可视化

快速指南，帮助你立即开始可视化去噪和重建结果。

## 最简单的使用方法

假设你已经训练了一个模型，checkpoint保存在 `outputs/denoising/best_model.pth`：

```bash
# 使用shell脚本（最简单）
./scripts/visualize_denoising.sh outputs/denoising/best_model.pth

# 或者直接使用Python
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25
```

就这么简单！结果会保存在 `outputs/visualizations/` 目录。

## 常见场景

### 场景 1: 使用默认参数可视化

```bash
# 使用shell脚本
./scripts/visualize_denoising.sh outputs/denoising/best_model.pth

# 这等同于：
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --data_path data/RML2016_denoising.pkl \
    --experiment_setting 1 \
    --mask_ratio 0.25 \
    --num_samples 10 \
    --output_dir outputs/visualizations
```

### 场景 2: 不同的mask ratio

如果训练时使用了 `--mask_ratio 0.15`：

```bash
# Shell脚本
./scripts/visualize_denoising.sh \
    outputs/denoising/best_model.pth \
    data/RML2016_denoising.pkl \
    1 \
    0.15

# Python
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.15
```

### 场景 3: 可视化更多样本

```bash
# 可视化20个样本
./scripts/visualize_denoising.sh \
    outputs/denoising/best_model.pth \
    data/RML2016_denoising.pkl \
    1 \
    0.25 \
    20
```

### 场景 4: 不同实验设置

```bash
# 实验设置3
./scripts/visualize_denoising.sh \
    outputs/denoising/exp3_best.pth \
    data/RML2016_denoising.pkl \
    3 \
    0.25 \
    10
```

### 场景 5: 无masking的纯去噪可视化

```bash
# 仅去噪（无重建）
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/no_mask_model.pth \
    --mask_ratio 0.0 \
    --num_samples 10
```

**注意**: 当 `mask_ratio=0.0` 时：
- 不会显示mask区域阴影
- 标题显示 "Denoising Only" 而非 "Denoising + Reconstruction"
- 仍然会显示10个ID + 10个OOD样本

### 场景 6: 高质量图片（用于论文/报告）

```bash
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --num_samples 10 \
    --dpi 300 \
    --output_dir outputs/viz_high_quality
```

## Shell脚本参数顺序

```bash
./scripts/visualize_denoising.sh [CHECKPOINT] [DATA_PATH] [EXP_SETTING] [MASK_RATIO] [NUM_SAMPLES] [OUTPUT_DIR]
```

| 位置 | 参数 | 默认值 | 说明 |
|-----|------|--------|------|
| 1 | CHECKPOINT | `outputs/denoising/best_model.pth` | 模型checkpoint路径 |
| 2 | DATA_PATH | `data/RML2016_denoising.pkl` | 数据集路径 |
| 3 | EXP_SETTING | `1` | 实验设置(1-12) |
| 4 | MASK_RATIO | `0.25` | Mask比例 |
| 5 | NUM_SAMPLES | `10` | 样本数量 |
| 6 | OUTPUT_DIR | `outputs/visualizations` | 输出目录 |

## 输出文件说明

运行后会在输出目录生成（以 `num_samples=10` 为例）：

```
outputs/visualizations/
├── sample_00_ID_QPSK_snr10.png      # ID样本详细图（样本0）
├── sample_01_ID_8PSK_snr5.png       # ID样本详细图（样本1）
├── sample_02_ID_QAM16_snr-5.png     # ID样本详细图（样本2）
├── ...
├── sample_09_ID_BPSK_snr15.png      # ID样本详细图（样本9）
├── sample_10_OOD_AM-DSB_snr8.png    # OOD样本详细图（样本10）
├── sample_11_OOD_WBFM_snr-2.png     # OOD样本详细图（样本11）
├── ...
├── sample_19_OOD_AM-SSB_snr12.png   # OOD样本详细图（样本19）
└── summary_all_samples.png          # 所有样本汇总图（20个）
```

**文件命名规则**: `sample_{序号}_{ID/OOD}_{调制类型}_snr{SNR值}.png`

- ID样本文件名包含 `_ID_`
- OOD样本文件名包含 `_OOD_`
- 便于快速识别样本类型

### 单样本详细图 (sample_XX_*.png)

6个子图布局：

```
┌─────────────────┬─────────────────┐
│ Noisy - I通道   │ Noisy - Q通道   │ ← 红色阴影=mask区域
├─────────────────┼─────────────────┤
│ Clean - I通道   │ Clean - Q通道   │ ← 黄色阴影=参考位置
├─────────────────┼─────────────────┤
│ Denoised - I    │ Denoised - Q    │ ← 橙色阴影=重建区域
└─────────────────┴─────────────────┘
```

### 汇总图 (summary_all_samples.png)

所有样本的I通道对比（包含ID和OOD）：

```
┌────────┬────────┬──────────┐
│ Noisy  │ Clean  │ Denoised │ ← 样本0 [ID]  (黑色标签)
├────────┼────────┼──────────┤
│ Noisy  │ Clean  │ Denoised │ ← 样本1 [ID]  (黑色标签)
├────────┼────────┼──────────┤
│   ...  │  ...   │   ...    │
├────────┼────────┼──────────┤
│ Noisy  │ Clean  │ Denoised │ ← 样本10 [OOD] (红色标签)
├────────┼────────┼──────────┤
│ Noisy  │ Clean  │ Denoised │ ← 样本11 [OOD] (红色标签)
├────────┼────────┼──────────┤
│   ...  │  ...   │   ...    │
└────────┴────────┴──────────┘
```

**汇总图特点**:
- 标题显示ID和OOD样本数量
- Y轴标签用**黑色**显示ID样本，**红色加粗**显示OOD样本
- 便于快速识别和对比ID/OOD性能

## 查看结果

### macOS

```bash
# 打开输出目录
open outputs/visualizations/

# 直接查看汇总图
open outputs/visualizations/summary_all_samples.png
```

### Linux

```bash
# 使用默认图片查看器
xdg-open outputs/visualizations/summary_all_samples.png

# 或使用特定程序
eog outputs/visualizations/summary_all_samples.png  # Eye of GNOME
```

### Windows

```bash
# 使用文件资源管理器
explorer outputs\visualizations\

# 或使用默认程序打开
start outputs\visualizations\summary_all_samples.png
```

## 常见问题

### Q: 怎么知道训练时用的mask_ratio是多少？

A: 查看训练日志或wandb记录。如果不确定，可以尝试不同的值：

```bash
# 尝试0.25
./scripts/visualize_denoising.sh outputs/denoising/best_model.pth data/RML2016_denoising.pkl 1 0.25

# 尝试0.15
./scripts/visualize_denoising.sh outputs/denoising/best_model.pth data/RML2016_denoising.pkl 1 0.15

# 无masking
./scripts/visualize_denoising.sh outputs/denoising/best_model.pth data/RML2016_denoising.pkl 1 0.0
```

### Q: 为什么找不到样本？

A: 可能的原因：
1. 验证集太小
2. 只有OOD样本（脚本跳过OOD）
3. 实验设置不匹配

解决方法：
```bash
# 增加batch_size加快搜索
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --batch_size 512  # 增加batch size
```

### Q: 图片太大/太小？

A: 调整figsize：

```bash
# 更大的图
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --figsize 24 14

# 更小的图
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --figsize 16 10
```

### Q: 需要更高质量的图片？

A: 增加DPI：

```bash
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --dpi 300  # 高质量，适合打印
```

## 🎲 随机样本选择 + ID/OOD对比

**新功能**: 脚本会同时可视化 **ID样本** 和 **OOD样本**！

### 重要更新

当设置 `--num_samples=10` 时，脚本会：
- 随机选择 **10个ID样本**（已知调制类型）
- 随机选择 **10个OOD样本**（未知调制类型）
- **总共可视化20个样本**

这样可以直接对比模型在已知类和未知类上的表现！

### 工作流程

1. **收集阶段**: 扫描整个验证集，分别收集ID和OOD样本
2. **随机选择**: 从ID和OOD中各随机选择指定数量的样本
3. **智能去噪**: 
   - ID样本：使用真实类别标签去噪
   - OOD样本：尝试所有已知类别，选择最佳重建结果
4. **多样性统计**: 分别显示ID和OOD样本的调制类型和SNR分布

### 优势

✅ **全面对比**: 同时看到模型在ID和OOD上的表现  
✅ **多样性**: 每次运行会看到不同的调制类型和SNR组合  
✅ **覆盖面**: 更容易发现不同条件下的模型表现  
✅ **可重复**: 使用 `--seed` 参数可以重现相同的选择  
✅ **智能标注**: 文件名和图表自动标注ID/OOD  

### 示例输出（有masking，mask_ratio=0.25）

```
Found 1234 ID samples and 567 OOD samples in validation set

Step 2: Randomly selecting samples...
  ID samples to select: 10
  OOD samples to select: 10

Selected sample diversity (Total: 20):

  ID Samples (10):
    Modulation types: 5
      8PSK: 2
      BPSK: 1
      QPSK: 3
      QAM16: 2
      QAM64: 2
    SNR distribution:
      Low SNR (<-5dB): 3
      Mid SNR (-5 to 10dB): 4
      High SNR (>10dB): 3

  OOD Samples (10):
    Modulation types: 3
      AM-DSB: 4
      WBFM: 3
      AM-SSB: 3
    SNR distribution:
      Low SNR (<-5dB): 2
      Mid SNR (-5 to 10dB): 5
      High SNR (>10dB): 3

Step 3: Processing and visualizing 20 samples...
  [1/20] ID  | QPSK     | SNR= 10dB | Mask:[ 45, 77)
  [2/20] ID  | 8PSK     | SNR= -5dB | Mask:[ 12, 44)
  ...
  [11/20] OOD | AM-DSB   | SNR=  5dB | Mask:[ 88,120)
  [12/20] OOD | WBFM     | SNR= 15dB | Mask:[ 23, 55)
  ...
```

### 示例输出（无masking，mask_ratio=0.0）

```
Found 1234 ID samples and 567 OOD samples in validation set

Step 2: Randomly selecting samples...
  ID samples to select: 10
  OOD samples to select: 10

Selected sample diversity (Total: 20):
  (同上)

Step 3: Processing and visualizing 20 samples...
  [1/20] ID  | QPSK     | SNR= 10dB | No mask      
  [2/20] ID  | 8PSK     | SNR= -5dB | No mask      
  ...
  [11/20] OOD | AM-DSB   | SNR=  5dB | No mask      
  [12/20] OOD | WBFM     | SNR= 15dB | No mask      
  ...
```

### 控制随机性

```bash
# 每次运行都不同（默认）
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --num_samples 10

# 使用固定种子（可重现结果）
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --num_samples 10 \
    --seed 123  # 相同的seed会选择相同的样本

# 获得完全不同的样本
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --num_samples 10 \
    --seed 456  # 不同的seed得到不同的样本
```

## 解读结果

### 好的结果特征

✅ Denoised信号接近Clean信号  
✅ Mask区域被成功重建（橙色部分）  
✅ 非mask区域的噪声被去除  
✅ 信号整体形状保持一致  

### 需要改进的特征

❌ Denoised信号与Clean差异大  
❌ Mask区域重建失败  
❌ 非mask区域仍有明显噪声  
❌ 信号形状发生扭曲  

## 下一步

1. **分析结果**: 查看可视化图表，评估模型性能
2. **调整训练**: 根据结果调整训练参数（mask_ratio, model_channels等）
3. **比较实验**: 可视化不同checkpoint，对比训练进度
4. **制作报告**: 使用高DPI图片制作论文/报告

## 完整文档

详细文档请参考：
- `VISUALIZATION_DENOISING.md` - 完整使用说明
- `MASKED_AUTOENCODER_TRAINING.md` - Masking训练详解
- `meanflow/train_denoising.py` - 训练脚本

## 示例工作流程

### 完整流程：训练 → 可视化

```bash
# 1. 训练模型（使用25% masking）
python -m meanflow.train_denoising \
    --data_path data/RML2016_denoising.pkl \
    --experiment_setting 1 \
    --mask_ratio 0.25 \
    --epochs 100 \
    --output_dir outputs/denoising

# 2. 可视化结果（10 ID + 10 OOD = 20个样本）
./scripts/visualize_denoising.sh \
    outputs/denoising/best_model.pth \
    data/RML2016_denoising.pkl \
    1 \
    0.25 \
    10 \
    outputs/viz_exp1

# 3. 查看结果
open outputs/viz_exp1/summary_all_samples.png
```

### 对比不同mask_ratio的效果

```bash
# 训练3个不同mask_ratio的模型
python -m meanflow.train_denoising --mask_ratio 0.0 --output_dir outputs/mask_0
python -m meanflow.train_denoising --mask_ratio 0.25 --output_dir outputs/mask_25
python -m meanflow.train_denoising --mask_ratio 0.50 --output_dir outputs/mask_50

# 分别可视化
python -m meanflow.visualize_denoising_reconstruction --checkpoint outputs/mask_0/best_model.pth --mask_ratio 0.0 --output_dir viz_mask_0
python -m meanflow.visualize_denoising_reconstruction --checkpoint outputs/mask_25/best_model.pth --mask_ratio 0.25 --output_dir viz_mask_25
python -m meanflow.visualize_denoising_reconstruction --checkpoint outputs/mask_50/best_model.pth --mask_ratio 0.50 --output_dir viz_mask_50

# 对比结果
open viz_mask_0/summary_all_samples.png
open viz_mask_25/summary_all_samples.png
open viz_mask_50/summary_all_samples.png
```

## 总结

✅ **完全兼容 `mask_ratio=0.0`**: 纯去噪可视化，无重建任务  
✅ **同时显示ID和OOD**: `num_samples=10` → 10 ID + 10 OOD = 20个样本  
✅ **随机多样化选择**: 每次运行看到不同调制类型和SNR组合  
✅ **智能标注**: 自动区分ID（黑色）和OOD（红色）样本  
✅ **灵活可控**: 使用 `--seed` 可重现结果  

完成！🎉
