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

### 场景 5: 高质量图片（用于论文/报告）

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

运行后会在输出目录生成：

```
outputs/visualizations/
├── sample_00_QPSK_snr10.png      # 单个样本详细图（样本0）
├── sample_01_8PSK_snr5.png       # 单个样本详细图（样本1）
├── sample_02_QAM16_snr-5.png     # 单个样本详细图（样本2）
├── ...
├── sample_09_BPSK_snr15.png      # 单个样本详细图（样本9）
└── summary_all_samples.png       # 所有样本汇总图
```

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

所有样本的I通道对比：

```
┌────────┬────────┬──────────┐
│ Noisy  │ Clean  │ Denoised │ ← 样本0
├────────┼────────┼──────────┤
│ Noisy  │ Clean  │ Denoised │ ← 样本1
├────────┼────────┼──────────┤
│   ...  │  ...   │   ...    │
└────────┴────────┴──────────┘
```

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

```bash
# 1. 训练模型（使用25% masking）
python -m meanflow.train_denoising \
    --data_path data/RML2016_denoising.pkl \
    --experiment_setting 1 \
    --mask_ratio 0.25 \
    --epochs 100 \
    --output_dir outputs/denoising

# 2. 可视化结果
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

完成！🎉

# 去噪和重建可视化

这个文档介绍如何使用 `visualize_denoising_reconstruction.py` 脚本来可视化去噪和重建的结果。

## 功能概述

该脚本会从验证集中选取样本，并为每个样本生成可视化图表，展示：

1. **原始带噪数据**（被mask的输入）- 红色阴影标注mask区域
2. **干净数据**（Ground Truth）- 黄色阴影标注原始mask位置
3. **去噪+重建后的数据** - 橙色阴影标注重建的mask区域

每个样本会生成两种可视化：
- 单独的详细图表（6个子图：I/Q通道分别显示三种信号）
- 汇总图表（所有样本的I通道对比）

## 使用方法

### 基本用法

```bash
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --data_path data/RML2016_denoising.pkl \
    --experiment_setting 1 \
    --mask_ratio 0.25 \
    --num_samples 10 \
    --output_dir outputs/visualizations
```

### 参数说明

#### 必需参数

- `--checkpoint`: 训练好的模型checkpoint路径
  - 例如: `outputs/denoising/best_model.pth`

#### 数据参数

- `--data_path`: 去噪数据集路径
  - 默认: `data/RML2016_denoising.pkl`

- `--experiment_setting`: 实验设置（1-12）
  - 默认: `1`
  - 必须与训练时使用的设置一致

- `--mask_ratio`: 训练时使用的mask比例
  - 默认: `0.25` (25%的信号长度)
  - **必须与训练时使用的mask_ratio一致**
  - 范围: 0.0 到 1.0

#### 可视化参数

- `--num_samples`: 要可视化的样本数量
  - 默认: `10`
  - 建议: 5-20个样本

- `--output_dir`: 输出图表的目录
  - 默认: `outputs/visualizations`

- `--dpi`: 保存图片的DPI
  - 默认: `150`
  - 建议: 150-300

- `--figsize`: 图片尺寸（宽度 高度），单位英寸
  - 默认: `20 12`

#### 系统参数

- `--device`: 使用的设备
  - 默认: 自动检测（有CUDA用cuda，否则用cpu）

- `--seed`: 随机种子（用于样本选择）
  - 默认: `42`

- `--batch_size`: 数据加载批次大小
  - 默认: `128`

- `--num_workers`: 数据加载线程数
  - 默认: `4`

## 使用示例

### 示例 1: 可视化10个样本（25% masking）

```bash
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/best_model.pth \
    --mask_ratio 0.25 \
    --num_samples 10
```

### 示例 2: 可视化20个样本（15% masking，高分辨率）

```bash
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/checkpoint_epoch_100.pth \
    --mask_ratio 0.15 \
    --num_samples 20 \
    --dpi 300 \
    --output_dir outputs/viz_high_res
```

### 示例 3: 不同实验设置

```bash
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/exp3_best.pth \
    --experiment_setting 3 \
    --mask_ratio 0.30 \
    --num_samples 15
```

### 示例 4: 无masking的去噪可视化

```bash
python -m meanflow.visualize_denoising_reconstruction \
    --checkpoint outputs/denoising/no_mask_best.pth \
    --mask_ratio 0.0 \
    --num_samples 10
```

## 输出文件

脚本会在输出目录中生成以下文件：

### 单样本详细图表

- 文件名格式: `sample_XX_MODULATION_snrYY.png`
- 例如: `sample_00_QPSK_snr10.png`, `sample_01_8PSK_snr-5.png`
- 内容: 6个子图展示I/Q通道的noisy/clean/denoised信号

### 汇总图表

- 文件名: `summary_all_samples.png`
- 内容: 所有样本的I通道对比（3列：noisy/clean/denoised）
- 便于快速对比多个样本的效果

## 可视化说明

### 颜色编码

- **蓝色**: Noisy信号的I通道
- **红色**: Noisy信号的Q通道
- **绿色**: Clean信号的I通道
- **品红色**: Clean信号的Q通道
- **青色**: Denoised信号的I通道
- **黄色**: Denoised信号的Q通道

### 阴影标注

- **红色阴影**: Noisy输入中的mask区域（已被置零）
- **黄色阴影**: Clean信号中标注原始mask位置（仅为参考）
- **橙色阴影**: Denoised信号中的重建区域（模型重建的部分）

## 注意事项

1. **mask_ratio必须一致**: 
   - 可视化时的`--mask_ratio`必须与训练时使用的一致
   - 否则mask区域检测会不准确

2. **仅显示ID样本**:
   - 脚本会跳过OOD样本（label=-1）
   - 只可视化已知调制类型的样本
   - 便于清晰展示模型在已知类上的表现

3. **Mask检测**:
   - 脚本通过检测连续的零值区域来识别mask
   - 如果信号本身包含很多零值，可能检测不准确

4. **内存使用**:
   - 大量样本或高DPI会占用较多内存
   - 建议每次可视化不超过20个样本

## 解读可视化结果

### 良好的去噪效果

- Denoised信号应该接近Clean信号
- Mask区域应该被成功重建（橙色区域）
- 非mask区域的噪声应该被去除

### 检查点

1. **噪声去除**: 比较noisy和denoised的非mask区域
2. **重建质量**: 比较denoised和clean的mask区域
3. **保真度**: 整体信号形状应保持一致

## 故障排除

### 问题: 找不到checkpoint文件

```
解决: 确保checkpoint路径正确，使用绝对路径或正确的相对路径
```

### 问题: Mask区域检测失败

```
解决: 
1. 确认mask_ratio与训练时一致
2. 检查数据集是否正确应用了masking
3. 查看是否使用了normalize（可能影响零值检测）
```

### 问题: 没有找到样本

```
解决:
1. 验证集可能只有OOD样本
2. 增加batch_size以加快搜索
3. 检查experiment_setting是否正确
```

### 问题: 图片太小/太大

```
解决: 调整--figsize参数，例如 --figsize 16 10
```

## 与训练脚本的对应关系

可视化脚本会自动从checkpoint中读取训练参数：

| 训练参数 | 可视化参数 | 说明 |
|---------|-----------|------|
| `--experiment_setting` | `--experiment_setting` | 必须一致 |
| `--mask_ratio` | `--mask_ratio` | 必须一致 |
| `--model_channels` | 自动读取 | 从checkpoint读取 |
| `--num_blocks` | 自动读取 | 从checkpoint读取 |
| `--dropout` | 自动读取 | 从checkpoint读取 |

## 高级用法

### 批量可视化多个checkpoint

```bash
for ckpt in outputs/denoising/checkpoint_epoch_*.pth; do
    epoch=$(basename $ckpt .pth | grep -oE '[0-9]+')
    python -m meanflow.visualize_denoising_reconstruction \
        --checkpoint $ckpt \
        --mask_ratio 0.25 \
        --num_samples 5 \
        --output_dir outputs/viz_epoch_$epoch
done
```

### 可视化不同SNR范围的样本

目前脚本从所有SNR中随机选择。如需特定SNR，可修改脚本添加SNR过滤。

## 参考

- 训练脚本: `meanflow/train_denoising.py`
- Masked Autoencoder文档: `MASKED_AUTOENCODER_TRAINING.md`
- 数据集实现: `meanflow/data/rml_dataset.py`

