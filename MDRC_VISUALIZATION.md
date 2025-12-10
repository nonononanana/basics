# MDRC 可视化功能说明

## 概述

在计算 MDRC (Multi-Domain Residual Complexity) 时，现在可以自动保存可视化图片，帮助分析每个类别的去噪效果和残差特征。

## 功能特性

### 自动生成的图片

对每个类别（共11个），会生成3张图片：

1. **去噪前图片** (`class_XX_noisy.png`)
   - 时域波形（I/Q通道）
   - I/Q星座图

2. **去噪后图片** (`class_XX_denoised.png`)
   - 时域波形（I/Q通道）
   - I/Q星座图

3. **残差图片** (`class_XX_residual.png`)
   - 时域波形（I/Q通道）
   - I/Q星座图
   - 残差幅度
   - 残差功率谱密度（PSD）

**总计：33张图片**

### 输出目录结构

```
output_dir/
└── eval_mdrc/
    ├── class_00_noisy.png
    ├── class_00_denoised.png
    ├── class_00_residual.png
    ├── class_01_noisy.png
    ├── class_01_denoised.png
    ├── class_01_residual.png
    ├── ...
    ├── class_10_noisy.png
    ├── class_10_denoised.png
    └── class_10_residual.png
```

## 使用方法

### 基本用法

```bash
python -m meanflow.evaluate_ood_denoising \
    --checkpoint model_checkpoint.pt \
    --data_path /path/to/data \
    --experiment_setting 1 \
    --method mdrc \
    --save_visualization \
    --output_dir ./results
```

### 参数说明

- `--save_visualization`: 启用可视化功能（默认不启用）
- `--output_dir`: 指定输出目录（默认为当前目录）

### 不保存可视化

如果不需要可视化，只需省略 `--save_visualization` 参数：

```bash
python -m meanflow.evaluate_ood_denoising \
    --checkpoint model_checkpoint.pt \
    --data_path /path/to/data \
    --experiment_setting 1 \
    --method mdrc
```

## 技术细节

### 残差计算

在第449行代码中：
```python
residual = input_sig - denoised_sig  # [batch, 2, length]
```

这个残差包含了去噪前后的差异，可以反映：
- 模型对该类别信号的理解程度
- 噪声去除的效果
- 可能存在的分类错误

### 可视化内容

#### 残差图片的四个子图

1. **时域波形**：显示I和Q通道的残差随时间的变化
2. **I/Q星座图**：显示残差在复平面上的分布
3. **残差幅度**：显示残差信号的包络
4. **功率谱密度**：显示残差信号的频域特性

这些图片帮助理解：
- ID（已知类别）样本应该产生较小且规律的残差
- OOD（未知类别）样本会产生较大且不规律的残差

## 性能考虑

- 可视化仅在第一个批次执行，使用该批次的第一个样本
- 对评估速度影响很小
- 图片以100 DPI保存，文件大小适中

## 示例分析

### 正常（ID）情况
- 去噪前：信号被噪声污染
- 去噪后：恢复出清晰的调制模式
- 残差：主要是随机噪声，没有明显的结构

### 异常（OOD）情况
- 去噪前：未知调制类型的信号
- 去噪后：无法完全恢复，可能产生扭曲
- 残差：包含结构化的信息，不是纯粹的噪声

## 注意事项

1. 需要安装 matplotlib 库
2. 可视化功能仅在使用 `mdrc` 方法时生效
3. 图片保存在 `<output_dir>/eval_mdrc/` 目录下
4. 如果目录不存在，会自动创建
5. 文件名格式：`class_<编号>_<类型>.png`

## 相关代码

- 主函数：`compute_ood_score_mdrc()` (第686行)
- 可视化函数：`save_mdrc_visualization()` (第434行)
- 残差计算：`extract_residual_features()` (第554行)

