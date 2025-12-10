# MDRC可视化快速开始

## 快速使用

### 启用可视化（生成33张图片）

```bash
python -m meanflow.evaluate_ood_denoising \
    --checkpoint your_model.pt \
    --data_path /path/to/data \
    --method mdrc \
    --save_visualization \
    --output_dir ./results
```

### 不使用可视化（更快）

```bash
python -m meanflow.evaluate_ood_denoising \
    --checkpoint your_model.pt \
    --data_path /path/to/data \
    --method mdrc
```

## 输出结果

启用 `--save_visualization` 后，会在 `<output_dir>/eval_mdrc/` 创建：

```
eval_mdrc/
├── class_00_noisy.png      # 类别0：去噪前
├── class_00_denoised.png   # 类别0：去噪后
├── class_00_residual.png   # 类别0：残差
├── class_01_noisy.png      # 类别1：去噪前
├── class_01_denoised.png   # 类别1：去噪后
├── class_01_residual.png   # 类别1：残差
...
└── class_10_residual.png   # 类别10：残差
```

**总计：33张图片**（11个类别 × 3张图片/类别）

## 图片内容

### 1. 去噪前 (noisy)
- I/Q时域波形
- I/Q星座图

### 2. 去噪后 (denoised)  
- I/Q时域波形
- I/Q星座图

### 3. 残差 (residual)
- I/Q时域波形
- I/Q星座图
- 残差幅度
- 功率谱密度

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--save_visualization` | 启用可视化 | False |
| `--output_dir` | 输出目录 | `.` (当前目录) |
| `--method` | 必须是 `mdrc` | - |

## 注意事项

1. 可视化只在使用 `mdrc` 方法时生效
2. 使用测试集第一个批次的第一个样本
3. 需要安装 matplotlib
4. 对性能影响很小（只保存一次）

## 更多信息

详细文档请参考：[MDRC_VISUALIZATION.md](MDRC_VISUALIZATION.md)

