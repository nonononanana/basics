# Mean Flow去噪模型流程说明

## 流的定义

在去噪任务中，Mean Flow的流定义如下：

### 训练阶段

**流的起点和终点：**
- **t=0 (终点)**: 干净信号 `x_clean`
- **t=1 (起点)**: 带噪信号 `x_noisy`

**插值路径：**
```python
z_t = (1 - t) * x_clean + t * x_noisy
```

- 当 t=0 时：`z_0 = x_clean` (干净信号)
- 当 t=1 时：`z_1 = x_noisy` (带噪信号)
- 当 t ∈ (0,1) 时：`z_t` 是两者的线性插值

**速度场：**
```python
v = x_noisy - x_clean  # 指向从clean到noisy的方向
```

**网络学习的目标：**
网络 `u(z_t, t)` 学习预测速度场，使得沿着流从 `x_noisy` (t=1) 移动到 `x_clean` (t=0)。

### 推理阶段（去噪）

**单步去噪（推荐）：**
```python
# 输入：带噪信号 x_noisy
z_1 = x_noisy  # 从t=1的位置开始

# 预测速度场
u = net(z_1, t=1, h=1, class_labels=modulation_type)

# 沿流移动到t=0，得到干净信号
x_clean = z_1 - u
```

**多步去噪（可选）：**
```python
z = x_noisy
dt = 1.0 / num_steps

for step in range(num_steps):
    t = 1.0 - step * dt
    u = net(z, t, class_labels=modulation_type)
    z = z - dt * u  # Euler积分

x_clean = z
```

## 条件信息

- **调制类型条件**: 通过 `class_labels` 参数传入，用于指定信号的调制方式
- **不使用SNR条件**: 模型不依赖SNR信息，只条件于调制类型

## 关键修正

之前的错误实现：
```python
# 错误：从随机噪声e开始，x_noisy作为额外conditioning
z = (1 - t) * x_clean + t * e  # e是随机噪声
```

修正后的实现：
```python
# 正确：从x_noisy开始，直接学习去噪流
z = (1 - t) * x_clean + t * x_noisy
```

这样确保了：
1. 训练时学习的是从noisy到clean的流
2. 推理时从实际的noisy信号出发
3. 流的起点和终点在训练和推理时保持一致

## 使用示例

```python
from meanflow.models.meanflow_denoising import MeanFlowDenoising

# 创建模型
model = MeanFlowDenoising(...)

# 训练
loss_dict = model.forward_with_loss(
    x_noisy=noisy_signals,
    x_clean=clean_signals,
    class_labels=modulation_labels
)

# 推理去噪
model.eval()
denoised = model.denoise(
    x_noisy=test_noisy_signals,
    class_labels=test_modulation_labels,
    num_steps=1  # 单步去噪
)
```

