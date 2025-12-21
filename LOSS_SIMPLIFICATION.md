# Loss简化修改

## 修改内容

**删除了冗余的 `denoising_loss`，统一为单一Mean Flow loss**

### 代码对比

```python
# 修改前 ❌
flow_loss = (u_pred - u_tgt)**2
denoising_loss = MSE(x_noisy - u, x_clean)
total_loss = flow_loss + 0.5 * denoising_loss

# 修改后 ✅  
loss = (u_pred - u_tgt)**2
total_loss = loss
```

## 原因

1. Denoising有2个loss，Modulation只有1个 → 不一致
2. Mean Flow本身就能学会完整轨迹 → 不需要额外监督
3. 你说的对：直接在x_noisy到x_clean之间采样就够了

## 逻辑

**训练**：在所有时间点 t∈[0,1] 学习速度场
- t=0: clean
- t=1: noisy
- 随机采样t，学习 `u(z_t, t)`

**推理**：从noisy走到clean
```python
u = net(x_noisy, t=1, h=1, class_labels)
x_clean = x_noisy - u
```

## 影响

- ✅ 与modulation版本逻辑一致
- ✅ 代码减少26行
- ⚠️ 旧checkpoint不兼容，需重新训练

---

**修改日期**: 2024-12-03






