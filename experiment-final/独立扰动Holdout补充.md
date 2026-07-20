# 独立扰动 Holdout 补充

## 协议

本测试在上一轮结果之后冻结，不再选择模型或调整扰动范围。对象路由固定为 Jar/Cellphone 使用 `wrist4`、USB stick 使用 baseline；集成固定为 seed 0/1/2 best checkpoint 的未裁剪 28 维确定性动作均值。

20 条逐轨迹扰动由 seed `20260720` 的独立 Latin-hypercube 维度生成，并被所有模型配对复用：对象水平平移 ±8 mm、竖直轴旋转 ±8°、摩擦系数 0.8–1.2、总质量 0.16–0.24 kg。manifest SHA256 为：

```text
8cec9bee7170c47782b8d6e19662f5698840b9958ed06449350c89fc788c020d
```

每个 `rollout_metrics.json` 都记录该哈希，每条 trajectory 同时记录实际扰动值，因而可以检查配对关系和复现实验。

## 结果

| 物体 | Seed 0 | Seed 1 | Seed 2 | 三模型集成 | 成员均值 |
|---|---:|---:|---:|---:|---:|
| Jar | 4/20 | 3/20 | 7/20 | 5/20（25%） | 23.33% |
| Cellphone | 3/20 | 3/20 | 4/20 | 5/20（25%） | 16.67% |
| USB stick | 3/20 | 1/20 | 4/20 | 2/20（10%） | 13.33% |
| Aggregate | 10/60（16.67%） | 7/60（11.67%） | 15/60（25%） | **12/60（20%）** | **32/180（17.78%）** |

![独立扰动 Holdout 对比](figures/perturbation_holdout_comparison.png)

在 60 个“物体×轨迹”配对样本上，集成相对三个成员的逐样本均值提高 2.22 个百分点；固定 seed 的 cluster bootstrap 95% 区间为 **[-3.33, 8.33] 个百分点**，包含 0。normalized lift 的配对差为 0.0061，95% 区间为 [-0.0474, 0.0643]，同样包含 0。

集成相对 seed 0 有 4 胜/2 负，相对 seed 1 有 5 胜/0 负，相对 seed 2 有 4 胜/7 负。集成成功的 12 个案例都至少被一个成员成功过；同时有 9 个“至少一个成员成功、集成失败”的案例。这说明动作均值主要压缩 seed 方差，并未产生新的成功模式。

## 真实回放

| Jar 成功 | Cellphone 失败 | USB stick 成功 |
|---|---|---|
| ![Jar holdout 成功](renders/perturbation_holdout/jar/success/success.gif) | ![Cellphone holdout 失败](renders/perturbation_holdout/cellphone/failure/failure.gif) | ![USB holdout 成功](renders/perturbation_holdout/usb_stick/success/success.gif) |

## 结论与下一步

独立 holdout 支持“集成优于平均成员和最坏 seed”的稳定化结论，但不支持“集成显著提升成功率”或“超过最佳成员”。因此集成仍应作为可选稳定部署路线，不替代公平实验策略，也不能宣传为统计显著提升。

下一轮若继续优化，应把本 holdout 只当最终测试证据，不再用它选择超参数。建议新建独立 development perturbation manifest，开发一个无需成功标签的在线成员选择信号（例如动作分歧与接触阶段置信度），然后只在全新 test manifest 上评估一次。

RTX 5070 Ti 的限制保持不变：旧 PyTorch 1.12.1+cu113 不含 `sm_120` kernel，所以策略推理在 CPU；GPU PhysX 仍运行在 RTX 5070 Ti 上，现代 PyTorch 训练也仍使用 GPU。
