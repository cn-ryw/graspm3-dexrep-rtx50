# Gate D：三物体 DexRep 缓存、现代 GPU 训练与物理评估

Last checked: 2026-07-19

## 结论

Gate D 已完成。三物体 raw GraspM3 数据已转为独立 DexRep 缓存，现代 PyTorch 在 RTX 5070 Ti 上完成序列级 train/validation 切分训练，最佳 checkpoint 已由旧 PyTorch 1.12 严格加载，并完成 60 条 Isaac Gym GPU PhysX 回放。

与官方 checkpoint 的同对象结果相比：

| 对象 | 官方 checkpoint | Gate D best checkpoint | 变化 |
|---|---:|---:|---:|
| jar | 1/20 = 5% | 4/20 = 20% | +15 pp |
| cellphone | 0/20 = 0% | 0/20 = 0% | 0 pp |
| USB stick | 0/20 = 0% | 7/20 = 35% | +35 pp |
| 合计 | 1/60 = 1.67% | 11/60 = 18.33% | +16.67 pp |

这是对三个已参与训练对象的 in-distribution 结果，不是未见对象泛化成绩。不能把 11 倍的比率提升外推为通用性能提升。

## 1. DexRep 缓存

raw dataset 和 meshdata 以只读方式挂载；缓存写入：

```text
experiments/gate-d-three-object-dexrep-cache/
```

| 对象 | raw 序列 | 通过 physics replay | 帧数 | 缓存大小 |
|---|---:|---:|---:|---:|
| jar | 20 | 20 | 70 | 31,977,845 B |
| cellphone | 20 | 13 | 70 | 20,785,909 B |
| USB stick | 20 | 20 | 70 | 31,977,845 B |

cellphone 在此前独立运行中曾得到 14/20，本次固定 seed 得到 13/20，表明 GPU PhysX replay 仍存在轻微非完全确定性。缓存 SHA-256：

```text
826abd3faf43ef41f32b5f854601ff8648c3a84207593d1f3192f8106f30dbe3  jar
51e3ad0f554bb9ebe0b89efd7594151107488229420a4ce42363739e68718c50  cellphone
aa3e69855db2e6b75b593df6ebdefc0f2b61e78885ee8a902ac12804ec26b855  USB stick
```

首次落盘测试还发现并修复了官方 `data_preprocess.py` 在绝对输出路径前强行拼接 `./` 的 bug；失败发生在保存阶段，未覆盖原始数据。

## 2. 训练设置与结果

```text
model                 ActorCriticDexRep-compatible MLP
PyTorch               2.7.1+cu128
GPU                   NVIDIA GeForce RTX 5070 Ti, sm_120
seed                  0
split                 per-object, sequence-level, 80/20
batch                 256
learning rate         2e-4
requested epochs      200
early-stop patience   30
completed epochs      157
best epoch            127
elapsed               4.514 s
```

切分严格以完整 sequence 为单位，未将同一轨迹中的相邻帧拆到 train 和 validation：

| 对象 | train sequences | validation sequences |
|---|---:|---:|
| jar | 16 | 4 |
| cellphone | 10 | 3 |
| USB stick | 16 | 4 |

总计 2,940 个训练帧、770 个验证帧。结果：

```text
epoch 1 train loss       0.899661
epoch 1 validation loss  0.473445
best validation loss     0.097382 (epoch 127)
epoch 157 train loss     0.009776
epoch 157 validation     0.099044
```

最佳 checkpoint：

```text
experiments/gate-d-three-object-modern-gpu-bc/best.ckpt
SHA-256: b1a1a02fb58eeccb0a3d02e035691383fd2fb8f827bf0357459347984f69916e
```

旧 PyTorch 1.12.1+cu113 验证：所有 `state_dict` 键严格匹配、前向 shape `(2, 28)`、输出全部有限。

## 3. Cellphone 瓶颈定位

最佳 checkpoint 的逐物体验证误差为：

| 对象 | validation BC loss | action MAE | wrist MSE | orientation MSE | finger MSE |
|---|---:|---:|---:|---:|---:|
| jar | 0.04148 | 0.03067 | 0.000536 | 0.002572 | 0.007164 |
| cellphone | 0.21330 | 0.13000 | 0.010975 | 0.016256 | 0.045092 |
| USB stick | 0.06635 | 0.04789 | 0.000674 | 0.000741 | 0.016368 |

cellphone 验证 BC loss 是 jar 的约 5.1 倍、USB stick 的约 3.2 倍；其中 wrist MSE 尤其突出。训练误差却与另外两个对象同量级，说明主要问题不是优化器没有拟合，而是：成功轨迹只有 13 条，姿态/轨迹间泛化不足，并且逐帧 MLP 对 cellphone 的 wrist 控制变化不稳。这也解释了 aggregate validation loss 已较低但 cellphone 闭环仍为 0%。

下一项改进应优先扩充同类 cellphone 成功轨迹并进行对象/姿态均衡采样，而不是单纯增加 epoch；当前曲线已经显示继续训练主要压低 train loss，validation 约在 0.10 平台震荡。

## 4. 产物

- 训练曲线与 success 对比：`experiments/gate-d-three-object-modern-gpu-bc/gate_d_training_and_success.png`
- SVG 版本：`experiments/gate-d-three-object-modern-gpu-bc/gate_d_training_and_success.svg`
- 原始训练历史：`experiments/gate-d-three-object-modern-gpu-bc/metrics.json`
- CSV：`experiments/gate-d-three-object-modern-gpu-bc/metrics.csv`
- 逐物体误差：`experiments/gate-d-three-object-modern-gpu-bc/per_object_metrics.json`
- 物理评估 YAML：`experiments/gate-c-gate-d-best-eval/one_best_trained_test_num40.yaml`
- 缓存日志：`reports/logs/gate-d-three-object-dexrep-cache-20260719-192153.log`
- 训练日志：`reports/logs/gate-d-three-object-modern-gpu-train-20260719-192509.log`
- 旧环境加载日志：`reports/logs/gate-c-gate-d-best-legacy-load-20260719-192546.log`
- 物理评估日志：`reports/logs/gate-c-gate-d-best-eval-20260719-192547.log`
- 逐物体分析日志：`reports/logs/gate-d-per-object-analysis-20260719-194537.log`

## 5. 下一阶段建议

1. 从完整 GraspM3 中选择更多 cellphone/相近扁平物体，生成额外 DexRep cache。
2. 使用按对象均衡 sampler，避免 20/13/20 的样本差异；对 wrist 分量尝试更高权重或姿态分桶。
3. 保留当前三个对象作为 in-distribution 验证，同时增加完全未参与训练的对象做泛化测试。
4. 仅在数据扩充仍无法改善 wrist validation error 时，再尝试轻量 GRU/TCN 或动作平滑正则。
