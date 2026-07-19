# 数据文件说明

- `metrics.csv`：逐 epoch 的 train/validation BC loss。
- `metrics.json`：训练设备、切分、最佳 epoch 和完整 history。
- `per_object_metrics.json`：最佳 checkpoint 的逐物体、逐动作分量误差。
- `rollout_results.yaml`：三个对象各 20 条 Isaac Gym 回放的成功率。

这些文件不包含 GraspM3 原始轨迹、mesh 或训练 checkpoint。
