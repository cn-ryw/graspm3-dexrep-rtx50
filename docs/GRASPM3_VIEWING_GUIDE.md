# GraspM3 数据查看指南

## 最快的方法

GraspM3 的每个 `.npy` 文件不是图片，而是一个 NumPy 字典。官方定义为：

```text
obj_rotmat: (B, 3, 3)    每条轨迹对应的物体旋转矩阵
obj_scale:  (B,)         每条轨迹对应的物体缩放系数
grasp_seqs: (B, T, 28)   抓取轨迹
```

每个 28 维抓取状态由以下内容组成：

```text
0:3    手的全局平移（相对参考点 [0, 0, 1]）
3:6    手的全局旋转
6:28   Shadow Hand 的 22 个关节角
```

项目提供了隔离查看器。它只读挂载官方数据、运行时断网，并把预览写入数据目录外的 `previews/`：

```bash
./scripts/run_graspm3_viewer.sh core-bottle-1071fa4cddb2da2fc8724d5673a063a6 0
```

输出位置：

```text
/path/to/GraspM3/previews/<object-id>/trajectory_000/
├── summary.json
├── trajectory.csv
├── trajectory_channels.png
└── mesh_preview.png
```

- `summary.json`：数组形状、类型、最小/最大值、旋转矩阵行列式和 mesh 边界。
- `trajectory.csv`：逐帧 28 维状态，也额外给出加上参考点后的 world z。
- `trajectory_channels.png`：平移、全局旋转曲线和 22 关节角热力图。
- `mesh_preview.png`：关联 `decomposed.obj` 的静态三维预览。

## 可以使用的软件

| 内容 | 推荐工具 | 用途 |
|---|---|---|
| `.npy` 数值与字段 | 本项目查看器、VS Code + Python/Jupyter | 查看 shape、轨迹、曲线和导出 CSV |
| `.csv` | LibreOffice Calc、VS Code | 逐帧检查数值 |
| `.obj` | MeshLab、Blender、VS Code 3D Viewer | 交互旋转、缩放并检查物体网格 |
| `.urdf` | RViz、PyBullet、Isaac Gym | 查看碰撞体、关节和仿真加载效果 |
| `.png` / `.json` | 系统图片查看器、VS Code | 快速汇报与记录 |

不要用文本编辑器直接打开 `.npy`，因为它是二进制格式。GraspM3 使用 object array 保存字典，NumPy 读取时需要 `allow_pickle=True`；因此不要加载来源不明的 `.npy`。本查看器把这一步限制在已校验的官方数据目录和隔离容器中。
