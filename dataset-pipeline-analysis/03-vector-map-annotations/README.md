# 03：向量地图标注

本节追踪地图 GT 从 nuScenes 静态地图到 VAD 训练样本的转换，重点是：以当前 LiDAR 位姿截取一块局部地图，把线/多边形边界变成带类别的 polyline，并为每条线采样固定数量的点。

配套演示：`python dataset-pipeline-analysis/03-vector-map-annotations/demo.py` 默认从合成的全局地图道路分隔线开始，演示“全局地图坐标中裁剪 LiDAR 对齐的 patch → 转到 LiDAR 局部坐标 → 固定点数采样”，只依赖 NumPy 和 Shapely，不需要 nuScenes 地图文件。

真实数据可视化可在仓库根目录运行：

```bash
conda run -n bevformer python dataset-pipeline-analysis/03-vector-map-annotations/demo.py --real-data
```

它会从 nuScenes 真实表中读取 LiDAR 位姿，并调用 VAD 的 `VectorizedLocalMap` 原样生成 GT，再把每条 20 点折线按类别着色，保存为 `dataset-pipeline-analysis/03-vector-map-annotations/real_map_gt.png`。默认使用已验证的样本 `e93e98b63d3b40209056d129dc53ceee`；本地输出包含 `divider=3`、`ped_crossing=1`、`boundary=4` 条向量。可用 `--sample-token <token>` 指定其他帧，或用 `--find-sample` 在数据集里抽样搜索三类齐全的帧；后者会比较慢，因为每个候选帧都要运行 VAD 地图几何处理。

## 1. 配置入口与执行时机

配置 `projects/configs/VAD/VAD_tiny_e2e.py` 设置：

```python
point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
map_classes = ['divider', 'ped_crossing', 'boundary']
map_fixed_ptsnum_per_gt_line = 20
```

Dataset 根据 `pc_range` 得到地图 patch 尺寸：

```text
patch_size = (pc_range_y_max - pc_range_y_min,
              pc_range_x_max - pc_range_x_min)
           = (60, 30)  # (height, width)
```

注意这个顺序是 `(height, width)`。在 LiDAR 局部平面中，地图 patch 的范围因此是 `x ∈ [-15, 15]`、`y ∈ [-30, 30]`。`map_fixed_ptsnum_per_gt_line=20` 表示每个 GT vector 采样 20 个点；配置中的 `map_num_vec=100` 是模型预测端的 map query 数，不是每帧 GT vector 数量。

`prepare_train_data()` 先运行普通图像/3D 标注 pipeline，再调用 `vectormap_pipeline(example, input_dict)` 添加：

- `map_gt_labels_3d`：每条地图线的类别 ID；
- `map_gt_bboxes_3d`：名字虽含 `bboxes`，实际承载的是 `LiDARInstanceLines`，即地图 polyline 实例。

## 2. 从传感器位姿定义局部地图 patch

`vectormap_pipeline()` 根据输入中的外参构造：

```text
lidar2global = ego2global @ lidar2ego
```

矩阵的平移部分给出当前 LiDAR 原点在全局地图中的位置，旋转部分给出 LiDAR 朝向。`gen_vectorized_samples()` 用这个位置和朝向构造一个在全局地图中旋转过的矩形 patch，然后调用 nuScenes map API 查询与 patch 相交的地图几何。

这里使用的是 **LiDAR 在 global 坐标系下的位姿**，不是直接把 ego 的 `ego2global_translation/rotation` 当作 patch 位姿。二者通过 `lidar2global = ego2global @ lidar2ego` 组合；由于 LiDAR 与 ego 之间存在标定外参，LiDAR 原点位置和 yaw 理论上都可能与 ego 位姿有细微差异。demo 中的 `lidar_global_xy` 和 `lidar_global_yaw_rad/deg` 就对应这个合成后的 LiDAR 位姿。

对于全局点 `(x_g, y_g)`，当前局部坐标等价于先减去 LiDAR 全局位置，再旋转 `-yaw`：

```text
dx = x_g - x_lidar_global
dy = y_g - y_lidar_global
x_local =  cos(yaw) * dx + sin(yaw) * dy
y_local = -sin(yaw) * dx + cos(yaw) * dy
```

因此自车/LiDAR 原点落在局部 `(0,0)`，局部 x 轴随 LiDAR 朝向转动。源码实际做法是：先在全局坐标中裁剪旋转后的 patch，再用 Shapely 旋转 `-patch_angle` 并平移 `(-patch_x, -patch_y)` 转回局部坐标。

## 3. 地图要素如何变成 vector 实例

配置的三个 GT 类别对应如下：

| VAD map 类别 | nuScenes 地图层 | 几何处理 | label |
|---|---|---|---:|
| `divider` | `road_divider`、`lane_divider` | 折线与局部 patch 相交；`MultiLineString` 拆成多条 `LineString` | 0 |
| `ped_crossing` | `ped_crossing` 多边形 | 多边形与 patch 相交，再取外环/内环作为线实例 | 1 |
| `boundary` | `road_segment`、`lane` 多边形 | 先合并区域，再把多边形外轮廓和内环转成线实例 | 2 |

代码中的 `CLASS2LABEL` 把 `road_divider` 和 `lane_divider` 都映射到 0，把 pedestrian crossing 映射到 1，把道路/车道区域轮廓 `contours` 映射到 2。矢量生成后会过滤掉 label 为 `-1` 的实例。

一个易混点：boundary 不是“地图 polygon 填充 mask”。这里取的是 road/lane polygon 的轮廓线，之后和 divider 一样以 polyline 形式监督。

## 4. polyline 固定点数表示

nuScenes 地图要素的原始折线点数不统一。VAD 用 `LiDARInstanceLines` 保存 Shapely `LineString` 列表；访问 `fixed_num_sampled_points` 时，对每条线按弧长均匀取 20 个位置：

```python
distances = np.linspace(0, line.length, fixed_num)
points = [line.interpolate(distance) for distance in distances]
```

所以一帧有 `N` 条有效 vector 时，采样坐标形状为：

```text
[N, 20, 2]  # vector 数、每条线的采样点数、(x, y)
```

`LiDARInstanceLines` 的 `fixed_num_sampled_points` 属性还会把 x/y 分别 clamp 到 patch 边界。标签形状是 `[N]`。实际 `vectormap_pipeline()` 把 `LiDARInstanceLines` 对象放在 CPU-only `DataContainer` 中，而不是立即展平成普通 Tensor；模型侧可按需访问不同的点采样/shift 属性。

GT 的 vector 数 `N` 随场景而变化，点数 20 才是固定的。不要把 `N` 和模型端 100 个 map queries 混为一谈。

## 5. 放回整个数据流

```text
info 中的 lidar2ego / ego2global / map_location
        ↓
vectormap_pipeline：构造 lidar2global
        ↓
VectorizedLocalMap：按位置和 yaw 查询局部地图 patch
        ↓
divider / crossing 边界 / road-lane contours
        ↓
转到 LiDAR 局部 xy，得到带 label 的 LineString 实例
        ↓
LiDARInstanceLines：每条线采样 20 个点
        ↓
map_gt_labels_3d + map_gt_bboxes_3d
```

这一步与 UniAD 中熟悉的 map GT 管线相似之处是：最终监督仍是局部 BEV 中的地图几何；增量重点是 VAD 这里由 `VectorizedLocalMap` 直接从 nuScenes 静态地图层构造实例，并用 `LiDARInstanceLines` 延迟提供固定点数/shift 表示。具体 map decoder 如何消费这些属性、如何计算损失，留到模型侧学习。

## 6. 自测

1. 为什么局部 patch 的尺寸按 `(height, width)=(60,30)` 保存，但 x 的范围是 `[-15,15]`、y 的范围是 `[-30,30]`？
2. 为什么地图点必须旋转 `-yaw` 并平移到 LiDAR 原点，而不能直接把全局坐标当作模型 GT？
3. `map_num_vec=100` 和 GT 点张量 `[N,20,2]` 中的 `N` 分别表示什么？

下一节追踪 `DataContainer` 如何经过 DataLoader 组 batch，以及 VAD 如何从时序图像中使用历史帧 BEV、只对当前帧计算监督损失。
