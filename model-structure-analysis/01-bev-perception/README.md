# 01：BEV 感知主干——从多相机特征到当前 BEV

本节追踪 `VADHead.forward()` 中最前面的 BEV 构建：相机特征如何被 BEV query 读取，历史 BEV 如何按自车运动对齐并融合。这里重点放在输入/输出 shape 和时序坐标；object/map decoder 留到后续章节。

配套演示：`python model-structure-analysis/01-bev-perception/demo.py`。它用 tiny 配置的 BEV 尺寸代入 VAD 的位移公式，验证米制位移如何变成归一化 BEV shift；只依赖 NumPy。

源码入口：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py`
- 图像特征：`projects/mmdet3d_plugin/VAD/VAD.py::extract_img_feat()`
- BEV query 与运动补偿：`projects/mmdet3d_plugin/VAD/VAD_transformer.py::get_bev_features()`
- BEV encoder：`projects/mmdet3d_plugin/VAD/modules/encoder.py::BEVFormerEncoder`
- 两类注意力：`projects/mmdet3d_plugin/VAD/modules/temporal_self_attention.py`、`spatial_cross_attention.py`

## 1. 先看当前配置中的 BEV 网格

`VAD_tiny_e2e.py` 的关键设置：

| 配置 | 值 | 作用 |
|---|---:|---|
| `bev_h`, `bev_w` | `100, 100` | 生成 10,000 个 BEV queries |
| `embed_dims` | 256 | 每个 BEV query / 图像特征的通道数 |
| `point_cloud_range` | `[-15,-30,-2,15,30,2]` | BEV query 覆盖的 LiDAR 平面与高度范围 |
| `num_points_in_pillar` | 4 | 每个 BEV cell 沿高度方向取 4 个 3D 参考点 |
| BEV encoder 层数 | 3 | 每层依次执行时序注意力、相机空间注意力和 FFN |
| FPN level 数 | 1 | 本配置只向 encoder 提供一个尺度的图像特征 |

由范围和网格数可得，每个 cell 的物理尺寸为：

```text
x 方向：30 米 / 100 = 0.3 米
y 方向：60 米 / 100 = 0.6 米
```

因此虽然网格是 `100×100`，物理覆盖区域不是正方形；每个格子的 x/y 尺寸也不同。

## 2. 当前帧图像怎样变成相机特征

`VAD.extract_img_feat()` 收到当前帧图像 `[B, N_cam, C, H, W]`，先把 batch 与相机维展平：

```text
[B, 6, 3, H, W] → [B×6, 3, H, W]
```

之后经过 ResNet-50 和 FPN，再按相机数恢复维度。tiny 配置的特征通道为 256、FPN level 数为 1，所以交给 head 的特征列表中，每个元素形状为：

```text
[B, 6, 256, h, w]
```

注意 `h,w` 是 backbone/FPN 特征图尺寸，不是原图尺寸；图像 backbone 并没有把六个相机先拼成一张全景图。

## 3. 空间交叉注意力：BEV query 到相机图像取证据

`VADHead.forward()` 创建并扩展可学习的 BEV query：

```text
bev_embedding.weight： [100×100, 256]
扩展 batch 后：         [10,000, B, 256]
```

encoder 为每个 BEV cell 构造一个竖直 pillar，并在高度范围内取 4 个 3D reference points。`BEVFormerEncoder.point_sampling()` 用每个样本的 `lidar2img` 把这些 LiDAR 坐标投影到六个相机图像上，再用深度和图像边界生成可见性 mask。

`SpatialCrossAttention` 按相机筛出可见的 BEV queries，仅让它们到对应相机特征图上做可变形采样；一个 BEV query 若在多个相机可见，就把各相机结果聚合。直观上：

```text
BEV 网格位置 → 3D pillar 参考点 → 投影到相机 → 采样图像特征 → 聚合回 BEV query
```

这与 BEVFormer 的核心空间感知路径一致。VAD tiny 使用单个 FPN level，每个空间交叉注意力配置 8 个采样点；每个 pillar 另有 4 个高度参考点。

## 4. 时序自注意力：历史 BEV 与当前 BEV query

BEV encoder 每层的配置顺序是：

```text
TemporalSelfAttention → LayerNorm
→ SpatialCrossAttention → LayerNorm
→ FFN → LayerNorm
```

时序注意力的队列长度固定为 2，组成是：

```text
[上一时刻 prev_bev, 当前时刻 bev_query]
```

它不是把 Dataset 中的 3 帧 RGB 图像一次性并成一个长注意力序列。历史 RGB 帧先在 `VAD.obtain_history_bev()` 中逐帧编码；每次得到的 BEV 作为下一时刻的 `prev_bev`。例如 `queue_length=3`：

| 时刻 | 输入图像 | 进入 encoder 时的 `prev_bev` | 该时刻输出 |
|---|---|---|---|
| `t0` | 最早历史帧 | 无 | `bev(t0)` |
| `t1` | 较近历史帧 | `bev(t0)` | 融入 `t0` 上下文的 `bev(t1)` |
| `t2` | 当前目标帧 | 上一步的 `bev(t1)` | 当前 BEV，进入任务 decoder |

历史帧这段在 `torch.no_grad()` 下运行。虽然每次注意力只显式接收“上一份 BEV + 当前 query”，但上一份 BEV 已经逐步融合更早帧的信息。

## 5. 自车运动如何用于 BEV 对齐

这条路径有两种相关但不同的处理：旋转已有的 `prev_bev`，以及给历史参考点添加平移 shift。

### 5.1 旋转历史 BEV

当 `prev_bev` 存在且 `rotate_prev_bev=True`，代码将其恢复成 `[B, C, bev_h, bev_w]`，再按 `can_bus[-1]` 旋转。该字段是相邻帧 yaw 差，单位为度；Dataset 的 `union2one()` 对历史序列将它改成相对前一帧的角度增量。

### 5.2 平移参考点

`get_bev_features()` 从 `can_bus[:3]` 取相邻帧位置差的 x/y，从 `can_bus[-2]` 取当前 ego 的绝对 yaw（弧度）。先把全局平移方向换算到 BEV 朝向，再按物理网格大小归一化：

```text
平移长度 = sqrt(delta_x² + delta_y²)
平移方向 = atan2(delta_y, delta_x)
shift_y = 平移长度 × cos(bev_angle) / grid_length_y / bev_h
shift_x = 平移长度 × sin(bev_angle) / grid_length_x / bev_w
shift = [shift_x, shift_y]
```

对于 tiny 配置，`grid_length_y=60/100=0.6m`，`grid_length_x=30/100=0.3m`。除以网格数后，`shift` 是归一化 reference-point 坐标中的偏移，不是以米为单位的位移。

此外，18 维 `can_bus` 会经过 `can_bus_mlp` 投影成 256 维向量，并加到 BEV query 上。这是运动状态注入；它与上面的显式空间旋转/平移补偿是两件事。

### 5.3 训练 metadata 中 yaw 两个字段的单位与含义

不要把 `can_bus[-2]` 和 `can_bus[-1]` 混为一个角度：

| 字段 | `get_data_info()` 写入 | `union2one()` 对历史帧的处理 | BEV transformer 用途 |
|---|---|---|---|
| `can_bus[-2]` | 绝对 yaw，弧度 | 保持绝对角度 | 换算成度，参与平移方向到 BEV 方向的计算 |
| `can_bus[-1]` | 绝对 yaw，度 | 改成相邻帧 yaw 差，度 | 旋转 `prev_bev` |

位置 `can_bus[:3]` 同样从绝对位置改成相邻帧位置差。序列首帧的相对位置和 `[-1]` 角增量置零。

## 6. 训练与视频推理的历史状态来源

| 阶段 | `prev_bev` 从哪里来 | 如何清空/初始化 |
|---|---|---|
| 训练 | `obtain_history_bev()` 对当前 Dataset 样本内的历史图像逐帧计算 | 每个 queue 从 `None` 开始，按时间向前滚动 |
| 视频推理 | 上一次 `forward_test()` 保存的 `prev_frame_info['prev_bev']` | scene 变化时清空；`video_test_mode=False` 时每帧都清空 |

当前 `VAD_tiny_e2e.py` 设置 `video_test_mode=True`。推理时还会保存上一帧绝对位置/yaw，并将当前 metadata 改成相邻帧差值，以便复用相同的旋转与平移对齐逻辑。

## 7. 两处值得记下的源码实现细节

1. `VAD/modules/encoder.py` 中写着 `shift_ref_2d = ref_2d`，而不是注释所说的 `ref_2d.clone()`。随后对 `shift_ref_2d` 做原地 `+= shift`，也会改动 `ref_2d` 本身。源码注释明确称这是为复现结果而保留的 bug；因此当 `prev_bev` 存在时，后续构造的两组 reference points 都可能已经带上 shift。阅读或改代码时不要默认为它实现了注释中的“只平移历史 reference”语义。
2. `VADPerceptionTransformer` 默认 `rotate_center=[100,100]`。VAD base 配置的 BEV 是 `200×200`，这个中心与网格中心相符；tiny 配置改成 `100×100`，却没有覆盖该默认值。由于 `torchvision.rotate()` 的 center 是旋转中心坐标，这里看起来与 tiny 网格中心不一致，属于需要用实际旋转结果核验的配置风险；本节只记录风险，不擅自改模型行为。

## 8. 和 UniAD/BEVFormer 的对照

- **相同主干思路**：可学习 BEV queries、历史 BEV 的时序自注意力、将 3D BEV reference 投影到多相机的空间交叉注意力。
- **本节最重要的复用**：你在 UniAD 中学过的 BEVFormer encoder 和相机几何投影基本可直接复用；现在只需把 VAD 的入口时序和配置参数映射进去。
- **VAD 需要特别核对**：VAD 代码位于独立的 `projects/mmdet3d_plugin/VAD/modules/` 实现；tiny 配置的网格是 `100×100`，且 `can_bus` 的绝对 yaw 与相对 yaw 差分别存于 `[-2]`、`[-1]`。

本节输出可以记为：

```text
bev_embed: [bev_h × bev_w, B, embed_dims]
         = [10,000, B, 256]  # VAD tiny
```

它随后供 object decoder 和 map decoder读取。下一节追踪 object query 如何从这份 BEV 中解码 3D 检测结果。

## 9. 自测

1. `queue_length=3` 时，为什么 BEVFormer 时序注意力仍称为长度 2？
2. `can_bus[-2]` 与 `can_bus[-1]` 分别是什么角度、单位是什么？
3. `SpatialCrossAttention` 怎样知道某个 BEV query 应该从哪些相机取特征？
4. tiny 配置中的 `rotate_center=[100,100]` 为什么值得检查？
