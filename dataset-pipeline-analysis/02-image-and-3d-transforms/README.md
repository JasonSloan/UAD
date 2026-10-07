# 02：图像与 3D 标注 transforms

本节沿 `VAD_tiny_e2e.py` 的 `train_pipeline`，追踪多相机图像、LiDAR 坐标系 3D 框及其附加标签如何被加载、增强、筛选和打包。这里先不展开 `vectormap_pipeline()`；地图 GT 是 `prepare_train_data()` 在普通 pipeline 之后添加的，留到第 03 节。

可运行的数值演示：`python dataset-pipeline-analysis/02-image-and-3d-transforms/demo.py`。它用合成相机矩阵验证“图像缩放 0.4 后，左乘缩放矩阵可让投影像素同步缩放”，并演示 padding 尺寸和 GT mask 对齐；不需要 nuScenes 数据或 MMDetection3D，只依赖 NumPy。

## 1. Pipeline 顺序

配置位置：`projects/configs/VAD/VAD_tiny_e2e.py::train_pipeline`。

```text
LoadMultiViewImageFromFiles
→ PhotoMetricDistortionMultiViewImage
→ LoadAnnotations3D
→ CustomObjectRangeFilter
→ CustomObjectNameFilter
→ NormalizeMultiviewImage
→ RandomScaleImageMultiViewImage
→ PadMultiViewImage
→ CustomDefaultFormatBundle3D
→ CustomCollect3D
```

读这条链时要把三类数据分开：

1. **图像像素**：加载、颜色扰动、归一化、缩放、padding、转 Tensor。
2. **3D 检测标注**：加载 3D 框和类别，按范围及类别筛选，再打包。
3. **相机几何元信息**：`get_data_info()` 先算出每个相机的 `lidar2img`；图像缩放时同步修改该矩阵，保证 LiDAR 点投影到缩放后图像的位置仍然正确。

## 2. 按步骤追踪

| 顺序 | Transform | 图像做什么 | 3D 框/标签与几何信息 |
|---|---|---|---|
| 1 | `LoadMultiViewImageFromFiles(to_float32=True)` | 逐相机读图，得到 6 张 `H×W×3` 的 float32 图像；图像以 BGR 顺序读入，并存成 list。 | 不改 3D 标注。`lidar2img` 已由 Dataset 的 `get_data_info()` 提供。 |
| 2 | `PhotoMetricDistortionMultiViewImage` | 对每张相机图像独立抽取亮度、对比度、饱和度、色调和通道扰动；只改变像素外观。 | 不改框、类别或 `lidar2img`。不同相机抽到的光度参数可能不同。 |
| 3 | `LoadAnnotations3D(with_bbox_3d=True, with_label_3d=True, with_attr_label=True)` | 不改图像。 | 从 `ann_info` 载入 `gt_bboxes_3d`、`gt_labels_3d`、`attr_labels`。框是 LiDAR 坐标系的 `LiDARInstance3DBoxes`；未知类别标签为 `-1`。`attr_labels` 拼接了 agent future trajectory、future mask、goal、LCF 特征和 future yaw。 |
| 4 | `CustomObjectRangeFilter` | 不改图像。 | 按配置范围 `[-15,-30,-2,15,30,2]` 的 BEV 平面范围过滤框，并对框、类别和附加属性使用同一 mask；同时把 yaw 角限制到标准周期范围。 |
| 5 | `CustomObjectNameFilter` | 不改图像。 | 只保留 `class_names` 中的 10 类（类别 ID `0..9`），并同步筛选框、类别和附加属性。 |
| 6 | `NormalizeMultiviewImage` | BGR 转 RGB 后逐通道执行 `(pixel - mean) / std`；mean/std 使用配置里的 ImageNet 数值。 | 不改 3D 框或投影矩阵。 |
| 7 | `RandomScaleImageMultiViewImage(scales=[0.4])` | 每张图像宽、高乘 0.4。名字里有 Random，但候选尺度只有一个，所以实际固定缩放 0.4。 | 3D 框仍在原 LiDAR 坐标中，不随图像缩小；`lidar2img` 左乘图像缩放矩阵，使投影坐标同步缩小。 |
| 8 | `PadMultiViewImage(size_divisor=32)` | 在图像右侧/下侧补零，使高宽都能被 32 整除。 | 不改 3D 框和 `lidar2img`。记录 `img_shape`、`pad_shape` 等元信息。 |
| 9 | `CustomDefaultFormatBundle3D` | 每张图从 HWC 转 CHW，六张堆成 Tensor，并放进 `DataContainer(stack=True)`。 | 3D 框放进 CPU-only `DataContainer`；类别及 ego future 字段转 Tensor/`DataContainer`。 |
| 10 | `CustomCollect3D` | 输出 `img`。 | 输出配置指定的检测、ego 规划字段，并把相机矩阵、图像 shape、scene/can_bus 等元信息放进 `img_metas`。之后 Dataset 才调用 `vectormap_pipeline()` 添加地图 GT。 |

源码对应关系：

- 多相机加载和标准 `LoadAnnotations3D`：当前 BEVFormer 环境中的 MMDetection3D 0.17.1 实现；本仓库自定义增强和过滤位于 `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`。
- VAD 对标注的组装：`projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py::get_ann_info()`。
- VAD 特有的格式和收集：`projects/mmdet3d_plugin/datasets/pipelines/formating.py::CustomDefaultFormatBundle3D`、`transform_3d.py::CustomCollect3D`。

## 3. 缩放时为什么要改 `lidar2img`，却不改 3D 框？

3D 框仍描述现实场景里目标相对 LiDAR 的位置、尺寸和朝向；缩小相机图像不会改变这些物理量。改变的是像素坐标系：图像中的投影点 `(u,v)` 变为 `(0.4u, 0.4v)`。所以代码保留 LiDAR 坐标系的框，只把投影矩阵同步成：

```text
lidar2img_scaled = image_scale_matrix @ lidar2img
image_scale_matrix = diag(0.4, 0.4, 1, 1)
```

这与 UniAD/BEVFormer 中“相机图像做几何变换时，必须同步维护投影元信息”的思路一致。光度增强和归一化只改像素值，不需要改投影矩阵。

## 4. shape 变化

nuScenes 相机图像通常是 `1600×900`。在这份配置下，假设读入尺寸如此：

```text
单相机读入：        [900, 1600, 3]  (H, W, C)
缩放 0.4 后：       [360,  640, 3]
按 32 padding 后：  [384,  640, 3]
格式化并堆相机：    [6, 3, 384, 640]  (N_cam, C, H, W)
```

随后第 01 节的 `union2one()` 堆叠时间维，DataLoader 再堆叠 batch 维：

```text
单个 Dataset 样本： [T, 6, 3, 384, 640]
模型收到的 batch：  [B, T, 6, 3, 384, 640]
```

## 5. 阅读源码时留意的两个实现细节

1. `RandomScaleImageMultiViewImage` 并没有更新 `scale_factor`，尽管它把图像缩放了；实际投影几何是通过更新 `lidar2img` 保持一致的。阅读时不要仅凭 `scale_factor` 判断这里没有缩放。
2. 该 transform 和随后的 `PadMultiViewImage` 会把 `ori_shape` 写成缩放后、padding 前的 shape。因此在这条 pipeline 的最终 metadata 中，`ori_shape` 不一定代表磁盘原图尺寸；`pad_shape` 才表示 padding 后的尺寸。

此外，训练配置没有在这条 pipeline 中做随机图像翻转或 3D 旋转；本节的空间图像增强主要是固定 0.4 缩放，且通过 `lidar2img` 对齐投影几何。

## 6. 和 UniAD 已学内容的增量关系

- **可复用**：多相机读图、`lidar2img` 投影几何、3D 框类别筛选、`DataContainer` 和 `img_metas` 的职责划分，与 BEVFormer/UniAD 的数据组织思路相通。
- **VAD 这里多出来的重点**：每个 agent 的未来轨迹等信息被拼进 `attr_labels`，范围/类别过滤必须对框和这些属性同步使用同一 mask。
- **本节先不展开**：这些 future labels 后面如何转成 motion 监督、如何算 loss；那属于模型监督部分。

## 7. 自测

1. 图像从 1600×900 缩到 0.4 后，为什么 3D 框的 x/y/z 不乘 0.4？
2. `RandomScaleImageMultiViewImage(scales=[0.4])` 为什么名字叫 Random，但训练时不是随机多尺度？
3. range filter 删除一个目标时，为什么也必须用相同 mask 删除 `attr_labels` 中对应的行？

下一节进入 `vectormap_pipeline()`：看地图矢量 GT 如何从全局地图坐标变换到当前 ego/LiDAR 局部坐标，并采样成模型使用的 polyline 标签。
