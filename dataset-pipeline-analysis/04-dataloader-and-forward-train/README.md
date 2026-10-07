# 04：Dataset、DataLoader 到 `VAD.forward_train()`

本节接上前几节的数据准备，追踪一个时序训练样本如何被 MMCV `DataContainer` 包装、由 DataLoader 组成 batch，再在模型入口拆分为历史帧和当前帧。

配套演示：`python dataset-pipeline-analysis/04-dataloader-and-forward-train/demo.py`。它用合成数据调用项目环境中的 MMCV `collate()`，展示堆叠张量、变长标注和 CPU-only 元信息的 batch 结构；不需要 nuScenes 数据或 GPU。

## 1. 从 Collect 到 Dataset 返回

配置 `projects/configs/VAD/VAD_tiny_e2e.py` 的 `CustomCollect3D` 收集：

```text
gt_bboxes_3d、gt_labels_3d、img、ego_his_trajs、ego_fut_trajs、
ego_fut_masks、ego_fut_cmd、ego_lcf_feat、gt_attr_labels
```

`CustomCollect3D` 另外把相机投影矩阵、图像尺寸、`scene_token`、`can_bus` 等放入 `img_metas`，并将它包装成 `DataContainer(cpu_only=True)`。随后 `prepare_train_data()` 调用 `vectormap_pipeline()`，再追加 `map_gt_bboxes_3d` 和 `map_gt_labels_3d`。因此地图 GT 不在配置的 `Collect3D.keys` 中，也不是 `CustomCollect3D` 收集的字段。

对训练样本来说，`union2one()` 把 queue 中每帧的图像沿时间维堆叠，形成 `[T, N_cam, C, H, W]`；同时把每帧的 `img_metas` 放入以时间位置为键的字典。检测、地图和规划标注仍取自 queue 最后一个元素，也就是目标帧。

## 2. `DataContainer` 的三种打包策略

`DataContainer` 本身不是 Tensor，也不等于“把数据搬到 GPU”。它给 MMCV 的 `collate` 和模型并行 scatter 提供打包规则。VAD 这条训练管线主要有三类：

| 数据例子 | 主要设置 | batch 时的处理 | 为什么 |
|---|---|---|---|
| `img`、ego 轨迹 Tensor | `stack=True`，通常 `cpu_only=False` | 同形状数据沿 batch 维堆叠；随后可分发到 GPU | 图像与固定形状的规划特征适合组成批量 Tensor |
| `gt_labels_3d`、`map_gt_labels_3d` | `stack=False`，`cpu_only=False` | 保留为每个样本各自的 Tensor 列表，再按设备分发 | 每帧目标/地图线数量可变，不能直接堆成矩形 Tensor |
| `img_metas`、3D 框对象、`map_gt_bboxes_3d` | `cpu_only=True` | 保留 Python 对象，在 CPU 上传给模型 | 元信息和自定义几何对象不应被默认堆叠或搬到 GPU |

地图字段的包装是在 `vectormap_pipeline()` 中明确设置的：地图类别 Tensor 为 `cpu_only=False`；`LiDARInstanceLines` 几何对象为 `cpu_only=True`。`gt_bboxes_3d` 也由默认 3D formatter 作为 CPU-only 对象处理。

另一个易混点是：`DataContainer.data` 在 DataLoader collate 之后可能多出一层“每卡分组”的列表。那一层不是新的时间维或 batch Tensor 维。训练模型经 MMCV scatter 后，`img` 才作为 Tensor 进入模型，CPU-only 元信息则以每个样本一个元素的列表形式传入。

## 3. 维度如何逐层增加

以配置中的六路相机、`queue_length=3` 为例，忽略具体图像高宽：

```text
单帧格式化图像                         [6, 3, H, W]
Dataset 的 union2one 返回一个时序样本   [3, 6, 3, H, W]
DataLoader 组成模型 batch               [B, 3, 6, 3, H, W]
```

`B` 是每个进程/GPU 的 batch size。当前 `VAD_tiny_e2e.py` 设 `samples_per_gpu=1`，所以标准训练配置下每张 GPU 的 `B=1`；多卡时全局 batch size 是 GPU 数量乘以 1。`T=3` 是单个样本内部的时序长度，不是 batch size。

`img_metas` 与图像的两层结构对应：模型收到一个长度为 `B` 的列表，每项是该样本按时间位置索引的 metadata 字典。第 `i` 个时间槽位的 `img_metas` 与 `img[:, i]` 对应。

## 4. `forward_train()`：历史上下文与当前帧监督

`projects/mmdet3d_plugin/VAD/VAD.py::forward_train()` 收到 `[B,T,N_cam,C,H,W]` 后，按时间维切分：

```python
prev_img = img[:, :-1, ...]  # 前 T-1 帧
img = img[:, -1, ...]        # 最后一帧，也就是 Dataset 的目标帧
```

然后执行以下步骤：

1. `obtain_history_bev(prev_img, prev_img_metas)` 在 `torch.no_grad()` 下逐帧提取历史 BEV。每个时刻的图像特征和对应 metadata 一起送入 `pts_bbox_head(..., only_bev=True)`；历史帧只提供时序上下文，不在这里计算监督损失。
2. 从每个样本的 metadata 字典中取最后一帧，得到当前帧 `img_metas`。
3. 当前帧图像经过 `extract_feat()`，再与历史 `prev_bev` 一起进入 `forward_pts_train()`。
4. `forward_pts_train()` 调用检测头得到预测 `outs`，并把当前帧的检测 GT、地图 GT、ego future 轨迹/掩码/指令及 `gt_attr_labels` 交给 head 的 `loss()`。

数据与监督的对应关系可以概括为：

```text
时序图像前 T-1 帧 ──无梯度──> 历史 BEV ─┐
                                          ├─> 当前帧 head 与 loss
时序图像最后 1 帧 ───────────> 当前特征 ─┘       ↑
                                                当前帧标注
```

所以“Dataset 样本里有 T 帧”不意味着“每帧都各自算一次 loss”。在这个训练入口中，历史帧用来形成 `prev_bev`，监督目标对应 queue 的最后一帧。

## 5. 与 UniAD 学习路径的增量关系

- 可以复用你对 `DefaultFormatBundle`、`Collect3D`、`DataContainer`、`img_metas` 和 batch 维的理解。
- 本节的关键增量是把这些通用打包机制与 VAD 的时序 queue 联系起来：`union2one()` 先在 Dataset 内堆时间，DataLoader 再堆 batch。
- VAD 在模型入口显式拆分历史图像和当前图像；历史 BEV 在无梯度模式下生成，当前帧的检测、地图和规划监督再进入统一 loss。

## 6. 自测

1. `[B,T,N_cam,C,H,W]` 中，`B` 和 `T` 分别由哪一层产生？
2. 为什么 `map_gt_bboxes_3d` 不能像图像那样直接堆成 Tensor？
3. `forward_train()` 为什么只把最后一帧的 GT 送入当前帧 loss？历史帧在本函数里起什么作用？

到这里，Dataset 到模型训练入口的数据路径已经闭合。下一阶段可以从模型侧开始，按感知、跟踪/运动预测、地图和规划等分支继续拆 VAD 的 forward、loss 与评估。
