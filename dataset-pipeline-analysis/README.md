# VAD Dataset Pipeline 学习记录

本目录按 UniAD analysis 的方式，逐节记录 VAD 从 temporal info pkl 到模型输入的数据流。每节会对应到真实配置和源码；涉及数值、shape 或几何变换时，再补充独立 demo 或可视化。

## 学习顺序

| 顺序 | 主题 | 状态 |
|---|---|---|
| [00 数据流总览](00-data-flow-overview/README.md) | 配置、Dataset、pipeline、向量地图和模型入口之间的关系 | 已完成 |
| [01 Dataset sample 与时序队列](01-dataset-sample-and-queue/README.md) | `get_data_info()`、`prepare_train_data()`、`union2one()` | 已完成 |
| [02 图像与 3D 标注 transforms](02-image-and-3d-transforms/README.md) | 按 `train_pipeline` 顺序追踪输入、输出和 shape | 已完成 |
| [03 向量地图标注](03-vector-map-annotations/README.md) | `vectormap_pipeline()`、局部坐标变换与 polyline 标签 | 已完成 |
| [04 Dataset 到 `forward_train()`](04-dataloader-and-forward-train/README.md) | DataContainer、batch 维、历史 BEV 与当前帧监督 | 已完成 |

数据管线部分到此结束。模型侧从 [VAD forward 总览](../model-structure-analysis/00-vad-forward-overview/README.md) 开始；后续模型结构、loss 和评估继续记录在独立的 `*-analysis` 目录中，避免把数据准备和模型计算混在一起。

## 总体数据流

```text
vad_nuscenes_infos_temporal_train.pkl
        ↓
VADCustomNuScenesDataset.get_data_info()
        ↓
prepare_train_data()：当前帧 + 历史帧
        ↓
单帧 train_pipeline
        ↓
vectormap_pipeline()：生成向量地图标注
        ↓
union2one()：堆叠时序图像并组织 img_metas
        ↓
DataLoader batch
        ↓
VAD.forward_train()
```
