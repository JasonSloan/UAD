# 00：VAD forward 总览

本节从刚学完的模型输入出发，先把 VAD 一次训练 forward 中的主干关系串起来。暂时不推导具体 loss，也不逐层拆 Transformer；后续章节再逐个分支深入。

源码入口：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py`
- Detector：`projects/mmdet3d_plugin/VAD/VAD.py`
- 多任务 head：`projects/mmdet3d_plugin/VAD/VAD_head.py`
- Transformer：`projects/mmdet3d_plugin/VAD/VAD_transformer.py`

## 1. 先回答：VAD 的感知部分是不是 BEVFormer？

是。当前配置的 `VADHead` 使用 `VADPerceptionTransformer`；它的 encoder 配置为 `BEVFormerEncoder`，里面包含 `TemporalSelfAttention` 和 `SpatialCrossAttention`。因此你学过的 BEVFormer BEV 构建机制可以直接复用。

但 VAD 不是把完整 UniAD 模型原样换个名字：VAD 的 `VADHead` 在 BEV 特征之上同时运行 object decoder 和 map decoder，再把检测对象与地图特征交给运动预测和自车规划模块。后几部分是本次学习的增量重点。

## 2. 从图像到多任务输出

以 `VAD_tiny_e2e.py` 为例，模型只启用相机输入（`use_camera=True`、`use_lidar=False`）：

```text
当前多相机图像 [B, 6, 3, H, W]
        │
        ├─ reshape：相机图像并入 batch 维
        ├─ ResNet-50 + FPN：提取多相机特征
        └─ reshape 回 [B, 6, C, h, w]
                    │
                    ├─ BEV query + 相机特征 + prev_bev
                    │       └─ VADPerceptionTransformer / BEVFormerEncoder
                    │                    └─ 当前 BEV 表征
                    │
                    ├─ object query decoder ──> 3D 检测结果
                    │                         └─ agent 多模态未来轨迹
                    │
                    ├─ map query decoder ─────> 地图类别与 polyline
                    │                         └─ 地图特征参与 agent 运动交互
                    │
                    └─ ego query
                         ├─ 与 agent 特征交互
                         ├─ 再与 map 特征交互
                         └─ ego future decoder ─> 自车规划轨迹
```

这里的 `prev_bev` 是历史 BEV 特征，不是历史 RGB 图像，也不是跟踪 ID。训练时它来自 Dataset queue 的历史图像；推理时在视频模式下由模型跨连续帧保存并更新。

## 3. Detector 与 Head 各自负责什么

`VAD` 继承 MMDetection3D 的 `MVXTwoStageDetector`，但当前 tiny 配置没有启用点云分支。它主要负责：

1. 将图像送入 backbone/FPN，整理成 `[B, N_cam, C, h, w]` 特征。
2. 在训练入口切分历史帧与当前帧，并为历史帧取得 `prev_bev`。
3. 把当前图像特征、metadata 和历史 BEV 交给 `VADHead`。
4. 训练时调用 head 的 loss；推理时把 head 输出解码成检测、地图、agent 轨迹和 ego 轨迹结果。

`VADHead` 是多任务主体：它创建 BEV/object/map queries，调用 Transformer 生成 BEV 与 query 特征，再由各预测分支解码结果。运动预测和规划并不是独立地从原始图像各自重新提特征，而是复用同一轮感知得到的 object/map 表征。

## 4. 训练 forward 的时序边界

第 04 节已经看到 `VAD.forward_train()` 如何接收 `[B,T,N_cam,C,H,W]`。放到模型总图里是：

```text
前 T-1 帧 ──无梯度逐帧编码──> prev_bev ─┐
                                         ├─> 当前帧 VADHead ─> 各任务预测与 loss
最后 1 帧 ───── backbone/FPN ───────────┘
```

历史帧的 BEV 通过 `only_bev=True` 生成；只有最后一帧进入带监督的多任务 head。当前帧的检测框、地图线和 ego future 标签由 Dataset queue 最后一帧提供。

## 5. 配置里的几个关键规模

以下均来自 `VAD_tiny_e2e.py`，先用来建立 query 数量级的直觉：

| 配置项 | 值 | 含义 |
|---|---:|---|
| 图像 backbone | ResNet-50 | 当前配置的相机特征提取器 |
| FPN 输出通道 | 256 | BEVFormer 与各 query 使用的特征维度 |
| BEV 网格 | `100 × 100` | BEV encoder 的空间 query 网格 |
| object queries | 300 | 检测 decoder 的候选对象槽位 |
| map vectors | 100 | 地图 decoder 的候选线实例槽位 |
| 每条预测地图线采样点 | 20 | map query 对应的 polyline 点数 |
| agent 未来模式 / 时间步 | `6 / 6` | 每个检测对象的多模态未来轨迹规模 |

注意：`100` 个 map vectors 是模型的候选 query 数，不是每帧 GT 地图线数；GT 数量仍随场景变化。自车规划的模式数由 `ego_fut_mode` 控制，与 agent 的 `fut_mode` 不是同一个参数。

## 6. 与 UniAD 的复用点和增量点

- **可直接复用**：相机 backbone/FPN、BEV query、BEVFormer 时序 encoder、object query 解码等知识。
- **需要增量学习**：VAD 用 map decoder 显式预测 vector map；agent 轨迹预测和 ego planning 都在同一 `VADHead` 中，并通过 object/map query 特征交互。
- **不要混淆**：检测 query 是当前帧候选对象，不等于具有跨帧身份的 track query；此处也没有 UniAD 的 occupancy 分支。

这是一张执行路径总图，不逐个解释每层注意力，也不展开损失和评估。下一节先沿可复用的 BEVFormer 主干，核对 VAD encoder 配置、`prev_bev` 对齐和当前 BEV 特征的形成过程。
