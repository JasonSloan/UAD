# VAD 模型结构学习记录

本目录从数据管线之后开始，按 VAD 的模型执行路径逐步学习，并与已学过的 UniAD 对照。先学习各分支的 forward，再分别拆解损失函数和评估指标。

| 顺序 | 主题 | 状态 |
|---|---|---|
| [00 VAD forward 总览](00-vad-forward-overview/README.md) | BEV 特征、检测/地图 query、运动预测与规划之间的数据流 | 已完成 |
| [01 BEV 感知主干](01-bev-perception/README.md) | VAD 如何使用 BEVFormer encoder 形成当前/时序 BEV | 已完成 |
| [02 检测分支](02-detection-branch/README.md) | object query、3D 框逐层细化与 NMS-free 解码 | 已完成 |
| [03 地图分支](03-map-branch/README.md) | map query、polyline 解码与输出 | 已完成 |
| [04 运动预测分支](04-motion-prediction/README.md) | agent query、多模态轨迹及地图交互 | 已完成 |
| [05 自车规划分支](05-ego-planning/README.md) | ego query、agent/map 交互与规划轨迹 | 已完成 |

各分支 forward 学完后，转入独立的 [Loss 学习记录](../loss-analysis/README.md)，再继续整理评估指标。
