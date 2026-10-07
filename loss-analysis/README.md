# VAD Loss 学习记录

本目录从 `VADHead.loss()` 的实际返回值出发，按监督依赖逐步拆解检测、运动预测、地图和规划 loss。先理解匹配与正负样本，再分别进入各任务的损失公式；评估指标另行整理。

| 顺序 | 主题 | 状态 |
|---|---|---|
| [01 检测匹配与目标分配](01-detection-matching/README.md) | Hungarian 匹配怎样监督检测框，并将 GT future 轨迹关联到 query | 已完成 |
| [02 运动预测损失](02-motion-prediction/README.md) | best-of-K 轨迹回归、模式分类与有效步 mask | 已完成 |
| [03 向量地图损失](03-vector-map-loss/README.md) | Hungarian 匹配、GT 点序 shift、点回归与方向约束 | 已完成 |
| [04 自车规划损失](04-ego-planning/README.md) | command 选择、轨迹回归、地图边界、碰撞与方向损失 | 已完成 |
| [05 多任务 loss 汇总](05-loss-aggregation/README.md) | `loss_dict`、辅助 decoder 层 loss 与配置权重 | 已完成 |

检测部分大体复用 DETR/3D 检测监督；VAD 需要重点关注的是：同一组检测正样本匹配还决定了哪些 query 会收到 agent future 轨迹监督。
