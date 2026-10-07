# 05：多任务 loss 汇总——从 `loss_dict` 到反向传播标量

前四节分别追踪了检测、agent 运动、向量地图和自车规划的监督。本节把它们重新放回 `VADHead.loss()`：模型先返回一组带名称的 loss 张量，检测器训练框架再把名称中含 `loss` 的项相加，形成用于反向传播的总 loss。

主要源码：

- tiny 配置：`projects/configs/VAD/VAD_tiny_e2e.py` 的 decoder 层数和 loss 配置
- 各分支组装：`projects/mmdet3d_plugin/VAD/VAD_head.py` 的 `loss()`
- 模型训练转发：`projects/mmdet3d_plugin/VAD/VAD.py` 的 `forward_train()`
- 字典解析与求和：当前 bevformer 环境 `mmdet/models/detectors/base.py` 的 `_parse_losses()`

## 1. 主 decoder 层的 loss

tiny 配置的检测 decoder 和 map decoder 都是 3 层，并开启 `return_intermediate=True`。`VADHead.loss()` 对每一层分别计算 loss，但把最后一层的结果作为不带层号的主 loss：

| 分支 | 主 loss key |
|---|---|
| 检测 | `loss_cls`、`loss_bbox`、`loss_traj`、`loss_traj_cls` |
| 地图 | `loss_map_cls`、`loss_map_bbox`、`loss_map_iou`、`loss_map_pts`、`loss_map_dir` |
| 规划 | `loss_plan_reg`、`loss_plan_bound`、`loss_plan_col`、`loss_plan_dir` |

所以主输出共有 13 个 loss key。这里“主”指最终 decoder 层的检测/地图结果；规划分支本身只产生一组 loss，不按这些 decoder 层逐层重复监督。

## 2. 辅助 decoder 层的 loss

前两层的 loss 以 `d0.`、`d1.` 前缀加入字典：

- 检测辅助层：每层只有 `loss_cls` 和 `loss_bbox`。
- 地图辅助层：每层有 `loss_map_cls`、`loss_map_bbox`、`loss_map_iou`、`loss_map_pts`、`loss_map_dir`。
- 没有 `d0.loss_traj`、`d1.loss_traj`，也没有按层展开的 planning loss。

需要区分“代码算过”和“最终放进字典”：`loss_single()` 会为各检测层计算运动回归和模式分类 loss，但组装 `loss_dict` 时只取最后一层的 `loss_traj`、`loss_traj_cls`。因此前层没有对应的运动 loss key。

在 tiny 配置里，检测和地图各有两层辅助输出，合计 14 个辅助 loss key。加上 13 个主 key，字典总计 27 项。地图框回归和 IoU 的权重在该配置中为 0，因此相应主/辅助 key 仍存在，但数值为 0。

## 3. 权重在哪里乘

各个 loss 模块通常在产生张量时已经乘过配置中的 `loss_weight`。例如 tiny 配置的检测分类权重为 2.0、检测框回归为 0.25、地图点回归为 1.0、自车规划方向项为 0.5。运动回归还会在 `loss_single()` 中乘其有效步归一化因子。

因此 `_parse_losses()` 汇总时，不会再按配置把这些权重乘第二遍；它对字典中的张量取 mean，然后把所有 key 名含 `loss` 的值求和。`d0.`、`d1.` 只是名字前缀，不会让辅助 loss 自动乘一个较小系数。当前配置没有给辅助层另设衰减权重，所以这些辅助项按各自 loss 模块已经加权的数值直接参与总和。

概念上可以写成：

```text
total_loss = 所有主 loss + 所有辅助 decoder loss
```

配置权重、mask、正负样本归一化等已在各个 loss 的内部计算中生效；这里的加法是最后的汇总步骤。

## 4. Encoder loss 为什么没有出现

tiny 配置使用 `as_two_stage=False`。`VADHead.loss()` 只有在 `enc_cls_scores` 或 `map_enc_cls_scores` 非空时才会加入 encoder/proposal loss；当前配置下这两组输出为 `None`，因此字典中没有 `enc_loss_cls`、`enc_loss_bbox` 或 `enc_loss_map_*`。

这不是“encoder loss 权重为 0”，而是当前 two-stage 配置没有生成这组 encoder proposal 预测及其 loss。

## 5. 从模型返回值到反向传播

数据流可以概括为：

```text
VADHead.loss()
  └─ 返回带名称的 loss_dict
       └─ BaseDetector._parse_losses()
            ├─ 每项张量取 mean，生成日志值
            └─ 将所有名称含 loss 的项相加，得到 total loss
                 └─ train_step 返回 total loss，训练循环据此反向传播
```

日志中既会看到各个分项，也会看到总 `loss`。因为辅助层 key 也包含 `loss`，它们同样被计入总和。配置权重相同也不意味着各任务梯度影响相同：各 loss 的数值范围、reduction、有效元素数量和梯度尺度都可能不同；这与上一节讨论的任务间梯度冲突是同一问题的另一面。

## 6. UniAD 对照与小结

VAD 和 UniAD 都会用最终 decoder 输出计算主任务 loss，并可将中间 decoder 输出作为辅助监督；具体有哪些辅助 key，仍要逐模型看 `loss()` 如何组装字典，不能只根据 decoder 层数推断。

本仓库 tiny VAD 的关键结论：

- 主 loss 13 项，辅助 decoder loss 14 项；字典共 27 个 key，其中地图 bbox/IoU 项因权重为 0 而为零。
- 检测辅助层只监督分类和框；地图辅助层保留五项地图损失。
- 运动 loss 虽在逐层计算函数中返回，但只把最后一层的运动回归/分类放入字典；规划 loss 也只计算一组。
- encoder/proposal loss 因 `as_two_stage=False` 不会生成。
- 最终训练标量是所有 `loss` 项求和；模块权重已乘过，辅助层没有额外层级衰减。

配套 `demo.py` 使用假数值演示“分项字典直接求和”的规则，不模拟任何真实 batch 的 loss 数值。
