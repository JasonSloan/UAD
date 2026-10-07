# 01：检测匹配与目标分配——框监督如何连到 motion GT

本节先拆 VAD 检测 loss 的第一步：300 个 object query 怎样和变长 GT 目标一对一匹配，正负 query 各自拿到什么监督。这里最重要的 VAD 衔接点是：**检测匹配不只决定框的监督，也把同一个 GT agent 的未来轨迹分配给对应正 query。**

主要源码：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py` 中 `train_cfg.pts.assigner` 与 `loss_cls/loss_bbox`
- 多任务 head：`projects/mmdet3d_plugin/VAD/VAD_head.py` 中 `_get_target_single()`、`get_targets()`、`loss_single()`
- GT 未来轨迹字段：`projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py` 中 `get_ann_info()`

## 1. 先看匹配，再看 loss

tiny 配置为检测分支提供 300 个 object query。每帧 GT 目标数随场景变化；Hungarian assigner 根据分类代价和 3D 框回归代价，求 query 与 GT 之间的一对一分配：

```text
300 个预测 query ── Hungarian assignment ──> 当前帧 GT 目标
        │                                      │
        ├─ 匹配到一个 GT：positive query       └─ 每个 GT 至多匹配一个 query
        └─ 没有匹配：negative / 背景 query
```

配置中的代价是 `FocalLossCost(weight=2.0)` 和 `BBox3DL1Cost(weight=0.25)`。`iou_cost` 的权重为 0，所以 IoU 项不影响匹配结果。匹配是一对一的，不是根据分数阈值挑选框，也不是推理时 top-k 解码。

## 2. 正负 query 的检测监督

`_get_target_single()` 先调用 assigner 得到 `pos_inds`、`neg_inds` 和正样本对应的 `pos_assigned_gt_inds`，随后为每个 query 填目标：

| Query | 分类目标 | 框回归目标 |
|---|---|---|
| 正样本：匹配到 GT | 对应 GT 类别 | 对应 GT 3D 框，回归权重置为 1 |
| 负样本：未匹配 GT | 背景/无目标 | 回归权重为 0，不回归 GT 框 |

分类使用 sigmoid Focal Loss，因此没有额外的显式背景 logit；未匹配 query 的类别目标相当于所有前景类别均为负。tiny 配置中的检测分类 loss 权重为 2.0，bbox L1 loss 权重为 0.25。框目标经过 `normalize_bbox()` 归一化，并乘以 `code_weights`；默认设置会降低速度两维的权重。

配置还声明了 `loss_iou=GIoULoss(loss_weight=0.0)`，但当前 detection `loss_single()` 实际返回并写入 `loss_dict` 的只有分类和 bbox loss；IoU cost/loss 都没有有效贡献。

## 3. VAD 的关键增量：正样本匹配同时分配未来轨迹

Dataset 将每个 GT 框的 future 位移和有效步 mask 放在 `gt_attr_labels` 中。`_get_target_single()` 用同一批 `pos_assigned_gt_inds` 取出对应 GT 的 future 数据：

```text
正 query q 匹配到 GT g
       ├─ bbox target ← GT[g] 的当前 3D 框
       └─ traj target ← GT[g] 的未来位移序列
```

代码对正样本执行：

```text
traj_targets[pos_inds] = gt_fut_trajs[pos_assigned_gt_inds]
traj_weights[pos_inds] = 1
```

之后再用 GT future mask 和 `valid_fut_ts` mask 清零无效时间步的回归权重。负 query 没有 future 回归 target；它在后续轨迹模式分类中被标记为背景。下一节会沿着这组 target 继续看 best-of-6 是怎样选回归模式的。

这与 UniAD 的重要区别呼应了上一节：VAD motion query 来自当前检测 query，所以 motion GT 的归属直接依赖检测 Hungarian 匹配；UniAD 的 MotionHead 则接收 tracking 分支提供的 track query 和匹配索引。

## 4. 单层 loss 的结构

每层 detection decoder 的预测都会送入 `loss_single()`。它先通过 `_get_target_single()` 生成分类、框和轨迹 target，再计算：

```text
loss_single = loss_cls + loss_bbox + loss_traj + loss_traj_cls
```

其中本节只展开 `loss_cls` 与 `loss_bbox` 的 target 来源。轨迹的 best-of-K 和模式分类细节留到下一节。tiny 配置的 detection decoder 有 3 层，检测和地图任务还会返回前置 decoder 层的辅助 loss；motion 分支只有一组预测，在 head 中会按 detection decoder 层数扩展后参与 target 计算，最终 `loss_dict` 保留最后一层的主 loss。

## 5. 和 UniAD 已学内容对照

检测 query 与 GT 的 Hungarian 一对一匹配、正样本回归框、负样本只做分类，这些可复用你在 UniAD detection/tracking 中学过的 DETR 式监督直觉。VAD 在这一步的增量不是换了一套分类/框 loss，而是把 GT future 属性接在同一匹配结果上，让检测候选同时成为运动预测候选。

配套 `demo.py` 不重写 Hungarian 算法，只用一个假设的 assignment 演示匹配结果如何传播到类别、框和轨迹 target；真实匹配代价与分配仍由配置中的 assigner 实现。

## 6. 本节小结

- 300 个 query 通过 Hungarian assignment 与当前帧 GT 框一对一匹配。
- 正 query 获得类别、框和对应 GT future 轨迹；负 query 只承担分类背景监督。
- future 轨迹 target 通过 `pos_assigned_gt_inds` 从 GT attributes 取出，因此检测分配决定 motion 监督对象。
- tiny 配置的检测主 loss 是 sigmoid Focal 分类和归一化 3D 框 L1；IoU 配置权重为 0，当前 `loss_single()` 也不返回 IoU loss。

下一节继续看 `loss_traj` 与 `loss_traj_cls`：6 个候选模式中哪条回归、模式分类标签如何产生，以及未来有效步 mask 怎样参与计算。
