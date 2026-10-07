# 03：向量地图损失——匹配 polyline、对齐点序并约束方向

本节追踪地图分支的监督：map query 怎样与 GT 向量做一对一匹配，预测线怎样在 GT 的多个合法点序中选出一个 target，以及分类、点回归和方向 loss 分别起什么作用。

主要源码：

- tiny 配置：`projects/configs/VAD/VAD_tiny_e2e.py` 中 `map_num_vec`、固定采样点数、map losses 和 `map_assigner`
- 多任务 head：`projects/mmdet3d_plugin/VAD/VAD_head.py` 中 `_map_get_target_single()`、`map_loss_single()`、`loss()`
- Hungarian map assigner：`projects/mmdet3d_plugin/core/bbox/assigners/map_hungarian_assigner_3d.py`
- GT 点序候选：`projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py` 中 `LiDARInstanceLines.shift_fixed_num_sampled_points_v2`
- 点损失：`projects/mmdet3d_plugin/VAD/utils/CD_loss.py` 中 `PtsL1Loss`、`PtsDirCosLoss`

## 1. 先看 tiny 配置里的张量规模

tiny 配置使用 100 个 map vector query，每条预测向量和每条 GT 向量都用 20 个 `(x,y)` 点表示；地图类别是 divider、ped_crossing、boundary。对一个样本，相关张量可以概括为：

```text
预测类别：[100 个 query, 3 类]
预测点：  [100 个 query, 20 个点, 2 坐标]
GT 点序候选：[GT 向量, 多种合法点序, 20 个点, 2 坐标]
```

点坐标在 LiDAR 平面坐标系表达，匹配和点 L1 前按 `point_cloud_range` 归一化；方向 loss 会把预测点反归一化后再比较相邻点方向。

## 2. 匈牙利匹配同时考虑类别与点序无关的最小点代价

对每个 query 与每条 GT，`MapHungarianAssigner3D` 会先计算预测点到 GT 各种点序候选的 L1 代价。它把候选 shift 维展开后再还原为 `[query, GT, shift]`，沿 shift 维取最小值，并记录达到最小值的 `order_index`：

```text
query q 与 GT g：
    比较 GT[g] 的所有合法点序
    取最小点代价 + 对应 shift 编号
                │
                ├─ 最小点代价参与 query-GT 匹配 cost
                └─ shift 编号稍后用于构造回归 target
```

最后 Hungarian matching 根据 query-GT 的总代价做全局一对一分配。tiny 配置中，匹配总代价实质上是分类代价和有序点 L1 代价：`cls_cost=2.0`、`pts_cost=1.0`；bbox L1 与 IoU 的权重都是 0。虽然代码仍计算 bbox/IoU cost，它们在当前配置下不改变匹配结果。

这不是先把每条 query 独立匹配到最近 GT：每个 pair 的点序先做最小化，随后 query 与 GT 的配对仍由 Hungarian 全局决定。

## 3. `v2` shift 如何处理线的起点和方向

配置使用 `map_gt_shift_pts_pattern='v2'`。生成候选时，代码依据折线首尾坐标是否相同判断它是闭合线还是开放线，而不是根据 `divider`、`ped_crossing` 等类别写死处理方式：

- **开放线**：生成正向采样点和反向采样点两种候选。这样同一条没有天然起点的线，不会仅因采样方向相反就产生很大的点回归误差。
- **闭合线**：把重复的末端点移除，对顶点起点做循环平移，再重新采样。循环候选保持原有环绕方向；`v2` 这段逻辑不会额外生成反向环绕候选。
- 候选 shift 数不足 `fixed_num - 1` 时用 `padding_value=-10000` 补齐；超过时随机抽到 `fixed_num - 1` 个候选。当前 `fixed_num=20`，所以 shift 维目标长度为 19。

匹配得到的 query-GT pair 确定之后，`_map_get_target_single()` 用 `order_index[query, gt]` 从该 GT 的 shift 候选中取出对应 20 点，作为这个正 query 唯一的点回归 target。未匹配 query 不回归点。

## 4. 三种地图监督各管什么

### 类别：`loss_map_cls`

所有 map query 都参与分类；匹配到 GT 的 query 监督对应地图类别，未匹配 query 监督背景。tiny 配置使用 sigmoid Focal Loss，权重 2.0。

### 点坐标：`loss_map_pts`

正 query 的 20 个预测点与选中的 GT 点序 target 做逐坐标 L1。tiny 配置权重 1.0。这个项同时约束线的位置、形状和点间距离，是当前地图几何监督的主体。

### 相邻点方向：`loss_map_dir`

代码对相邻采样点做差：

```text
预测段向量[i] = 预测点[i+1] - 预测点[i]
GT 段向量[i]   = GT 点[i+1] - GT 点[i]
```

随后用 `CosineEmbeddingLoss`（目标为方向相同）比较两者夹角。它关注线段朝向是否一致，不直接约束线段长度；线的位置与形状仍主要由点 L1 负责。tiny 配置 `map_dir_interval=1`，即比较每对相邻点，方向 loss 权重为 0.005。方向 target 使用匹配阶段选定的 shift，因此开放线若选了反向点序，方向监督也会相应反向。

## 5. tiny 配置里哪些 map loss 真正起作用

| Loss 项 | 当前形式 | 配置权重 | 是否产生有效训练贡献 |
|---|---|---:|---|
| `loss_map_cls` | sigmoid Focal | 2.0 | 是，正负 query 分类 |
| `loss_map_pts` | 点坐标 L1 | 1.0 | 是，正 query 点回归 |
| `loss_map_dir` | 相邻线段 cosine loss | 0.005 | 是，正 query 方向约束 |
| `loss_map_bbox` | 归一化框 L1 | 0.0 | 否，loss 值乘零权重 |
| `loss_map_iou` | GIoU | 0.0 | 否，loss 值乘零权重 |

因此虽然 `loss_dict` 中仍有 `loss_map_bbox`、`loss_map_iou` 两项，tiny 配置实际地图监督重点是**分类 + 点 L1 + 小权重方向约束**。同一套 map loss 会对各 map decoder 层计算；最后一层写为主 loss，前置层也以 `d0.*`、`d1.*` 等辅助 loss 的形式保留。

## 6. 与 UniAD 的增量对照

这里有一个重要区别：**UniAD 的 PanSegFormer 地图分支不是 VAD 这种 polyline 点集预测头。**在 UniAD 已学的 `PansegformerHead` 中，地图/场景结构经 thing/stuff mask decoder 输出空间 mask；地图元素经 query 预测类别、框和 mask。VAD 则直接让 map query 回归固定数量的有序 `(x,y)` 点，形成向量线。

| 对比点 | VAD map 分支 | UniAD PanSegFormer map 分支 |
|---|---|---|
| 几何输出 | 每条向量 20 个有序平面点 | 每个 query 预测空间 mask；stuff query 还输出区域 mask |
| 主要几何监督 | 匹配 shift 后的点 L1 + 相邻点方向 cosine loss | mask Dice loss，另有类别与框监督 |
| 点序问题 | 必须处理开放线方向、闭合线起点等点序歧义 | 以 mask 表示空间范围，不用为一串回归点选起点/方向 shift |
| 匹配 | map Hungarian cost 含分类和有序点代价；tiny 的框/IoU匹配权重为 0 | PanSegFormer 对 thing query 使用类别、框及 mask 相关的匹配/监督；stuff 区域分支另行处理 |

因此，与 UniAD 对照时，VAD 的增量不是简单“地图分支也用了 Hungarian”，而是它把地图表达从 mask 进一步变为显式有序 polyline 点集。这带来了更直接的矢量几何监督，也带来了点序歧义处理和方向 loss。

## 7. 本节小结

- tiny 配置有 100 个 map query，每条预测向量为 20 个点，类别为 divider、ped_crossing、boundary。
- 每个 query-GT pair 先在 GT 的 shift 候选中取最小有序点 L1，再参与全局 Hungarian 匹配。
- `v2` 对开放线提供正反两个方向，对闭合线提供循环起点候选；匹配确定后，正 query 使用对应 shift 的 GT 点序。
- 实际有效地图 loss 是 Focal 分类、点 L1 和小权重相邻段 cosine 方向 loss；map bbox L1 与 GIoU 权重为 0。

下一节进入自车规划损失：command 如何选轨迹监督分支，以及 imitation、地图边界、碰撞和方向约束各自依赖什么预测输入。
