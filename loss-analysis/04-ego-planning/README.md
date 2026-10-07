# 04：自车规划损失——command 选轨迹，约束项看周围环境

本节追踪 VAD 规划分支的四个 loss：轨迹模仿、地图边界、agent 碰撞和车道方向。先要记住一个前提：tiny 配置的规划头一次输出三条候选轨迹，训练时用数据里的 one-hot command 选中其中一条监督；command 本身不作为当前 VAD ego forward 的输入。

主要源码：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py` 中 `loss_plan_*` 权重
- loss 调用：`projects/mmdet3d_plugin/VAD/VAD_head.py` 中 `loss_planning()` 和 `loss()`
- 约束实现：`projects/mmdet3d_plugin/VAD/utils/plan_loss.py`
- 自车 GT 和 command：`tools/data_converter/vad_nuscenes_converter.py` 中 `ego_fut_trajs`、`ego_fut_masks`、`ego_fut_cmd`
- UniAD 对照：`/mnt/nfs_docker_volume/training-container-space/code/end2end/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py`

## 1. 输入是什么

tiny 配置的自车预测形状为 `[B, 3, 6, 2]`：三条 command 模式、每条 6 个未来步、每步二维位移。GT 是 `[B, 6, 2]` 的逐步位移，另有：

- `ego_fut_masks[B,6]`：该 GT 时间步是否有效；
- `ego_fut_cmd[B,3]`：one-hot 的右转、左转、直行 command；
- 预测地图向量与类别分数；
- 检测框中心、agent 多模态 future 位移和 motion mode 分数。

converter 从当前帧往后读取自车 pose，变换到当前 LiDAR 坐标系后计算相邻位置差，形成逐步位移；若 scene 后续帧不足，剩余 future mask 保持为 0。command 按末端 future 坐标的 x 分量生成：`x >= 2` 为右转 `[1,0,0]`，`x <= -2` 为左转 `[0,1,0]`，否则直行 `[0,0,1]`。

## 2. `loss_plan_reg`：只回归 command 指定的模式

代码把同一条 GT 轨迹复制给三个候选，再构造权重：

```text
权重[batch, mode, time, xy]
  = command[batch, mode] × future_mask[batch, time]
```

因此，每个样本只有 command 指定的那条预测轨迹、且只有 mask 有效的时间步参与 L1；另外两条候选的 imitation 权重为 0。tiny 配置的 `loss_plan_reg` 权重为 1.0。

这里的 command 是从 GT future 生成的监督选择信号，而不是规划网络 forward 的在线输入。也就是说，训练这个 loss 时不是让三条模式各自拟合不同 GT；是一条由 GT command 选中的模式拟合该 GT 轨迹。

## 3. 三个环境约束共用 command 选中的轨迹

`loss_plan_bound`、`loss_plan_col` 和 `loss_plan_dir` 都接收：

```python
ego_fut_preds[ego_fut_cmd == 1]
```

布尔索引会为 batch 中每个样本取出 one-hot command 对应的预测模式。因此这些约束不是只作用于固定的 mode 1，也不是同时作用于三条候选；每个样本上仍是 command 选中的那条轨迹。它们也都接收 `ego_fut_masks`，无效 future 步不计入 loss。

需要注意，ego head 输出的是逐步位移，三个约束实现会先 `cumsum` 成相对当前自车位置的累计轨迹点，再计算距离、交叉或方向关系。

## 4. `loss_plan_bound`：远离预测的 boundary 向量

该项从预测地图中筛出 `boundary` 类（类别索引 2）且 sigmoid 分数不低于 0.5 的向量，把点坐标反归一化回 LiDAR 平面坐标。对每个 ego future 位置，它找最近的 boundary 向量及其最近采样点：

- 点到 boundary 最近采样点的距离小于 1.0 m 时，惩罚 `1.0 - distance`；
- 距离不小于 1.0 m 时，该点的距离惩罚为 0；
- 另外会检查 ego 相邻轨迹段是否与所选 boundary 折线段相交；源码在检测到相交后，将该时刻及之后的 boundary loss 清零。

因此它不是基于 GT 地图做的严格道路可行驶区域约束，而是基于模型预测的 boundary 向量计算近距离惩罚和线段相交处理。当前配置 `loss_plan_bound` 权重为 1.0，`dis_thresh=1.0`。

一个实现层面的注意点：`PlanMapBoundLoss` 默认 `perception_detach=False`，所以除 ego 轨迹外，选中的 map 点坐标也可能沿着该 loss 收到梯度；类别阈值和最近邻索引本身是离散筛选。

## 5. `loss_plan_col`：基于 agent 未来点的轴向安全间距

碰撞项的处理顺序是：

1. 对每个检测 query 取最大类别分数；分数低于 0.5 的 agent 被屏蔽。
2. 只保留类别索引 0–4 的 vehicle 类预测；其余类别被屏蔽。
3. 每个 agent 只取 motion mode 分数最高的一条 future 轨迹。
4. 将该 future 位移累计，并加上 agent 当前检测中心；与累计后的 ego 轨迹逐步比较。
5. 过滤与 ego 点的欧氏距离大于 3 m 的 agent 点，然后分别计算 x、y 方向最小间距。

当前 tiny 配置的阈值为 `x_dis_thresh=1.5 m`、`y_dis_thresh=3.0 m`：某轴间距小于阈值时，按“阈值减实际间距”惩罚；间距达到阈值则该轴惩罚为 0。`loss_plan_col` 权重为 1.0。

这不是 oriented 3D box 相交测试，也不是对所有 agent modes 取期望；它是针对筛选后的 vehicle、每个 agent 的最高分 future mode 做点级、逐轴 clearance 惩罚。实现还分别选择 x 距离最小的 agent 和 y 距离最小的 agent，因此两个轴的惩罚不一定来自同一辆车。

## 6. `loss_plan_dir`：使 ego 朝向跟附近 divider 方向一致

方向项从预测地图中筛出 `divider` 类（索引 0）且分数不低于 0.5 的向量。对每个 ego future 位置：

1. 找最近的 divider 向量，再找该线上的最近采样点；
2. 用该点相邻点构造局部 lane 方向；若最近点在向量末尾，就改用它前一个点；
3. 用连续 ego 轨迹点的差计算 ego 朝向，最后一步沿用前一步朝向；
4. 当最近 divider 距离超过 2 m，或整条 ego future 的首末位移小于 1 m（静止），该步方向损失置 0；否则惩罚 ego yaw 与 lane yaw 的绝对角差。

代码把角差折叠到约 `[-π/2, π/2]`，所以同一条 divider 的正向/反向被视为同一条无向车道方向。tiny 配置 `loss_plan_dir` 权重为 0.5，`dis_thresh=2.0 m`。

它比较的是 ego 未来轨迹的运动方向与预测 divider 的局部切线，不是直接监督车辆车身 yaw GT。与 boundary 项类似，当前实现使用预测地图点，而不是 map GT。

## 7. 权重汇总与示例

| Loss 项 | 约束对象 | tiny 权重 | 主要 mask/筛选 |
|---|---|---:|---|
| `loss_plan_reg` | ego 逐步位移与 GT | 1.0 | one-hot command × ego future mask |
| `loss_plan_bound` | ego 点到 boundary 的安全距离 | 1.0 | command 选中轨迹、mask、boundary score ≥ 0.5 |
| `loss_plan_col` | ego 与 vehicle future 的轴向间距 | 1.0 | command 选中轨迹、mask、agent score/class/mode 筛选 |
| `loss_plan_dir` | ego 运动方向与 divider 切线夹角 | 0.5 | command 选中轨迹、mask、divider score ≥ 0.5 |

例如某帧 command 是 `[0,1,0]`（左转），6 步 mask 是 `[1,1,1,1,0,0]`：reg loss 只监督 mode 1 的前四步；bound/col/dir 也只拿 mode 1，并且只在前四步生效。mode 0 和 mode 2 在这些 planning loss 中没有直接轨迹监督。

配套 `demo.py` 用假设轨迹演示 command 选模式、位移累计和有效步 mask，并用简化的距离算例解释边界/碰撞阈值；它不复刻 map 最近邻、线段相交等完整张量实现。

## 8. 与 UniAD Planning loss 的增量对照

两者的主要区别与前向结构一致：

- VAD forward 不把 command embedding 输入 ego query，而是并行预测 3 条轨迹，由 GT command 在训练时选中一条做 imitation 和环境约束。
- UniAD `PlanningHeadSingleMode` 把 navigation command 融入规划 query，输出 command 条件化的 SDC 轨迹；训练 loss 中有 `loss_ade` 和配置的 collision losses，推理时还可用 occupancy 做非线性碰撞优化。
- VAD 的 bound、collision、direction 三项直接使用 VAD 自己预测的地图线和 agent future；是可微训练约束的一部分，而非只在推理末端做一次优化。

所以 VAD 的 planning loss 不只是“拟合人类驾驶轨迹”：它还会用预测出的道路边界、divider 和周车未来来塑造被 command 选中的候选轨迹。

## 9. 本节小结

- 自车预测为三条 6 步位移轨迹；GT command one-hot 选中唯一接受规划监督的模式。
- 四项 loss 都使用 ego future mask；三项环境约束也只处理 command 选中轨迹。
- boundary 是预测边界向量的近距离/相交处理；collision 是筛选 vehicle 的最高分 motion mode 后做逐轴点间距惩罚；direction 是对齐附近 divider 局部切线。
- VAD 是训练时显式加入地图/agent 约束；UniAD 则用 command 条件化规划，并可在推理时用 occupancy 优化轨迹。

下一节进入多任务 loss 汇总：逐项核对 `loss_dict` 的主 loss、辅助 decoder 层 loss，以及训练框架最终如何把它们加总。
