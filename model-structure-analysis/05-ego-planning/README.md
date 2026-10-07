# 05：VAD 自车规划分支——ego query 到未来轨迹

本节追踪 `VADHead.forward()` 中的 planning 路径：ego query 如何先读取 agent 特征、再读取地图线特征，最后回归自车未来轨迹。重点是搞清楚 tiny 配置下哪些输入实际启用，以及三条输出轨迹和驾驶指令的关系；损失与规划指标留到后续专题。

主要源码：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py`
- 多任务 head：`projects/mmdet3d_plugin/VAD/VAD_head.py` 中 `forward()` 的 planning 部分
- 规划监督：同文件中的 `loss_planning()`，本节只追踪输入与数据流
- 自车轨迹和 command 标注：`tools/data_converter/vad_nuscenes_converter.py`
- 测试时轨迹选择：`projects/mmdet3d_plugin/VAD/VAD.py` 中 `simple_test_pts()` 与规划评估部分
- UniAD 对照：`/mnt/nfs_docker_volume/training-container-space/code/end2end/UniAD/projects/mmdet3d_plugin/uniad/dense_heads/planning_head.py`

## 1. 一句话概括

VAD 用一个可学习的 ego query 作为规划起点。它先通过 cross-attention 读取 agent 特征，再以更新后的 ego query 读取地图线特征；将两路结果拼接后，MLP 一次性输出三条候选自车轨迹，每条包含 6 步二维位移。

```text
可学习 ego query
      │ cross-attention
      ▼
筛选后的 agent 特征 ──> ego-agent 特征
                              │ cross-attention
预测地图线特征 ──────────────┤
                              ▼
                   拼接两路特征 [B, 1, 512]
                              │ MLP
                              ▼
              3 条候选轨迹 [B, 3, 6, 2]
                   每步是 (dx, dy)
```

## 2. tiny 配置下的 ego query 从哪里来

head 支持两种 ego 初始特征来源：

- 如果启用 `ego_his_encoder`，就把历史自车位移序列编码成 ego 特征。
- 如果未启用，就使用一个可学习的 `ego_query` embedding。

当前 `VAD_tiny_e2e.py` 明确设置 `ego_his_encoder=None`，所以本配置使用可学习 embedding；数据管线虽然仍传入 `ego_his_trajs`，但这个分支不会把它编码进 ego query。类似地，`ego_lcf_feat_idx=None`，所以当前配置也不把 CAN bus/车辆状态特征拼入规划 MLP。

这不代表模型完全没有时间上下文：BEVFormer 仍可利用历史帧 BEV。区别在于，规划头没有直接把显式 ego 历史轨迹或 LCF 状态作为此处的输入特征。

## 3. 先与 agent 交互，再与地图交互

### Ego-agent attention

运动分支把每个检测 query 的 6 个 motion-mode 特征展平后，经 `agent_fus_mlp` 压成一个 256 维 agent 特征。tiny 配置的 `query_thresh=0.0`，helper 对分类分数做 sigmoid 后按阈值筛选；因 sigmoid 分数通常大于 0，这个设置实际上不会像常见的正阈值那样积极剔除低分 query。每个 agent 的检测中心用于位置编码。

ego query 作为 attention query，agent 特征作为 key/value，得到 `ego_agent_query`。因此规划头读取的是 agent 的融合运动特征，而不是把每一条 agent 未来轨迹直接拼到自车轨迹输出上。

### Ego-map attention

地图分支将每条向量线的 20 个点特征经 `LaneNet` 汇总成线级特征。每条线取离归一化 BEV 坐标原点最近的采样点作为位置代表；map 分类分数经 sigmoid 后由 `query_thresh` 筛选，位置代表用于位置编码。随后以 `ego_agent_query` 作为 query、地图线特征作为 key/value，得到 `ego_map_query`。

注意执行顺序：第二次注意力的 query 是第一次得到的 `ego_agent_query`。也就是说，最终 ego 表征是在已经读取 agent 信息的基础上继续读取地图信息。

## 4. 三条轨迹模式和 command 的真实关系

head 默认 `ego_fut_mode=3`、`fut_ts=6`。tiny 配置中的回归 MLP 输入为 agent/map 两个 256 维特征拼接，输出维度为 `3 × 6 × 2 = 36`，reshape 后为：

```text
ego_fut_preds: [B, 3, 6, 2]
```

每条模式是 6 步 `(dx,dy)` 位移序列。沿时间维做累计和后，才得到相对于当前自车位置的未来坐标轨迹。

这里有个容易误解的点：**当前 VAD head 的 forward 不接收 `ego_fut_cmd`，而是无条件并行输出三条轨迹。**训练损失用数据中的 one-hot command 选择对应模式进行监督；数据转换器按未来轨迹末端横向偏移生成 command，顺序为右转、左转、直行。测试评估代码也从数据中取这个 command 来选轨迹，再与 GT 比较。

因此在这份实现中，不能把 `ego_fut_cmd` 理解成已经输入规划网络的在线导航指令。评测路径使用的是由未来标注生成的 command 来挑选候选模式；如果要做真实部署，需要接入在线导航意图，或另外设计不依赖 GT command 的候选选择策略。

配套 `demo.py` 会演示三条轨迹如何按 command 索引选择并从位移累加成位置。它特别标明：command 选择是演示评测侧后处理，不是 VAD ego forward 的输入。

## 5. 与 UniAD PlanningHead 的关键区别

二者都从特征预测自车轨迹，也都需要理解道路和周边交通；但规划头拿到上下文的路径不同：

| 对比点 | VAD tiny | UniAD `PlanningHeadSingleMode` |
|---|---|---|
| 规划 query 来源 | 可学习 ego query；当前配置不编码显式 ego 历史轨迹 | MotionHead 提供 SDC track query 和 SDC trajectory query |
| 周围信息 | 先显式 cross-attention 到 agent 特征，再 cross-attention 到预测地图线特征 | 将 SDC query 与 command embedding 融合，再 cross-attention 到 BEV 特征 |
| 在线 command 的位置 | 不进入 VAD ego forward；训练/评估用 command 关联或选择三条候选 | navigation command embedding 直接拼进规划 query，显式条件化规划 |
| 轨迹输出 | MLP 并行输出 3 条 6 步位移轨迹 | Planner 根据输入 query 输出一条由 command 条件化的轨迹 |
| occupancy 的使用 | 本分支不直接读取 occupancy | 推理可用 occupancy 做碰撞优化 |

这也和上一节运动预测的区别相呼应：UniAD 把 SDC 当作运动预测目标之一，再把 SDC 的 motion query 交给 planning；VAD 则另设 ego query，让它直接汇总 agent/map 特征生成自车轨迹。

## 6. 输出和后续处理

`VADHead.forward()` 返回所有三个模式的逐步位移。测试封装将 `ego_fut_preds` 和数据提供的 `ego_fut_cmd` 一起放进结果；规划指标代码按 command 取其中一条，并对预测与 GT 位移做累计和后计算轨迹指标。

本节暂不讨论 imitation、边界、方向与碰撞 loss 的数学形式，也不展开评估指标定义；这些将放在后续专门章节，届时再核对各个监督项实际使用的是哪条模式、哪种坐标表示。

## 7. 本节小结

- tiny 配置以学习到的 ego query 起步，不使用显式 ego-history encoder 或 LCF 特征；时序 BEV 仍能提供历史视觉上下文。
- ego query 先 cross-attend agent，再 cross-attend map，最终以两路特征拼接后的 512 维向量预测轨迹。
- 输出是 `[B,3,6,2]` 的三条逐步位移轨迹，累加后才是未来位置。
- 当前 VAD ego forward 不接收 command；数据 command 用于训练监督和测试时挑选模式。它由未来轨迹生成，不能直接当作部署时可用的在线 command。
- UniAD 则把在线 command embedding 放入规划 query，并让规划 query 读取 BEV；两边规划信息流的结构差别明显。

至此，BEV 感知、检测、地图、运动预测和自车规划的 forward 主线已走完。下一阶段可以按计划拆解各分支的 loss 与评估指标。
