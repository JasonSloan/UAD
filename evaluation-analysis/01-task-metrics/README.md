# 01：任务评估指标与测试口径

本节沿着 VAD 验证调用链，拆开四类任务的指标：检测、向量地图、agent 运动预测和自车规划。重点不仅是指标公式，也包括预测如何匹配 GT、哪些样本参与平均，以及指标最终在哪里出现。

主要源码：

- 推理与逐样本指标：`projects/mmdet3d_plugin/VAD/VAD.py` 的 `simple_test_pts()`、`compute_motion_metric_vip3d()`、`compute_planner_metric_stp3()`
- 数据集汇总与检测/地图评估：`projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py` 的 `evaluate()`、`_evaluate_single()`
- 规划几何指标：`projects/mmdet3d_plugin/VAD/planner/metric_stp3.py`
- 地图匹配与 AP：`projects/mmdet3d_plugin/datasets/map_utils/tpfp.py`、`tpfp_chamfer.py`
- tiny 评估设置：`projects/configs/VAD/VAD_tiny_e2e.py`
- UniAD 对照：`/mnt/nfs_docker_volume/training-container-space/code/end2end/UniAD/projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py`、`projects/mmdet3d_plugin/uniad/apis/test.py`

## 1. 先看评估数据流

```text
VAD.simple_test_pts()
  ├─ 检测框、地图向量和多模态轨迹预测
  ├─ 逐样本计算 motion metric 和 planning metric
  └─ 把 metric_results 附在每个样本结果上
       └─ VADCustomNuScenesDataset.evaluate()
            ├─ 汇总并打印 motion / planning 指标
            ├─ 格式化预测，跑 nuScenes detection evaluator
            └─ 跑向量地图 AP，返回检测和地图指标字典
```

一个容易忽略的实现细节：`evaluate()` 会打印 motion 与 planning 结果，但最终 `return` 的 `results_dict` 是由 `_evaluate_single()` 填充的，内容主要是 nuScenes 检测指标和地图 AP。因此它们未必会像 `NDS` 一样出现在标准 EvalHook 返回的指标字典中；查看验证日志时要区分“控制台打印的指标”和“evaluate 返回的指标”。

tiny 配置的 `evaluation` 使用 `metric='bbox'`、`map_metric='chamfer'`。模型每次测试仅支持 batch size 1；数据集层再汇总全验证集。

## 2. 检测：nuScenes mAP 与 NDS

检测输出交给定制 nuScenes Detection Evaluator，报告逐类 AP、TP error，以及总体 `mAP` 和 `NDS`。这仍是 3D 检测框评估：预测按类别和中心距离等 nuScenes 规则与 GT 匹配；NDS 综合 mAP 与平移、尺度、朝向、速度、属性误差。

VAD 没有在这段检测评估里计算 tracking ID 连续性指标。检测 mAP/NDS 不能回答“同一个目标跨帧 ID 是否稳定”。

## 3. 向量地图：Chamfer AP，而不是规划约束 loss

tiny 配置使用 `map_metric='chamfer'`。数据集读取预测向量和 map GT，按类别做置信度排序的一对一匹配，再计算 AP。Chamfer 相似度基于预测折线与 GT 折线的双向平均最近点距离；距离阈值为 0.5、1.0、1.5 m，逐阈值 AP 再做平均，输出 `divider`、`ped_crossing`、`boundary` 的 AP 和总体 mAP。

因此地图 AP 衡量“预测的线与标注线是否匹配”，不是上一节的 `loss_plan_bound` 或 `loss_plan_dir`。规划 loss 使用预测地图约束 ego；地图 AP 则直接拿预测地图与 map GT 比较。

配置也允许 `map_metric='iou'`，阈值是 0.50 到 0.95；但 tiny 默认评估走 Chamfer。

## 4. Agent 运动预测：ADE、FDE、MR、EPA

逐样本计算发生在 `VAD.simple_test_pts()`：

1. 检测分数先用 `> 0.6` 筛选。
2. 预测框与 GT 框按中心距离做 Hungarian 匹配，距离超过 2 m 不接受；匹配只考虑代码定义的动态类别。
3. 评估类别汇总为 `car` 和 `pedestrian`；`car` 合并 car、truck、bus、trailer。若 GT future 有效步不足 6 步，它仍可参与 ADE，但只在 6 步全有效时计入完整轨迹 GT/FDE/MR。
4. 每条候选轨迹的逐步位移先累计成位置，再算和 GT 的欧氏距离。

指标定义与分母：

- `ADE`：逐步位置误差的时间平均；在所有 motion modes 中取最小值。汇总时除以有匹配且至少有一个有效 future 步的 GT 数。
- `FDE`：最后一个有效预测时刻的位置误差；代码只对 6 步全有效的 GT 计算，并在 modes 中取最小值。
- `MR`：完整 6 步轨迹中，若最小 FDE 大于 2 m 则记一次 miss；按可计算 FDE 的匹配 GT 数归一化。
- `EPA`：`(hit - 0.5 × FP) / GT`。`hit` 是 FDE 不超过 2 m 的完整轨迹；FP 来自分数过滤后未匹配的相关类别预测；GT 分母是完整 future 有效的 GT 数。

注意：ADE 和 FDE 的最优 mode 是分别取最小值，不保证来自同一条候选轨迹。这里的 ADE/FDE 是匹配 GT 后的 best-of-K 误差，不是把每个 query 的所有输出都直接平均。

## 5. 自车规划：1/2/3 秒 L2 与两种碰撞率

规划评估取 `ego_fut_cmd` 对应的 ego 候选轨迹，把逐步位移累计为轨迹点，再与 GT 比较。`plan_L2_1s/2s/3s` 分别取前 2、4、6 个未来点（数据频率为 2 Hz），计算这些点的平均欧氏位置误差；它们是 horizon 内的平均 L2，不是只看终点的 FDE。

规划碰撞栅格由 **GT future agent boxes** 生成，包含车辆和行人类别，然后计算：

- `plan_obj_col_*`：ego 轨迹中心点落入动态目标占用栅格的比例；
- `plan_obj_box_col_*`：把 ego 近似成固定长宽的矩形后，与动态目标栅格相交的比例。

代码会检查 GT ego 轨迹自身的 box collision，并在 GT 已碰撞的时刻不累加预测碰撞计数。只有 `fut_valid_flag=True`（完整规划未来有效）的样本参与 1/2/3 秒规划指标平均；无效样本的 planning 指标在逐样本函数里置零，并在数据集聚合时跳过。

### 一个与规划训练 loss 的重要差异

上一节的 `PlanCollisionLoss` 在训练时仅保留类别索引 0–4 的 agent（car、truck、construction_vehicle、bus、trailer），不对 pedestrian 加碰撞惩罚；但这里的规划**评估** occupancy 明确包含 human 和 vehicle 类 GT。也就是说，VAD 当前规划碰撞训练项与规划碰撞指标的参与对象并不完全一致。

此外，评估从数据管线读取 GT future 派生的 `ego_fut_cmd`，然后从三条预测中选对应 mode。它衡量的是“给定该 command 后选中轨迹的表现”，不是模型自己预测导航 command 的准确率；部署时 command 通常应来自导航系统，这个 benchmark 口径假设 command 已提供。

## 6. 和 UniAD 对照：相近任务名，评估对象不完全相同

| 任务 | VAD 本仓库 | UniAD 本地实现 |
|---|---|---|
| 3D 感知 | nuScenes detection mAP/NDS | 同样可评估 mAP/NDS，并额外按配置评估 tracking AMOTA/AMOTP 等 |
| 地图 | divider/ped-crossing/boundary 向量 AP，tiny 默认 Chamfer | 栅格化地图类别 IoU（如 drivable、lane、divider、crossing、contour） |
| 运动预测 | 自定义 car/pedestrian 的 best-mode ADE/FDE/MR/EPA | `MotionEval` 可跑 standard、motion mAP/minFDE 和 EPA 等协议，且可分别按 motion/detection category 汇总 |
| 规划 | command 选模式；1/2/3 秒前缀平均 L2、中心点/ego-box collision | 规划头输出轨迹；评估器逐 0.5 秒记录 L2、中心点 collision、ego-box collision，并结合 occupancy GT |
| Occupancy | 没有独立 occupancy 预测分支指标；规划碰撞用 GT agent raster | 独立 occupancy 分支可报告 IoU、PQ/SQ/RQ 等 |

所以不能只比较名字相同的 `ADE` 或 `collision` 数字就下结论：类别合并、匹配策略、有效样本、时间聚合方式和 occupancy 构造都可能不同。做 VAD/UniAD 数值对照时，必须先把这些 protocol 对齐。

## 7. 本节小结

- tiny VAD 的标准返回评估结果主要是 nuScenes detection mAP/NDS 和向量地图 AP；motion/planning 指标由数据集汇总并打印。
- Motion ADE/FDE 使用匹配后的 best-of-K，EPA 还扣除 FP；FDE/MR/EPA 的 GT 统计要求完整 6 步 future。
- Planning L2 是每个 horizon 内各点的平均位置误差；碰撞指标来自 GT 车辆与行人占用栅格。
- Planning 评估用 GT 派生 command 选 VAD 的候选 mode；这不是 command 预测指标。
- VAD 与 UniAD 的 map、motion、occupancy、planning 评估 protocol 不是完全同构，跨模型比较需要先对齐定义。

配套 `demo.py` 用小型假轨迹说明 minADE 与 minFDE 可能来自不同 mode、planning horizon L2 是前缀平均，以及 EPA 的计数公式；它不替代完整 nuScenes evaluator。
