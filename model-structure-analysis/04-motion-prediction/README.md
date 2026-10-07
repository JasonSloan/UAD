# 04：VAD 运动预测分支——检测 query 到多模态未来轨迹

本节沿 `VADHead.forward()` 追踪运动预测分支：检测 query 如何扩展成多个未来模式，如何在 agent 之间交换特征，以及每个模式如何读取附近地图特征。损失函数与运动评估指标留到后续专题。

主要源码：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py`
- 多任务 head：`projects/mmdet3d_plugin/VAD/VAD_head.py`，重点看 `forward()`、`_get_target_single()` 和 `select_and_pad_pred_map()`
- 未来轨迹数据生成：`tools/data_converter/vad_nuscenes_converter.py` 中生成 `gt_agent_fut_trajs` 的部分
- 运动结果评估：`projects/mmdet3d_plugin/VAD/VAD.py` 中的 `compute_motion_metric_vip3d()`

## 1. 一句话概括

VAD 不为每个 agent 单独运行一个逐时刻的循环解码器，而是把每个检测 query 与 6 个可学习的 mode embedding 相加，形成 6 个“agent-模式”query。它们先通过 motion decoder 交换特征，再分别对附近的地图向量做 cross-attention；最后一个 MLP 一次性回归每个模式未来 6 步的二维位移，并为 6 个模式各输出一个置信分数。

```text
当前帧 object query [B, 300, 256] + 检测位置
                  │
          与 6 个 mode embedding 相加
                  ↓
          [B, 300, 6, 256]
                  │
  motion decoder：agent/mode query 间特征交互
                  │
  每个 agent-mode 读取附近 map query 特征
                  ↓
          拼接两路特征 [B, 300, 6, 512]
             ┌────┴─────┐
   回归未来 6 步 xy 位移   预测 6 个 mode 分数
```

## 2. 预测对象来自检测 query，不是跨帧 track

motion 分支输入的是检测 decoder 最后一层的 `hs[-1]` 和检测框中心 `outputs_coords_bev[-1]`。tiny 配置有 300 个 object query，因此训练 forward 中先为 300 个 query 都构造运动特征；它们不是持久化的跨帧 track ID，也不是每个 GT agent 直接对应一个 query。

训练时，检测分支的 Hungarian 匹配把正样本 query 与 GT 框对应起来，匹配到的 query 才有该 GT agent 的未来位移监督。未匹配 query 会进入运动模式分类的背景标签处理。推理时，模型再将对应 query 的轨迹与检测结果一起返回。

因此可把分支输入理解为：“当前帧检测器认为可能是 agent 的候选及其特征”，而不是先完成跨帧跟踪后拿到的稳定身份序列。这与 UniAD 以 track query 为核心的预测路径是一个值得留意的差别。

## 3. mode query 表示多种完整未来，不表示未来时刻

head 默认 `fut_mode=6`、`fut_ts=6`。`motion_mode_query` 是 6 个可学习向量；对每个检测特征 `h_a`，代码执行：

```text
q[a,m] = h[a] + mode_embedding[m]
```

其中 `m` 是整条未来轨迹的模式索引，不是时间步索引。每个 mode query 后面会一次性回归 6 个时间点的 `(dx,dy)`，即每个模式输出 12 个数。因此预测张量整理后是：

```text
all_traj_preds       [层数, B, 300, 6, 12]
all_traj_cls_scores  [层数, B, 300, 6]
```

轨迹头先把 6 个时刻的二维位移展平为最后一维 12；损失与后续处理再将其视作 `[B,300,6,6,2]`。`mode 0` 到 `mode 5` 可以分别表示不同的可能走向；它们不是“第 0 到第 5 帧”。这也是多模态预测的含义：同一个 agent 可同时给出多条候选未来，再为候选模式打分。

## 4. motion decoder：让候选 agent/mode 特征彼此交互

motion query 在进入 decoder 前展平为 `[A×M, B, D]`，tiny 配置即 `[1800, B, 256]`。其位置编码来自每个检测 query 的 BEV 中心，并在同一个 agent 的 6 个 mode 间重复。

配置中的 motion decoder 有一层，操作顺序是 `cross_attn → norm → ffn → norm`。调用时 `query`、`key`、`value` 都是这批 motion query，所以注意力是在当前样本的全部 agent-mode 候选之间做信息交互。这里没有显式的未来时间 token 序列；时间维度是在后面的轨迹回归输出中一次性产生的。

配置没有设置 `motion_det_score`，因此这一步没有通过该阈值预先屏蔽低分检测 query。后续训练匹配和推理检测结果筛选负责处理候选是否为有效目标。

## 5. motion-map 交互：每个 agent-mode 读取附近地图

地图分支最后一层的点 query 特征形状为 `[B, 100, 20, 256]`。`LaneNet` 将一条 polyline 的 20 个点特征编码成一个线级特征 `[B,100,256]`。随后 `select_and_pad_pred_map()` 为每个 agent 选择地图候选：

1. 对每条地图线取类别最大 sigmoid 分数，要求高于 `map_thresh=0.5`。
2. 用该地图线中距离坐标原点最近的采样点，作为这条线的位置代表。
3. 把地图代表位置与 agent 检测中心比较，只让距离不超过 `dis_thresh=0.2` 的地图候选参与该 agent 的注意力；其余候选通过 padding mask 屏蔽。
4. 若 `pe_normalization=True`，地图代表位置会减去对应 agent 的位置，再送入位置编码网络。

要注意，源码用于距离筛选的是“线的位置代表点”，不是计算 agent 到整条 polyline 的精确最短距离。另外，检测中心和地图点此处使用归一化 BEV 坐标，因此配置中的 `0.2` 不是 0.2 米。

为每个 agent 的 6 个模式都准备同一组附近地图特征，然后 motion-mode query 作为 query、地图线特征作为 key/value 做 cross-attention。各模式的 query 特征因此可以根据周边道路几何得到不同表示。

## 6. 轨迹回归值是逐步位移

数据转换器逐个读取 agent 后续时刻的 3D 框，将框中心变换到当前 LiDAR 坐标系，再计算相邻时刻中心差：

```text
gt_fut_trajs[t] = center[t+1] - center[t]
```

因此监督目标是每一步的 `(dx,dy)` 位移，不是直接相对当前帧的绝对未来坐标。网络对每个 agent-mode 特征回归 `[6,2]` 位移序列。要画出相对当前 agent 的未来位置轨迹，需要沿时间维累加：

```text
位置[0] = 位移[0]
位置[1] = 位移[0] + 位移[1]
...
```

源码里的 best-mode 选择与评估也会先对预测和 GT 做 `cumsum`，再比较累计路径。配套 `demo.py` 用一个小例子演示 mode 展开、张量形状和位移累加。

## 7. 输出特征如何流向规划分支

motion decoder 特征与 motion-map cross-attention 特征沿通道拼接成 `[B,A,6,512]`。轨迹回归头和模式分类头分别读取这份融合特征，输出轨迹位移及模式分数。

与此同时，VAD 会把 6 个 mode 的融合特征展平后送入 `agent_fus_mlp`，压成每个检测 query 一个 256 维 agent 特征。规划分支随后按检测分数筛选 agent query，并让 ego query 读取这些 agent 特征。因此运动分支的产物不只有可视化的多条轨迹，也包含供自车规划使用的 agent 上下文表示。

## 8. 本节暂不展开的 loss 与指标

轨迹回归使用 best-of-6 机制：在 6 个模式中按累计轨迹终点误差选择一个 best mode，再对其与 GT 的逐步位移计算回归损失；模式分类则以最接近 GT 的模式为正类，未匹配检测 query 使用背景类。数据缺失 mask 会屏蔽无效未来时间步。

这些机制以及 minADE/minFDE 等评估实现留到后续的 loss 与评估专题，避免把前向数据流和监督细节混在一起。

## 9. 本节小结与下一节

- 运动预测 query 由当前检测 query 和可学习 mode embedding 相加得到；mode 代表候选未来，不代表时间步。
- tiny 配置中每个检测 query 预测 6 种模式，每种模式包含 6 个未来时刻、每个时刻两个位移值。
- agent-mode query 先相互交互，再读取附近地图线特征；地图距离筛选基于归一化坐标中的代表点。
- 轨迹监督是 LiDAR 局部坐标系下的逐步位移，累计后才是相对当前位置的路径。
- 融合后的 agent 特征会继续传给 ego planning 分支。

下一节进入自车规划分支：ego query 怎样汇总历史自车信息、agent 与地图特征，并回归自车未来轨迹。
