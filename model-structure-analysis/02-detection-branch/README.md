# 02：VAD 检测分支——object query 到 3D 框

本节只拆 object query 检测路径：它如何从 BEV 特征中读出对象信息、逐层细化 3D 参考点、构造分类和框回归结果，以及推理时如何解码。轨迹如何使用检测 query 会在第 04 节展开；这里会先标出两者的接口。

主要源码：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py`
- 检测头：`projects/mmdet3d_plugin/VAD/VAD_head.py`
- Transformer：`projects/mmdet3d_plugin/VAD/VAD_transformer.py`
- 检测 decoder：`projects/mmdet3d_plugin/VAD/modules/decoder.py`
- 框编码/解码：`projects/mmdet3d_plugin/core/bbox/util.py`、`projects/mmdet3d_plugin/core/bbox/coders/fut_nms_free_coder.py`
- 推理结果包装：`projects/mmdet3d_plugin/VAD/VAD.py`

## 1. 先用一句话概括

VAD 有 300 个可学习的 object query。每个 query 在 BEV 特征上经过 3 层 decoder，逐层输出类别分数和 3D 框参数；同一组 query 特征还会接入后续的 agent 运动预测。推理导出检测结果时，VAD 把“query × 类别”的分数摊平做 top-k，而不是调用传统 NMS。

```text
当前帧 BEV 特征 [B, 10000, 256]
             │
300 个 object query（内容 + 位置）
             │
3 层检测 decoder：query 自注意力 + 对 BEV 的可变形交叉注意力
             │ 每层各自的分类头 / 回归头
             ├──> all_cls_scores [3, B, 300, 10]
             ├──> all_bbox_preds [3, B, 300, 10]
             └──> 最后一层 query 特征 + 框中心 ──> 后续 agent 运动预测

推理导出：最后一层分类分数 ─> sigmoid ─> query×类别 top-k
                              └> 选框、解码、范围过滤 ─> 3D 检测结果
```

图中的 10 类是 tiny 配置的类别数；默认分类采用 sigmoid，因此没有额外的“背景类”通道。

## 2. Query 从哪里来，如何读取 BEV

配置 `num_query=300`、`as_two_stage=False`。因此 query 不是由 BEV encoder 生成的 proposal，而是来自可学习的 `nn.Embedding(300, 2 * embed_dims)`。tiny 配置 `embed_dims=256`，所以每个 query embedding 是 512 维，在 `VADPerceptionTransformer.forward()` 中沿通道拆成两半：

- `query`：初始 query 内容，形状 `[300, B, 256]`。
- `query_pos`：query 的位置编码，形状 `[300, B, 256]`。

初始 `reference_points` 由 `query_pos` 经过线性层和 sigmoid 得到，形状 `[B, 300, 3]`。这三个归一化坐标对应 point-cloud 范围内的 x、y、z 参考位置。但三维参考点不等于三维采样位置：检测 decoder 传给 BEV 注意力的只有 x/y；z 不用于读取 BEV 特征。同时 query 之间做 self-attention，使候选对象能够彼此交换上下文。

因此，`reference_points` 的 x/y 是 BEV 平面上的采样参考；x/y/z 则共同作为逐层细化 3D 框中心的归一化参考值。它们都不是最终输出框的物理坐标，最终中心会由 head 映射到配置的 `point_cloud_range`。

## 3. 三层 decoder 如何逐层细化参考位置

tiny 配置的检测 decoder 有 3 层，`with_box_refine=True`，所以分类头和回归头按层复制，每层都有自己的一套参数。`DetectionTransformerDecoder` 每层大致执行：

1. 只用当前参考点的 x/y 引导 query 从二维 BEV 特征图采样；采样偏移也是二维的，不沿 z 轴采样。
2. 得到更新后的 query 特征。
3. 当前层回归头产生框参数残差。
4. 将 x、y 残差加到参考点 x/y 的 logit 上；将回归量第 4 维（零起始索引 4，也就是 z 对应分量）加到参考点 z 的 logit 上。
5. sigmoid 得到下一层的参考点，并 `detach()` 后交给下一层。

可写成（这两行表示下一层框中心参考值的更新，不表示 BEV 上的采样）：

```text
ref_xy_next = sigmoid(delta_xy + inverse_sigmoid(ref_xy))
ref_z_next  = sigmoid(delta_z  + inverse_sigmoid(ref_z))
```

`detach()` 表示后层继续用前层更新的位置，但不会通过“下一层参考点”这条路径把梯度传回前层；各层预测本身仍各自参与训练监督。

### 关键澄清：z 不是 BEV 注意力的采样轴

检测 decoder 的输入 BEV 被展平为 `[B, bev_h * bev_w, C]`，并通过 `spatial_shapes=[[bev_h, bev_w]]` 告诉注意力模块它是一张二维网格。decoder 把三维 `reference_points` 切成 `reference_points[..., :2]` 后才传入注意力；可变形注意力预测的 `sampling_offsets` 最后一维也是 2，采样位置计算为“二维参考点 + 二维偏移”。所以这里每个 query 读取的是 BEV 平面上的若干 `(x, y)` 位置，没有 `(x, y, z)` 采样。

z 的用途在另一条计算上：每层回归头预测一个 z 残差，decoder 用它更新下一层的 z 参考值；head 随后把该参考值与回归残差组合，得到 3D 框中心高度，再映射到 point-cloud 的 z 范围。可以把它理解为“在二维 BEV 特征上找对象，同时预测对象的高度”，而不是“沿高度轴从 BEV 特征里取样”。

BEVFormer encoder 里的情况不同：它会为 BEV 网格位置构造多个高度锚点，将这些三维点投影到各相机图像上采样图像特征，再聚合成 BEV 特征。高度方向的采样发生在 encoder 从图像构建 BEV 时；到检测 decoder 阶段，输入已经是二维 BEV 表征，不再沿 z 采样。

在 head 中，第 0 层预测使用初始参考点，第 1/2 层预测分别使用 decoder 返回的前一层更新参考点。回归头对中心残差加参考点后做 sigmoid，再按 `point_cloud_range=[-15,-30,-2,15,30,2]` 映射到物理坐标：

```text
x = sigmoid(x_logit) * (x_max - x_min) + x_min
y = sigmoid(y_logit) * (y_max - y_min) + y_min
z = sigmoid(z_logit) * (z_max - z_min) + z_min
```

其他框分量不做这个中心坐标映射。

## 4. 框回归的 10 维与 9 维

这是读 VAD 检测源码时最值得记清楚的维度区别。

回归头输出 10 维，训练目标和 coder 输入排列为：

| 索引 | 表示 | 说明 |
|---:|---|---|
| 0 | `x` | 框中心 x，head 输出时已映射到 point-cloud 物理范围 |
| 1 | `y` | 框中心 y，head 输出时已映射到 point-cloud 物理范围 |
| 2 | `log(w)` | 宽度的对数 |
| 3 | `log(l)` | 长度的对数 |
| 4 | `z` | 框中心 z，head 输出时已映射到 point-cloud 物理范围 |
| 5 | `log(h)` | 高度的对数 |
| 6 | `sin(yaw)` | 朝向角正弦 |
| 7 | `cos(yaw)` | 朝向角余弦 |
| 8 | `vx` | x 方向速度 |
| 9 | `vy` | y 方向速度 |

`normalize_bbox()` 将 GT 的宽、长、高取对数，并把 yaw 展开成 sin/cos，因此监督目标是 10 维。预测解码时，`denormalize_bbox()` 对尺寸取指数、用 `atan2(sin, cos)` 恢复 yaw，并拼成 9 维结果：

为什么不只回归 `sin(yaw)`？因为正弦有歧义：`sin(30°)=sin(150°)=0.5`，只知道正弦无法判断象限；余弦的符号和值补足了方向信息。训练目标里的 `(sin(yaw), cos(yaw))` 来自同一个 GT 角度，理想情况下满足 `sin²+cos²=1`。

推理时两个回归输出是网络独立预测的，代码没有额外强制它们落在单位圆上。如果预测的 sin/cos“不匹配”或长度不为 1，`atan2(sin_pred, cos_pred)` 仍会把它们当作平面向量 `(x=cos_pred, y=sin_pred)`，取向量方向作为角度，向量长度会被忽略。例如预测 `sin=0.5, cos=0.4`，解码角度是 `atan2(0.5, 0.4) ≈ 51.3°`。训练中的框回归损失会分别把两个分量拉向 GT 的 sin/cos 值；若两者都接近 0，方向会变得不可靠。

```text
[x, y, log(w), log(l), z, log(h), sin(yaw), cos(yaw), vx, vy]
                           │ 解码
                           ▼
[x, y, z, w, l, h, yaw, vx, vy]
```

所以 `VADHead.code_size` 默认是 10，表示回归编码长度；`denormalize_bbox()` 的输出和最终 3D 框是 9 维。`fut_nms_free_coder.py` 的 docstring 把回归头输入也写作 9 维，是源码注释过时，不能据此理解实际张量。

另一个细节：`denormalize_bbox(normalized_bboxes, pc_range)` 的 `pc_range` 参数在该函数中没有被使用。原因是 head 已经先把中心 x/y/z 从 sigmoid 归一化值映射到了物理范围，coder 只需还原尺寸与朝向。

## 5. 最后一层预测如何变成检测结果

`CustomNMSFreeCoder.decode()` 只取最后一层的分类、框和轨迹预测。对 batch 中每个样本，`decode_single()`：

1. 对分类 logits 做 sigmoid，得到每个 query 对每个类别的独立分数。
2. 把 `[300, 10]` 分数展平成 3000 个“query-类别”候选，取最高的 100 个（tiny 配置 `max_num=100`）。
3. 从展平索引还原类别 `index % 10` 和 query `index // 10`，并用 query 索引取对应框和轨迹。
4. 将 10 维回归编码还原为 9 维框，再按框中心的 x/y/z 做 `post_center_range` 过滤。
5. 返回框、分数、类别和与这些候选对应的轨迹。

这里没有传统 NMS，也不保证每个 query 最终只保留一个类别：同一 query 若对多个类别都排进 top-k，可能以多个类别候选出现。配置中的 `score_threshold` 为 `None`，因此实际主要经过 top-k 和中心范围过滤。

随后 `VADHead.get_bboxes()` 根据 `img_metas[i]['box_type_3d']` 包装 3D 框，并把 z 从中心高度改成底面中心约定（`z -= h / 2`）。`VAD.simple_test_pts()` 再把结果放进 `boxes_3d`、`scores_3d`、`labels_3d`、`trajs_3d` 等字段。

## 6. 检测结果与运动预测不是“先 top-k 再预测”

推理导出 top-k 框发生在 `get_bboxes()` 的 coder 解码阶段。但在 `VADHead.forward()` 内部，运动预测拿的是检测 decoder 最后一层的全部 300 个 query 特征，以及对应的全部 300 个中心坐标。之后可根据 `motion_det_score` 等设置屏蔽低置信 query，再做 agent-agent / agent-map 交互。

因此要区分两条路径：

- **多任务内部计算**：300 个 object query 继续供运动预测和规划模块使用；这不是最终 top-k 结果列表。
- **检测结果导出**：coder 将 300×10 的类别候选排序，取最多 100 个，再包装成评测/可视化结果。

这一点和 UniAD 的学习经验可以这样对照：这里的 object query 是当前帧的候选检测槽位，不是跨帧持续维护身份的 track query。VAD 通过时序 BEV 利用历史信息，但不能因此把检测 query 误认为具备稳定跟踪 ID。

## 7. 本节暂不展开的 loss

训练时每个 decoder 层都计算检测监督。`loss_single()` 调用 Hungarian assigner 建立 query 与 GT 的匹配，再计算分类与框回归监督；VAD 还在匹配对象上监督未来轨迹及其模式分类。框目标经 `normalize_bbox()` 转成前述 10 维编码。

本节先记住“每层都有预测、每层都可监督”和“GT 框被编码成 10 维”即可。匹配代价、正负样本、分类损失、框损失和轨迹损失放在后续 loss 专题中推导。

## 8. 小实验

运行同目录的 `demo.py`，用纯 Python 演示两件事：10 维框表示如何解码成 9 维框，以及 flatten top-k 如何从 query×类别分数中还原 query 与类别。示例不加载 VAD checkpoint，也不依赖项目编译的 CUDA 算子。

## 9. 本节小结与下一节

- tiny 配置有 300 个可学习 object query，检测 decoder 为 3 层。
- 参考点在 decoder 层间迭代细化，head 把中心映射到 point-cloud 物理坐标。
- 框回归编码是 10 维，最终框表示是 9 维；coder 注释中的 9 维输入描述已过时。
- 推理检测采用 query×类别 top-k，不做传统 NMS；top-k 导出并不决定内部运动预测使用哪些 query。

下一节转到 map 分支：它与检测分支共享 BEV，但有自己的 map query、类别预测和 polyline 点预测，并为后续运动预测/规划提供显式向量地图特征。
