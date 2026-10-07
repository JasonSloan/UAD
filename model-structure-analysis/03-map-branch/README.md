# 03：VAD 地图分支——map query 到向量地图

本节沿着 map 分支的 forward 路径，解释候选地图线如何表示、如何从 BEV 特征中解码，以及最后怎样变成 LiDAR 局部坐标系中的 polyline。重点是先分清“点 query”“地图向量实例”和“地图类别”这三个层次。地图匹配、点损失、方向损失与评估留到后续专题。

主要源码：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py`
- 多任务 head：`projects/mmdet3d_plugin/VAD/VAD_head.py`
- Transformer：`projects/mmdet3d_plugin/VAD/VAD_transformer.py`
- map decoder：`projects/mmdet3d_plugin/VAD/VAD_transformer.py` 中的 `MapDetectionTransformerDecoder`
- map 解码器：`projects/mmdet3d_plugin/core/bbox/coders/map_nms_free_coder.py`
- 点坐标反归一化：`projects/mmdet3d_plugin/VAD/utils/map_utils.py`
- 数据侧局部地图 GT：`dataset-pipeline-analysis/03-vector-map-annotations/README.md`

## 1. 一句话概括

VAD 为地图设置 100 个候选向量实例；每个实例由 20 个点 query 表示。所有点 query 在同一张二维 BEV 特征上解码出有序的 20 个 `(x,y)` 点，再将每条线的点特征平均后预测一个地图类别。推理时在“向量实例 × 类别”分数中取 top-k，并把归一化点坐标还原为 LiDAR 局部坐标。

```text
BEV 特征 [B, 10000, 256]
            │
100 个实例 embedding + 20 个点位置 embedding
            └──> 2000 个 point query（按 vector 分组）
                         │
                   3 层 map decoder
                    ┌────┴─────┐
      每点回归二维坐标       每条 vector 汇总 20 个点特征
       [B,100,20,2]              [B,100,3 类]
                    └────┬─────┘
                         │
       query×类别 top-k、范围过滤、坐标反归一化
                         │
              最多 50 条局部 polyline
```

## 2. 先分清模型里的三个“数量”

tiny 配置中：

| 配置 | 值 | 含义 |
|---|---:|---|
| `map_num_vec` | 100 | 模型可预测的候选地图向量实例数 |
| `map_fixed_ptsnum_per_pred_line` | 20 | 每条预测向量上的有序采样点数 |
| `map_num_classes` | 3 | `divider`、`ped_crossing`、`boundary` 三类 |
| `map_num_query` | 2000 | Transformer 里的点 query 总数，等于 `100 × 20` |
| map decoder 层数 | 3 | 逐层更新点 query 与点坐标 |
| map coder `max_num` | 50 | 推理时最多保留的向量-类别候选数 |

**100 个候选 vector 不等于每帧有 100 条 GT 地图线。**GT 线条数量随场景变化，数据侧会为每条 GT 线采样 20 个点；模型则始终提供 100 个候选实例槽位。`2000` 是按点展开后的 query 数，解码输出重新分组后仍是 100 条候选线。

## 3. 每个点 query 如何带有“属于哪条线”的信息

配置使用 `map_query_embed_type='instance_pts'`。head 分别学习：

- 100 个 `map_instance_embedding`：表示候选向量实例身份；
- 20 个 `map_pts_embedding`：表示该向量上的点位置索引。

两者相加后形成 `[100, 20, 512]`，再按“先 vector、再点”的顺序展平为 `[2000, 512]`。`512=2×256`，随后和 object query 一样，拆成 query 内容与 query 位置，各为 256 维。

这样第 `v` 条候选线的第 `p` 个点 query 同时带着“我是第 v 条候选线”和“我是这条线上的第 p 个点”两种可学习身份。展平并没有丢掉分组关系：head 后面会用 `view(B, 100, 20, ...)` 恢复 vector/point 两层结构。

因此这里的点不是彼此无关的 2000 个地图目标。20 个点 query 一组共同组成一条 polyline；点索引给出模型表示中的顺序。这个顺序是线的参数化顺序，不等同于现实世界给道路规定了唯一的行驶方向。训练时 GT 点序对齐和 shift/方向相关处理会在 loss 专题再仔细看。

## 4. Map decoder 怎样预测点

map query 和检测 query 使用同一份当前 BEV 表征，但走独立的 map decoder。tiny 配置的 map decoder 也有 3 层，层内包含 query self-attention、对 BEV 的可变形 cross-attention 和 FFN。

每个点 query 有自己的二维 reference point。map decoder 只把 reference point 的 `(x,y)` 传给二维 BEV 注意力；每层 map 回归头输出二维残差，用它更新该点下一层的 reference point。和检测分支一样，这是在二维 BEV 平面采样，不存在 z 轴地图点预测。

map head 对每个点 query 输出 2 个数。加上当前 reference point 的 logit 后做 sigmoid，得到归一化的 `(x,y)`；三层 decoder 的预测逐层细化。tiny 配置最终张量的核心形状是：

```text
map_hs                 [层数, B, 2000, 256]
map_all_pts_preds      [层数, B, 100, 20, 2]
map_all_cls_scores     [层数, B, 100, 3]
map_all_bbox_preds     [层数, B, 100, 4]
```

这里每条线有一组 20 个点坐标，但只有一个类别分数向量。类别头先把最后一层特征恢复成 `[B,100,20,256]`，沿 20 个点的维度取平均，再预测 3 个类别 logits。因此一个 vector 实例的类别是对整条线预测的，不是每个点单独分类。

## 5. 点坐标、外接框与坐标系

每个点的模型输出是 `[0,1]` 范围内的归一化坐标。按照配置 `point_cloud_range=[-15,-30,-2,15,30,2]`，二维点还原为 LiDAR 局部平面坐标的公式是：

```text
x_lidar = x_norm * (15 - (-15)) + (-15)
y_lidar = y_norm * (30 - (-30)) + (-30)
```

所以 map polyline 的最终 `(x,y)` 是以当前 LiDAR/自车附近局部坐标表示，不是 nuScenes global 地图坐标；数据侧 GT 的坐标系可回看上一份 vector-map annotations 文档。

head 还会对每条线的 20 个点分别取 `xmin/xmax/ymin/ymax`，构造一个轴对齐外接矩形。这个矩形不是 polyline 本身，而是辅助表示/过滤用的 envelope。`map_transform_box()` 先从点得到归一化 `cx,cy,w,h`；推理解码后转成物理坐标下的 `[xmin,ymin,xmax,ymax]`。

配套 `demo.py` 用两条合成线演示：扁平点列表如何按 vector 分组、如何由点得到包围范围，以及归一化 `(x,y)` 如何映射回上述 LiDAR 局部范围。

## 6. 推理时怎样挑选并返回地图线

`MapNMSFreeCoder` 只取最后一层预测。与检测 coder 类似，它对每条 vector 的 3 个类别 logits 做 sigmoid，将 `[100,3]` 展平后取 top-k。当前配置 `max_num=50`，所以最多保留 50 个“vector-类别”候选；同一个 vector 如果多个类别分数都很高，理论上可能以多个类别候选进入 top-k。这里不做传统 NMS，配置也没有启用 `score_threshold`。

选出候选后，coder 用候选 vector 索引同步取类别对应的线点和 envelope，再将两者反归一化；随后按 envelope 是否落在 `post_center_range` 范围内过滤。最后 `VAD.simple_test_pts()` 把地图结果放入：

- `map_boxes_3d`：局部坐标下的轴对齐 envelope；
- `map_scores_3d`、`map_labels_3d`：向量实例的类别分数和类别；
- `map_pts_3d`：局部坐标下的 polyline 点序列。

这里的“3d”是项目结果字段命名；这条地图线的有效几何仍是 BEV 平面中的二维 `(x,y)` 点，不含点级 z 坐标。

## 7. 地图分支如何交给后续任务

VAD 不只把 map 分支用于最终地图评估。head 内部会保留最后一层 map query 特征，将每组 20 个点的特征交给 `lane_encoder` 汇总成线级特征；同时保留预测线点/线位置及类别置信度。运动预测分支据此筛选并读取 agent 附近地图特征，自车规划分支也会读取候选地图线，与自车 query 做交互。

所以有两种不同的 map 输出用途：

- **几何/类别输出**：解码后的 polyline、类别、分数，供推理结果和地图评估使用；
- **内部特征输出**：map query hidden states 与预测位置，供 motion/planning 的注意力模块使用。

后续任务不需要等推理 coder 先筛出最终 50 条线才启动；任务 head 在本次 forward 中直接消费 map head 内部的预测与特征。下一节运动预测会继续追踪这个接口。

## 8. 本节暂不展开的 loss

训练时 map 分支每层都会产生类别、点坐标和外接框预测。GT vector 与预测 vector 的匹配涉及 Hungarian assignment；点监督还涉及固定采样点、GT 点序 shift，以及方向一致性。配置里的 `map_gt_shift_pts_pattern='v2'`、`loss_map_pts` 和 `loss_map_dir` 都与这些细节相关。

本节先记住前向结构与张量分组，不提前把匹配代价和点/方向 loss 混进来。后续会把它们与数据侧 GT 的点序表示对应起来分析。

## 9. 本节小结与下一节

- 100 个候选地图 vector，每条由 20 个有序点 query 表示，共 2000 个 point query。
- 同一条线的 20 个点特征平均后预测一个线级类别；点回归则逐点输出二维坐标。
- 点坐标从归一化平面还原到 LiDAR 局部坐标；由点计算出的外接框只是 envelope，不等于 polyline。
- 推理最多保留 50 个 vector-类别候选，使用 top-k 与范围过滤，不做传统 NMS。
- map query 的内部特征会被后续运动预测与自车规划继续使用。

下一节进入运动预测分支：检测到的 agent query 如何扩展成多模态未来轨迹，怎样与地图特征交互。
