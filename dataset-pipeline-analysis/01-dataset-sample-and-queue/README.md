# 01：Dataset sample 与时序队列

本节跟踪单个 nuScenes info 如何变成一个训练样本，重点看三个函数：

```text
get_data_info(index)
    → prepare_train_data(index)
        → union2one(queue)
```

源码：`projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py`。

本节配套演示：`python dataset-pipeline-analysis/01-dataset-sample-and-queue/demo.py`。脚本用合成帧复现历史帧检查和 `deepcopy(example)` 入队的关键逻辑，不需要 nuScenes 数据或 OpenMMLab 依赖。

## 1. `get_data_info()`：从 info 记录组装单帧输入

函数从 `self.data_infos[index]` 取一条 temporal info，构造传给 pipeline 的 `input_dict`。

主要内容分为四组：

| 内容 | 典型字段 | 后续用途 |
|---|---|---|
| 样本与时序 | `sample_idx`、`scene_token`、`frame_idx`、`prev_idx`、`next_idx`、`timestamp` | 区分场景、检查帧顺序、组织 history queue |
| 自车状态 | `can_bus`、`ego_his_trajs`、`ego_fut_trajs`、`ego_fut_masks`、`ego_fut_cmd`、`ego_lcf_feat` | temporal BEV、自车规划监督和状态输入 |
| 相机几何 | `img_filename`、`lidar2img`、`camera2ego`、`camera_intrinsics` | 读取多相机图像并将相机特征投影到 BEV |
| 3D 标注 | `ann_info` 中的 3D boxes、labels、`attr_labels` | 目标检测及目标未来运动监督 |

相机矩阵在这里由 info 内的传感器标定数据计算。`lidar2img` 把 LiDAR 坐标中的点投影到各相机图像；`camera2ego` 和 `camera_intrinsics` 则作为相机几何元信息传给后续模型。

`get_data_info()` 还把全局自车位置、姿态和朝向写入 `can_bus`，并计算 `lidar2global`。这些位姿随后用于时序 BEV 对齐和向量地图局部坐标变换。

## 2. `prepare_train_data()`：当前帧与历史帧

配置中的 `queue_length=3` 是最终 queue 的目标长度。代码先从当前 index 之前取三个候选 index：

```python
prev_indexs_list = list(range(index - self.queue_length, index))
random.shuffle(prev_indexs_list)
prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
```

这里有一个容易漏看的细节：候选帧有 3 个，随机打乱后切掉一个，再排序留下 2 个历史帧。当前帧随后加入，因此正常情况下总计是 3 帧：

```text
随机从 {t-3, t-2, t-1} 中选两个 + 当前帧 t
```

被选中的帧按从新到旧的顺序遍历，并通过 `insert(0, ...)` 放入 queue，所以最终排列是时间从旧到新：

```text
[较早历史帧, 较近历史帧, 当前帧]
```

每帧各自执行：

```text
get_data_info → pre_pipeline → pipeline → vectormap_pipeline
```

这里的“前一个已接受帧”指循环中最近一次通过检查的帧。代码用变量 `frame_idx` 记住它：开始时是当前帧的编号；每接受一个历史帧，就把它更新成该历史帧的编号。因此历史帧必须同时满足：

- `frame_idx` 小于当前记住的编号，也就是沿时间顺序继续往过去走；
- `scene_token` 与目标帧相同，不能把另一个 scene 的帧当作历史上下文。

要特别注意：候选帧检查失败时，代码只是跳过对这个候选帧的 `pipeline` 处理，并没有跳过这次入队。循环末尾仍会执行 `data_queue.insert(0, copy.deepcopy(example))`；此时 `example` 还保存着最近一次成功处理的帧，所以它会被再复制一份来占一个 queue 位置。

例如目标帧是 scene A 的 `t10`，候选依次为 `t9(A)`、`t8(B)`、`t7(B)`：`t9(A)` 被正常处理；`t8(B)` 和 `t7(B)` 因 scene 不同而不处理，但两次都把 `t9(A)` 的 `example` 深拷贝入队。最终时序槽位类似：

```text
[t9(A) 的副本, t9(A) 的副本, t9(A), t10(A)]
```

所以“补齐长度”不是生成了新的历史帧，而是让 queue 保持固定帧数；代价是边界附近可能有重复帧。这里要区分两种“相同”：`deepcopy` 会复制 Python 数据结构（例如字典及其中的数据），让每个 queue 槽位能独立保存；但这些副本记录的仍是同一个数据集样本——对应相同的 `frame_idx`、时间戳和原始图像/标注。也就是说，它们是“同一帧的多个副本”，不是“不同帧”。

## 3. `union2one()`：合并图像并改写时序元信息

Dataset 先取出每帧经 formatter 处理后的图像：

```python
imgs_list = [each['img'].data for each in queue]
```

每帧图像的维度为：

```text
[N_cam, C, H, W]
```

`torch.stack(imgs_list)` 沿新维度堆叠时间，得到单样本：

```text
[T, N_cam, C, H, W]
```

然后 `union2one()` 将每帧的 `img_metas` 放进按时间 index 索引的字典，并处理 `can_bus`：

- 第一帧设 `prev_bev=False`，位置和角度归零，作为时序参考帧。
- 后续帧设 `prev_bev=True`，`can_bus` 的位置和 yaw 改为相对前一帧的增量。

最后它把堆叠后的图像和时序元信息写回最后一个 queue 元素并返回。因为最后一个元素是当前帧，当前帧的检测/地图/规划标注仍是这个训练样本的监督标签；历史帧主要提供 BEV 时序上下文。

DataLoader 再增加 batch 维，模型入口收到的图像形状为：

```text
[B, T, N_cam, C, H, W]
```

## 4. 与模型训练入口衔接

`VAD.forward_train()` 根据第二维拆分时间：

```python
prev_img = img[:, :-1, ...]
img = img[:, -1, ...]
```

历史帧送入 `obtain_history_bev()`，逐帧计算 BEV 并在 `torch.no_grad()` 下运行；最后一帧单独提取特征并进入带监督的 `forward_pts_train()`。

```text
前 T-1 帧 → 历史 BEV（无梯度）
最后 1 帧 → 检测 / 地图 / motion / planning loss
```

这解释了为什么 Dataset 返回一个时序图像 queue，但检测和规划标注仍对应当前目标帧。

## 5. 和 UniAD 已学内容的映射

| UniAD 中熟悉的概念 | VAD 对应实现 |
|---|---|
| `get_data_info()` 构造样本字典 | `VADCustomNuScenesDataset.get_data_info()` |
| temporal queue | `prepare_train_data()` 的历史帧采样 |
| 合并多帧图像与 meta | `union2one()` |
| 用历史 BEV 辅助当前帧 | `obtain_history_bev()` |

VAD 的新点是历史帧采样带有随机丢帧，以及每一帧在普通 image/3D annotation pipeline 之后还会生成 vector map 标注。

## 6. 本节检查题

当 `queue_length=3` 时，`prev_indexs_list` 为什么最后只保留两个历史 index？`union2one()` 之后，哪一帧的 3D 检测标注会被用于当前训练 loss？
