# 02：运动预测损失——best-of-6 如何选中一条轨迹

本节沿着上一节的检测匹配结果继续追踪：匹配到 GT 的 object query 如何获得未来轨迹监督，6 个候选模式中哪一个负责回归，以及模式分类分支怎样学习“哪条候选最像 GT”。

主要源码：

- 配置：`projects/configs/VAD/VAD_tiny_e2e.py` 中 `loss_traj`、`loss_traj_cls`、`valid_fut_ts`
- 多任务 head：`projects/mmdet3d_plugin/VAD/VAD_head.py` 中 `_get_target_single()`、`loss_single()`、`get_best_fut_preds()`、`get_traj_cls_target()`
- future GT 生成：`tools/data_converter/vad_nuscenes_converter.py` 中 `gt_fut_trajs` 和 `gt_fut_masks`
- 与 UniAD 对照：`/mnt/nfs_docker_volume/training-container-space/code/end2end/UniAD/projects/mmdet3d_plugin/losses/traj_loss.py`

## 1. 先接上检测匹配

上一节中，Hungarian matching 已经把正检测 query 对应到当前帧的 GT agent。`_get_target_single()` 用同一组 `pos_assigned_gt_inds` 取出这些 GT 的 future 位移和有效步 mask：

```text
正 query q 匹配到 GT g
       └─ future target ← GT[g] 的未来位移序列和有效步 mask
```

因此，VAD 只让匹配到 GT 的 query 回归该 GT 的轨迹。未匹配 query 没有某个 GT agent 的轨迹可回归，但仍会收到模式分类的背景监督。

## 2. 当前配置的预测与目标形状

tiny 配置中 `fut_mode=6`、`fut_ts=6`。一个检测 query 预测 6 种候选未来，每种未来含 6 个时间步、每步两个位移分量：

```text
轨迹预测：[query, mode=6, 时间步=6, (dx,dy)]
GT 目标：[query, 时间步=6, (dx,dy)]
有效步：[query, 时间步=6]
```

GT 和预测都是逐步位移。比较路径时，代码先沿时间维做 `cumsum`，把位移转换为相对起点的累计位置；回归 loss 则仍直接比较被选中模式的逐步位移与 GT 逐步位移。

## 3. best-of-6：按累计终点误差挑一个模式

`get_best_fut_preds()` 对预测和 GT 位移分别累计，再计算每个模式每个时间步的二维欧氏距离，并乘上 GT 的有效步 mask。随后代码只取最后一个时间步的距离，按最小值选择模式：

```text
累计预测位置 ─┐
              ├─ 最后时间步距离最小的模式 → best mode
累计 GT 位置 ─┘
```

所以此处选模依据是**有效终点的位移误差**，不是整条轨迹的平均误差 ADE，也不是 6 个模式分摊回归误差。`loss_traj` 只对 best mode 的逐步 `(dx,dy)` 计算带权 L1 loss；其余模式这一轮不接收轨迹回归目标。

这是一种 winner-take-all 监督：一个 GT agent 的一次前向，只把回归梯度主要给当前选中的候选模式。模式分类分支则学习把这个被选中的模式标成正类。

### mask 的实现细节

先区分两种容易混淆的“无效”：VAD 数据转换器里的 `gt_fut_masks` **不是 BEV 网格范围 mask**。它沿着 nuScenes annotation 的 `next` 链读取 future 框；找到下一条标注就写入位移并将 mask 置 1，链条结束后才把剩余 mask 留为 0。因此 future agent 即使移动到当前 BEV 感知范围之外，只要对应 future annotation 存在，通常仍是有效监督点，不会仅因超出 BEV 网格而被这里屏蔽。

逐步位移回归权重由三部分共同决定：

1. query 必须匹配到 GT；
2. GT future mask 标记该时间步有效；
3. `valid_fut_ts` 允许该时间步参与训练。当前 tiny 配置为 6。

无效步的回归权重为 0。源码的选模实现还有一个值得注意的细节：它先把每个时间步的距离乘 mask，之后直接读取序列最后一项。如果 annotation 链在预测 horizon 结束前就中断，导致 GT 最后一个时间步无效，那么最后距离会被置为 0，多个模式可能因此并列，`argmin` 会选第一个模式。GT future mask 若全为 0，也会出现同样的首模式退化；这些情况下回归权重为 0，但模式分类目标仍可能被设为模式 0。这里的无效指 future 标注缺失，不是“轨迹坐标超过 BEV 网格”。这是当前实现行为，不代表每个训练样本都会触发。

## 4. 模式分类：一个 sigmoid logit 对应一个模式

tiny 配置的 `loss_traj_cls` 是 sigmoid Focal Loss。因为使用 sigmoid，head 为每个模式输出一个 logit，而不是输出“6 个模式加一个显式背景类”的 softmax logits。

- 正 query：label 是按上述终点误差选出的 best mode；Focal Loss 将它编码为该模式为正、其余模式为负。
- 负 query：label 设为 `fut_mode`（这里是 6），在 sigmoid 编码下相当于 6 个模式全为负，不存在额外的第 7 个背景 logit。

可以把 6 个模式想成 6 个独立开关。正 query 若 best mode 是 2，target 是 `[0, 0, 1, 0, 0, 0]`；负 query 的 target 是 `[0, 0, 0, 0, 0, 0]`。例如负 query 的 sigmoid 分数若是 `[0.9, 0.1, 0.05, 0.02, 0.1, 0.03]`，第 0 个模式分数过高，就会受到明显的负类惩罚；其余低分模式也有负类项，但惩罚较小。这里做的不是“预测背景轨迹”，而是压低一个非目标 query 的所有模式置信度。

sigmoid Focal Loss 对单个负类 logit 的形式可直观写成：

```text
L_neg = (1 - alpha) * p^gamma * (-log(1 - p))
```

其中 `p` 是该模式的 sigmoid 分数。分数越高，负类惩罚越大；分数已很低时，`p^gamma` 会进一步压低容易负样本的损失。tiny 配置中 `alpha=0.25`、`gamma=2`。

还要把**逐 query 的 loss 权重**和**归一化因子 `avg_factor`**分开：生成 target 时 `label_weights` 对所有 query 都为 1，因此负 query 的六个负类项会进入 loss；但源码把 `traj_bg_cls_weight` 设为 0，所以负 query 不计入模式分类 loss 的 `avg_factor`。这不等于屏蔽负 query 的 loss，只是它们不增加归一化分母。框/轨迹回归则不同：负 query 的 `traj_weights` 为 0，不回归任何 GT 轨迹。

## 5. tiny 配置下实际的 motion loss

配置给出的权重：

| Loss 项 | 形式 | 配置权重 | 作用对象 |
|---|---|---:|---|
| `loss_traj` | L1 | 0.2 | 正 query 的 best mode，按 future mask 对逐步位移回归 |
| `loss_traj_cls` | sigmoid Focal | 0.2 | 正 query 分类 best mode；负 query 将所有模式作为负类 |

配置还设置了 `use_traj_lr_warmup=False`，因此当前实验不额外乘轨迹 loss warm-up 系数。虽然 head 会把同一组 motion 预测扩展到检测 decoder 各层并逐层构造 target，最终 `loss_dict` 只保留最后一层的 `loss_traj` 与 `loss_traj_cls`；检测中间层的分类和框 loss 则单独保留为辅助 loss。

## 6. 一个三步例子：终点最接近不一定 ADE 最小

假设 GT 累计位置沿 x 轴为 `1、2、3`：

- 模式 0 的累计位置是 `1、1、1`，终点误差为 2，三步平均误差为 1；
- 模式 1 的累计位置是 `0、0、2.8`，终点误差为 0.2，三步平均误差约为 1.07。

VAD 当前的选模代码会选模式 1，因为它的终点更接近；即使模式 0 的整条路径平均误差更小。配套 `demo.py` 用纯 Python 展示终点选模、被选模式的位移回归目标，以及负 query 的模式分类 target。

## 7. 与 UniAD 的增量对照

两者都做多模态轨迹预测，也都需要从多个候选中确定监督/评价所用的模式；但训练连接方式和选模细节不同：

| 对比点 | VAD | UniAD `TrajLoss` |
|---|---|---|
| 预测 query 的来源 | 当前帧检测 object query；由检测 Hungarian 匹配决定对应 GT agent | tracking 分支提供 track query；运动 loss 使用 track 与 GT 的匹配索引筛选有效目标 |
| VAD/UniAD 此处的正样本关系 | 匹配 query 才有轨迹回归；未匹配 query 仍做所有模式的 sigmoid 负类监督 | `compute_loss_traj()` 先筛出匹配到 GT 的 track，再送入 `TrajLoss` |
| best-mode 选择 | 对累计位置比较，按最后时刻误差选模式 | `TrajLoss` 计算 minFDE 和 minADE；用于最终回归轨迹及分类分数的索引来自 minADE |
| 回归形式 | best mode 的逐步位移 L1 | 默认不使用方差时，回归项为 best-mode 的 minADE；可配置分类、NLL、minADE、minFDE 等权重 |
| loss 层级 | motion 预测复制到检测层以便配 target，但只返回最后一层 motion loss | 各 motion decoder 层计算 loss，前置层也写入辅助 loss 项 |

因此，宏观上两边都有“匹配目标 + 多模态 best-of-K”，但不能把 VAD 的 `loss_traj` 直接等同于 UniAD 的 minADE/NLL 式轨迹损失：VAD 选模看终点，回归量是逐步位移，且未匹配检测 query 也参与模式负类训练。

## 8. 本节小结

- 检测 Hungarian 匹配把 GT future 轨迹交给正 object query；未匹配 query 没有轨迹回归目标。
- VAD 先累计位移成路径，再按终点误差选择一个 best mode；只有该模式计算逐步位移 L1。
- 模式分类用每个模式一个 sigmoid logit，正 query 监督 best mode，负 query 监督所有模式为负。
- future mask 控制回归有效步；当前实现若最后时刻无效，选模可能退化为首模式。
- 对比 UniAD 时，重点记住匹配 query 的来源、best-mode 指标、回归量和辅助层 loss 都不同。

下一节进入向量地图损失：地图 query 如何与 GT 向量做 Hungarian 匹配，闭合/非闭合线的点序 shift 如何参与点回归，以及方向 loss 监督什么。
