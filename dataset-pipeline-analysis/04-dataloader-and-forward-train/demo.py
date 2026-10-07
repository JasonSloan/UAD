"""用合成样本展示 MMCV collate 对 VAD 数据容器的处理。"""

import torch
from mmcv.parallel import DataContainer
from mmcv.parallel import collate


def make_sample(sample_id, object_count):
    """构造一个结构接近 Dataset 返回值的合成样本。"""
    temporal_metas = {
        time_id: {
            "sample_id": sample_id,
            "time_id": time_id,
            "scene_token": "scene_demo",
        }
        for time_id in range(3)
    }

    return {
        # 每个样本的图像形状为 [时间, 相机, 通道, 高, 宽]。
        "img": DataContainer(
            torch.zeros(3, 6, 3, 8, 8), stack=True
        ),
        # 元信息保留为 Python 字典，只在 CPU 上传递。
        "img_metas": DataContainer(temporal_metas, cpu_only=True),
        # 数量可变的标签不堆叠成一个 Tensor。
        "gt_labels_3d": DataContainer(
            torch.arange(object_count), stack=False
        ),
        # 3D 框和地图线在这里用字符串占位，说明它们是 CPU-only 对象。
        "gt_bboxes_3d": DataContainer(
            ["样本 {} 的 3D 框对象".format(sample_id)], cpu_only=True
        ),
        "map_gt_labels_3d": DataContainer(
            torch.tensor([0, 2]), stack=False
        ),
        "map_gt_bboxes_3d": DataContainer(
            ["样本 {} 的地图折线对象".format(sample_id)], cpu_only=True
        ),
    }


def main():
    # 用两个样本组成一个 GPU batch，便于观察 batch 维；实际示例配置每卡为 1。
    samples = [make_sample(0, 2), make_sample(1, 4)]
    batch = collate(samples, samples_per_gpu=2)

    print("图像容器中的 batch Tensor 形状:")
    print(batch["img"].data[0].shape)
    print("说明: [batch, 时间, 相机, 通道, 高, 宽]")

    print("\n元信息的分组层级:")
    print("每卡分组数:", len(batch["img_metas"].data))
    print("该卡样本数:", len(batch["img_metas"].data[0]))
    print("样本 0 的时间槽位:", list(batch["img_metas"].data[0][0].keys()))

    print("\n变长检测标签保持逐样本存放:")
    labels_per_sample = batch["gt_labels_3d"].data[0]
    print([labels.tolist() for labels in labels_per_sample])

    print("\n地图折线对象保持 CPU-only:")
    print(batch["map_gt_bboxes_3d"].cpu_only)
    print(batch["map_gt_bboxes_3d"].data[0])


if __name__ == "__main__":
    main()
