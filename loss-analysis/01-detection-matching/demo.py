"""演示 Hungarian 匹配结果如何传给检测框和未来轨迹监督。"""


def build_targets(query_to_gt, gt_labels, gt_boxes, gt_future_trajs, num_classes):
    """根据已完成的 query-GT 分配，构造简化的分类、框和轨迹目标。"""
    if not (len(gt_labels) == len(gt_boxes) == len(gt_future_trajs)):
        raise ValueError("每个 GT 的类别、框和未来轨迹数量必须一致")

    targets = []
    for query_index, gt_index in enumerate(query_to_gt):
        if gt_index is None:
            targets.append({
                "query": query_index,
                "matched_gt": None,
                "class_target": [0] * num_classes,
                "bbox_target": None,
                "future_target": None,
                "bbox_weight": 0,
                "trajectory_weight": 0,
            })
            continue

        class_target = [0] * num_classes
        class_target[gt_labels[gt_index]] = 1
        targets.append({
            "query": query_index,
            "matched_gt": gt_index,
            "class_target": class_target,
            "bbox_target": gt_boxes[gt_index],
            "future_target": gt_future_trajs[gt_index],
            "bbox_weight": 1,
            "trajectory_weight": 1,
        })
    return targets


if __name__ == "__main__":
    # 直接指定已经完成的一对一匹配，只演示监督目标如何传递，不实现匹配算法。
    query_to_gt = [2, None, 0, 1, None]
    gt_labels = [0, 2, 1]
    gt_boxes = [
        [3.0, 1.0, 0.0, 2.0, 4.0, 1.5],
        [-2.0, 5.0, 0.0, 1.0, 2.0, 1.7],
        [0.5, -4.0, 0.0, 2.5, 5.0, 2.0],
    ]
    gt_future_trajs = [
        [[0.2, 0.0], [0.3, 0.1]],
        [[0.0, 0.5], [0.0, 0.5]],
        [[-0.2, 0.0], [-0.2, 0.0]],
    ]

    targets = build_targets(
        query_to_gt, gt_labels, gt_boxes, gt_future_trajs, num_classes=3
    )

    for target in targets:
        print(f"query {target['query']}：")
        print("  匹配 GT：", target["matched_gt"])
        print("  sigmoid 分类目标：", target["class_target"])
        print("  框回归权重：", target["bbox_weight"])
        print("  框目标：", target["bbox_target"])
        print("  轨迹回归权重：", target["trajectory_weight"])
        print("  未来位移目标：", target["future_target"])
