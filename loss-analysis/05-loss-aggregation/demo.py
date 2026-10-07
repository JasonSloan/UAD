"""用假数值演示 VAD loss 字典如何汇总成训练总 loss。"""


def build_example_losses():
    """构造 tiny 配置对应的 loss key 示例，数值仅用于演示。"""
    losses = {
        "loss_cls": 0.8,
        "loss_bbox": 0.2,
        "loss_traj": 0.3,
        "loss_traj_cls": 0.1,
        "loss_map_cls": 0.4,
        "loss_map_bbox": 0.0,
        "loss_map_iou": 0.0,
        "loss_map_pts": 0.6,
        "loss_map_dir": 0.02,
        "loss_plan_reg": 0.5,
        "loss_plan_bound": 0.1,
        "loss_plan_col": 0.15,
        "loss_plan_dir": 0.04,
    }

    # 两层检测辅助输出，每层监督分类和框回归。
    for layer_index in range(2):
        losses[f"d{layer_index}.loss_cls"] = 0.25
        losses[f"d{layer_index}.loss_bbox"] = 0.08

    # 两层地图辅助输出，每层包含五项地图损失。
    for layer_index in range(2):
        losses[f"d{layer_index}.loss_map_cls"] = 0.2
        losses[f"d{layer_index}.loss_map_bbox"] = 0.0
        losses[f"d{layer_index}.loss_map_iou"] = 0.0
        losses[f"d{layer_index}.loss_map_pts"] = 0.3
        losses[f"d{layer_index}.loss_map_dir"] = 0.01

    return losses


def sum_loss_entries(losses):
    """将名称中含 loss 的分项相加。"""
    return sum(value for name, value in losses.items() if "loss" in name)


if __name__ == "__main__":
    example_losses = build_example_losses()
    main_count = sum(1 for name in example_losses if not name.startswith("d"))
    auxiliary_count = len(example_losses) - main_count

    print("示例 loss key 总数：", len(example_losses))
    print("主 loss key 数：", main_count)
    print("辅助 loss key 数：", auxiliary_count)
    print("主 loss + 辅助 loss 汇总值：", sum_loss_entries(example_losses))
    print("注意：数值是假设的；模块权重已体现在分项值中，汇总时不再重复乘权重。")
