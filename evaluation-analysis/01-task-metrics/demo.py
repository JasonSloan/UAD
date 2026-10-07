"""用小型假数据演示 VAD 常见运动与规划指标的计算口径。"""


def point_errors(predicted, target):
    """计算轨迹每个时刻的二维欧氏距离。"""
    return [
        ((point[0] - truth[0]) ** 2 + (point[1] - truth[1]) ** 2) ** 0.5
        for point, truth in zip(predicted, target)
    ]


def best_of_k_errors(predicted_modes, target):
    """分别计算所有模式中的最小 ADE 和最小 FDE。"""
    errors_by_mode = [point_errors(mode, target) for mode in predicted_modes]
    ade_by_mode = [sum(errors) / len(errors) for errors in errors_by_mode]
    fde_by_mode = [errors[-1] for errors in errors_by_mode]
    best_ade_mode = min(range(len(ade_by_mode)), key=ade_by_mode.__getitem__)
    best_fde_mode = min(range(len(fde_by_mode)), key=fde_by_mode.__getitem__)
    return (
        ade_by_mode[best_ade_mode],
        best_ade_mode,
        fde_by_mode[best_fde_mode],
        best_fde_mode,
    )


def prefix_average_l2(predicted, target, num_steps):
    """计算指定未来时域内逐点 L2 的平均值。"""
    errors = point_errors(predicted[:num_steps], target[:num_steps])
    return sum(errors) / len(errors)


def planning_collision_rates(center_collisions, box_collisions, num_steps):
    """计算规划时域内中心点碰撞率和 ego 外框碰撞率。"""
    center_rate = sum(center_collisions[:num_steps]) / num_steps
    box_rate = sum(box_collisions[:num_steps]) / num_steps
    return center_rate, box_rate


def epa(hit_count, false_positive_count, ground_truth_count, alpha=0.5):
    """按 VAD 代码中的计数形式计算 EPA。"""
    if ground_truth_count == 0:
        raise ValueError("EPA 的 GT 分母不能为 0")
    return (hit_count - alpha * false_positive_count) / ground_truth_count


if __name__ == "__main__":
    target_traj = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]]
    predicted_modes = [
        [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [8, 0]],
        [[2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [6, 0]],
    ]

    min_ade, ade_mode, min_fde, fde_mode = best_of_k_errors(
        predicted_modes, target_traj
    )
    print("minADE：", min_ade, "，对应 mode：", ade_mode)
    print("minFDE：", min_fde, "，对应 mode：", fde_mode)
    print("注意：minADE 与 minFDE 可以由不同 mode 得到。")

    planning_prediction = [[1.2, 0], [2.4, 0], [3.6, 0], [4.8, 0], [6, 0], [7.2, 0]]
    print("规划前 1 秒平均 L2：", prefix_average_l2(planning_prediction, target_traj, 2))
    print("规划前 3 秒平均 L2：", prefix_average_l2(planning_prediction, target_traj, 6))

    center_collisions = [False, True, False, False, True, False]
    box_collisions = [False, True, True, False, True, False]
    print("前 3 秒中心点/ego 外框碰撞率：",
          planning_collision_rates(center_collisions, box_collisions, 6))
    print("EPA 示例（hit=8, FP=2, GT=10）：", epa(8, 2, 10))
