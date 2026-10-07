"""演示 VAD 运动分支如何选 best mode 并生成模式分类 target。"""


def accumulate_displacements(displacements):
    """将逐步位移累计为相对起点的位置。"""
    positions = []
    x = 0.0
    y = 0.0
    for delta_x, delta_y in displacements:
        x += delta_x
        y += delta_y
        positions.append([x, y])
    return positions


def masked_step_distances(pred_positions, gt_positions, valid_mask):
    """计算逐步位置距离，并按 VAD 代码乘上有效步 mask。"""
    distances = []
    for pred, target, is_valid in zip(pred_positions, gt_positions, valid_mask):
        distance = ((pred[0] - target[0]) ** 2 + (pred[1] - target[1]) ** 2) ** 0.5
        distances.append(distance * is_valid)
    return distances


def choose_best_mode(mode_displacements, gt_displacements, valid_mask):
    """按照累计轨迹最后一步的距离选出模式。"""
    gt_positions = accumulate_displacements(gt_displacements)
    mode_positions = [accumulate_displacements(mode) for mode in mode_displacements]
    distances = [
        masked_step_distances(positions, gt_positions, valid_mask)
        for positions in mode_positions
    ]
    final_distances = [mode_distances[-1] for mode_distances in distances]
    best_mode = min(range(len(final_distances)), key=final_distances.__getitem__)
    return best_mode, gt_positions, mode_positions, distances, final_distances


def make_mode_class_target(best_mode, is_positive, num_modes):
    """生成 sigmoid 模式分类的逐模式二元 target。"""
    if not is_positive:
        return [0] * num_modes
    return [int(mode_index == best_mode) for mode_index in range(num_modes)]


def masked_best_mode_l1(best_displacements, gt_displacements, valid_mask):
    """计算示例中的逐步位移误差，并屏蔽无效时间步。"""
    errors = []
    for pred, target, is_valid in zip(best_displacements, gt_displacements, valid_mask):
        errors.append([abs(pred[0] - target[0]) * is_valid,
                       abs(pred[1] - target[1]) * is_valid])
    return errors


if __name__ == "__main__":
    gt_displacements = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    mode_displacements = [
        [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [2.8, 0.0]],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    ]
    valid_mask = [1, 1, 1]

    best_mode, gt_positions, mode_positions, distances, final_distances = choose_best_mode(
        mode_displacements, gt_displacements, valid_mask
    )

    print("GT 累计位置：", gt_positions)
    for mode_index, positions in enumerate(mode_positions):
        print(f"模式 {mode_index} 累计位置：{positions}")
        print(f"模式 {mode_index} 逐步位置误差：{distances[mode_index]}")
        print(f"模式 {mode_index} 终点误差：{final_distances[mode_index]:.2f}")

    positive_target = make_mode_class_target(best_mode, True, len(mode_displacements))
    negative_target = make_mode_class_target(best_mode, False, len(mode_displacements))
    print("\n按终点误差选中的模式：", best_mode)
    print("正 query 的逐模式 sigmoid target：", positive_target)
    print("负 query 的逐模式 sigmoid target：", negative_target)
    print("best mode 的逐步位移 L1 绝对误差：",
          masked_best_mode_l1(mode_displacements[best_mode], gt_displacements, valid_mask))

    # 最后一个 GT 时间步无效时，源码的最后距离会被 mask 成零，所有模式并列后取首个。
    partial_mask = [1, 1, 0]
    _, _, _, partial_distances, partial_final_distances = choose_best_mode(
        mode_displacements, gt_displacements, partial_mask
    )
    print("\n最后一步无效时的逐步误差：", partial_distances)
    print("最后一步无效时用于选模的终点误差：", partial_final_distances)
