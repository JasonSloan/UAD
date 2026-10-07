"""演示 command 选轨迹、future mask 和规划约束的简化计算。"""


def select_commanded_mode(mode_predictions, command):
    """根据 one-hot command 选取对应模式的逐步位移。"""
    if not command or sum(command) != 1:
        raise ValueError("command 必须是 one-hot 向量")
    mode_index = command.index(1)
    return mode_index, mode_predictions[mode_index]


def accumulate_displacements(displacements):
    """将逐步位移累计为相对当前自车位置的轨迹点。"""
    positions = []
    x = 0.0
    y = 0.0
    for delta_x, delta_y in displacements:
        x += delta_x
        y += delta_y
        positions.append([x, y])
    return positions


def masked_l1(prediction, target, valid_mask):
    """计算只保留有效 future 步的逐坐标 L1 误差。"""
    errors = []
    for pred_xy, target_xy, is_valid in zip(prediction, target, valid_mask):
        errors.append([
            abs(pred_xy[0] - target_xy[0]) * is_valid,
            abs(pred_xy[1] - target_xy[1]) * is_valid,
        ])
    return errors


def margin_penalty(distance, threshold):
    """计算距离阈值内的简化安全间距惩罚。"""
    return max(0.0, threshold - distance)


def axis_clearance_penalty(delta_x, delta_y, x_threshold=1.5, y_threshold=3.0):
    """计算单个周车下的 x、y 方向间距惩罚示例。"""
    return [
        margin_penalty(abs(delta_x), x_threshold),
        margin_penalty(abs(delta_y), y_threshold),
    ]


if __name__ == "__main__":
    # 三种模式分别代表右转、左转、直行候选；这里只用四步简化演示。
    mode_predictions = [
        [[1.0, 0.2], [1.0, 0.4], [1.0, 0.6], [1.0, 0.8]],
        [[1.0, -0.2], [1.0, -0.4], [1.0, -0.6], [1.0, -0.8]],
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    ]
    gt_command = [0, 1, 0]
    gt_displacements = [[0.9, -0.1], [1.1, -0.3], [1.0, -0.5], [0.0, 0.0]]
    future_mask = [1, 1, 1, 0]

    selected_mode, selected_prediction = select_commanded_mode(mode_predictions, gt_command)
    print("GT command：", gt_command)
    print("被选中的模式编号：", selected_mode)
    print("用于规划监督的逐步位移：", selected_prediction)
    print("GT future mask：", future_mask)
    print("有效步上的逐坐标 L1 误差：",
          masked_l1(selected_prediction, gt_displacements, future_mask))
    print("累计后的预测轨迹点：", accumulate_displacements(selected_prediction))

    print("\n距离 boundary 0.4 m 时的 1.0 m margin 惩罚：",
          margin_penalty(0.4, 1.0))
    print("与一辆车的相对间距为 dx=1.0 m、dy=4.0 m 时的轴向惩罚：",
          axis_clearance_penalty(1.0, 4.0))
