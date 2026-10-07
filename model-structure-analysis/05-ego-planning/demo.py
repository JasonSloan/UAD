"""演示 VAD 三条自车轨迹候选的选择和位移累计。"""


def accumulate_displacements(displacements):
    """把逐步位移累计成相对于当前自车位置的轨迹。"""
    positions = []
    current_x = 0.0
    current_y = 0.0
    for delta_x, delta_y in displacements:
        current_x += delta_x
        current_y += delta_y
        positions.append([current_x, current_y])
    return positions


def select_command_mode(trajectories, command):
    """按评测使用的 one-hot command 选择对应候选轨迹。"""
    if len(command) != 3 or sum(command) != 1:
        raise ValueError("command 必须是长度为三的 one-hot 列表")
    if len(trajectories) != 3:
        raise ValueError("示例需要三条候选轨迹")
    mode_index = command.index(1)
    return mode_index, trajectories[mode_index]


if __name__ == "__main__":
    # 三种模式依次对应右转、左转、直行；每个点是一步的局部 xy 位移。
    predicted_displacements = [
        [[0.2, 0.8], [0.4, 0.8], [0.6, 0.7]],
        [[-0.2, 0.8], [-0.4, 0.8], [-0.6, 0.7]],
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    ]
    command_names = ["右转", "左转", "直行"]
    command = [0, 0, 1]

    print("VAD forward 输出的三条位移候选：")
    for mode_index, displacements in enumerate(predicted_displacements):
        print(f"  模式 {mode_index}（{command_names[mode_index]}）：{displacements}")

    selected_index, selected_displacements = select_command_mode(
        predicted_displacements, command
    )
    positions = accumulate_displacements(selected_displacements)
    print("\n评测侧给定 command：", command_names[selected_index])
    print("选中模式索引：", selected_index)
    print("累计后的自车未来位置：", positions)
    print("注意：这里模拟的是按 command 选候选，不表示 VAD ego forward 接收了 command。")
