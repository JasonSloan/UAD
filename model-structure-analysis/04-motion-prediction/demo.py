"""演示 VAD motion query 的模式展开、输出形状和位移累加。"""


def expand_agent_modes(agent_features, mode_embeddings):
    """把每个 agent 特征与每个可学习模式向量相加。"""
    if not agent_features:
        raise ValueError("至少需要一个 agent 特征")
    if not mode_embeddings:
        raise ValueError("至少需要一个模式向量")

    expanded = []
    for agent_index, feature in enumerate(agent_features):
        for mode_index, mode in enumerate(mode_embeddings):
            if len(feature) != len(mode):
                raise ValueError("agent 特征与模式向量的维度必须相同")
            expanded.append({
                "agent": agent_index,
                "mode": mode_index,
                "query": [value + mode_value for value, mode_value in zip(feature, mode)],
            })
    return expanded


def accumulate_displacements(displacements):
    """把逐时刻位移累加成相对起点的未来位置。"""
    positions = []
    current_x = 0.0
    current_y = 0.0
    for delta_x, delta_y in displacements:
        current_x += delta_x
        current_y += delta_y
        positions.append([current_x, current_y])
    return positions


if __name__ == "__main__":
    # 为了便于阅读，示例用二维特征和两个模式；模型实际使用 256 维特征和六个模式。
    agent_features = [[10.0, 1.0], [20.0, 2.0]]
    mode_embeddings = [[1.0, 0.0], [-1.0, 0.5]]
    expanded = expand_agent_modes(agent_features, mode_embeddings)

    print("agent 与模式相加后的 query：")
    for item in expanded:
        print(
            f"  agent={item['agent']}，mode={item['mode']}，"
            f"query={item['query']}"
        )

    # tiny 默认规模：300 个检测 query、每个 6 个模式、每种模式预测 6 步二维位移。
    batch_size = 1
    num_agents = 300
    num_modes = 6
    future_steps = 6
    embed_dims = 256
    print("\n模型中的主要形状：")
    print(f"  展平 motion query：[A×M, B, D] = [{num_agents * num_modes}, {batch_size}, {embed_dims}]")
    print(f"  轨迹预测：[B, A, M, T, 2] = [{batch_size}, {num_agents}, {num_modes}, {future_steps}, 2]")
    print(f"  模式分数：[B, A, M] = [{batch_size}, {num_agents}, {num_modes}]")

    # 这里的每一行是一个时刻的位移，累加后才得到从当前 agent 位置出发的路径。
    displacements = [[1.0, 0.0], [1.0, 0.5], [0.0, 0.5]]
    print("\n逐步位移：", displacements)
    print("累计后的相对位置：", accumulate_displacements(displacements))
