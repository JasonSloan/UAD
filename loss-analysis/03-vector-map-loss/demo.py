"""演示开放地图线的点序候选选择及其方向监督。"""


def normalize_points(points, point_cloud_range):
    """按照配置中的点云范围归一化平面点坐标。"""
    min_x, min_y, _, max_x, max_y, _ = point_cloud_range
    width = max_x - min_x
    height = max_y - min_y
    return [[(x - min_x) / width, (y - min_y) / height] for x, y in points]


def ordered_l1_cost(pred_points, target_points, point_cloud_range):
    """计算两组有序采样点的逐坐标 L1 代价。"""
    pred_normalized = normalize_points(pred_points, point_cloud_range)
    target_normalized = normalize_points(target_points, point_cloud_range)
    return sum(
        abs(pred_xy[0] - target_xy[0]) + abs(pred_xy[1] - target_xy[1])
        for pred_xy, target_xy in zip(pred_normalized, target_normalized)
    )


def choose_point_order(pred_points, gt_point_orders, point_cloud_range):
    """在一个 query-GT pair 内选取点 L1 代价最小的 GT 点序。"""
    costs = [
        ordered_l1_cost(pred_points, candidate, point_cloud_range)
        for candidate in gt_point_orders
    ]
    best_order = min(range(len(costs)), key=costs.__getitem__)
    return best_order, costs


def segment_vectors(points):
    """将相邻采样点转换为有方向的线段向量。"""
    return [
        [next_point[0] - point[0], next_point[1] - point[1]]
        for point, next_point in zip(points[:-1], points[1:])
    ]


def cosine_direction_loss(pred_vectors, target_vectors):
    """计算方向余弦损失示例，忽略线段长度差异。"""
    losses = []
    for pred, target in zip(pred_vectors, target_vectors):
        pred_norm = (pred[0] ** 2 + pred[1] ** 2) ** 0.5
        target_norm = (target[0] ** 2 + target[1] ** 2) ** 0.5
        if pred_norm == 0 or target_norm == 0:
            losses.append(0.0)
            continue
        cosine = (pred[0] * target[0] + pred[1] * target[1]) / (pred_norm * target_norm)
        losses.append(1.0 - cosine)
    return losses


if __name__ == "__main__":
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

    # 示例缩短为 4 个点；真实 tiny 配置每条地图向量采样 20 个点。
    gt_forward = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    gt_reverse = list(reversed(gt_forward))
    prediction = [[3.0, 0.05], [2.0, 0.03], [1.0, -0.02], [0.0, 0.0]]

    best_order, costs = choose_point_order(
        prediction, [gt_forward, gt_reverse], point_cloud_range
    )
    selected_target = [gt_forward, gt_reverse][best_order]

    print("正向 GT 的归一化有序点 L1 代价：", round(costs[0], 6))
    print("反向 GT 的归一化有序点 L1 代价：", round(costs[1], 6))
    print("匹配到该 GT 后选择的点序编号：", best_order)
    print("用于正 query 点回归的 GT 点序：", selected_target)

    pred_directions = segment_vectors(prediction)
    target_directions = segment_vectors(selected_target)
    direction_losses = cosine_direction_loss(pred_directions, target_directions)
    print("预测相邻线段向量：", pred_directions)
    print("GT 相邻线段向量：", target_directions)
    print("逐段方向 cosine loss 示例：", [round(value, 6) for value in direction_losses])
