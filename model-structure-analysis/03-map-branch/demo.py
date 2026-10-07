"""演示 map 分支的点 query 分组、polyline 外接框和坐标反归一化。"""


def group_points(flat_points, points_per_vector):
    """按 vector 优先、点索引其次的顺序恢复候选 polyline 分组。"""
    if len(flat_points) % points_per_vector != 0:
        raise ValueError("点总数必须是每条线点数的整数倍")
    return [
        flat_points[start:start + points_per_vector]
        for start in range(0, len(flat_points), points_per_vector)
    ]


def get_normalized_bbox(points):
    """从一条归一化 polyline 计算 [cx, cy, w, h] 外接框。"""
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    return [
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        x_max - x_min,
        y_max - y_min,
    ]


def denormalize_points(points, point_cloud_range):
    """把归一化点坐标映射回 LiDAR 局部 xy 范围。"""
    x_min, y_min, _, x_max, y_max, _ = point_cloud_range
    return [
        [
            x * (x_max - x_min) + x_min,
            y * (y_max - y_min) + y_min,
        ]
        for x, y in points
    ]


if __name__ == "__main__":
    # 两条合成 polyline，每条三个归一化点；正式配置每条线使用二十个点。
    flat_points = [
        [0.10, 0.25], [0.20, 0.30], [0.40, 0.35],
        [0.55, 0.60], [0.70, 0.65], [0.90, 0.80],
    ]
    point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    vectors = group_points(flat_points, points_per_vector=3)

    for vector_index, points in enumerate(vectors):
        bbox = get_normalized_bbox(points)
        local_points = denormalize_points(points, point_cloud_range)
        print(f"候选 vector {vector_index}：")
        print("  归一化点：", points)
        print("  归一化外接框 [cx, cy, w, h]：", [round(value, 3) for value in bbox])
        print("  LiDAR 局部点：", [[round(value, 2) for value in point] for point in local_points])
