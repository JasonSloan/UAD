"""演示 VAD 检测框的 10 维编码解码，以及 query-类别 top-k 还原。"""

import math


def decode_bbox(encoded_bbox):
    """把 VAD 的 10 维回归编码还原成 9 维 3D 框。"""
    (
        x,
        y,
        log_width,
        log_length,
        z,
        log_height,
        yaw_sin,
        yaw_cos,
        velocity_x,
        velocity_y,
    ) = encoded_bbox
    return [
        x,
        y,
        z,
        math.exp(log_width),
        math.exp(log_length),
        math.exp(log_height),
        math.atan2(yaw_sin, yaw_cos),
        velocity_x,
        velocity_y,
    ]


def sigmoid(value):
    """把分类头的单个 logit 转换为概率分数。"""
    return 1.0 / (1.0 + math.exp(-value))


def query_class_topk(class_logits, max_count):
    """模拟 coder 展平 query×类别分数后取 top-k 的索引恢复过程。"""
    candidates = []
    for query_index, logits_per_query in enumerate(class_logits):
        for class_index, logit in enumerate(logits_per_query):
            candidates.append((sigmoid(logit), query_index, class_index))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[:max_count]


if __name__ == "__main__":
    # 例子中的框中心已是物理坐标；三个尺寸使用对数编码。
    yaw_radians = math.radians(30)
    encoded_bbox = [
        4.0,
        -2.0,
        math.log(2.0),
        math.log(4.0),
        0.5,
        math.log(1.5),
        math.sin(yaw_radians),
        math.cos(yaw_radians),
        1.2,
        -0.3,
    ]
    print("解码后的框 [x, y, z, w, l, h, yaw, vx, vy]：")
    print([round(value, 4) for value in decode_bbox(encoded_bbox)])

    # 两个 query、三个类别。第一个 query 可因不同类别分数较高而重复入选。
    class_logits = [
        [3.0, 2.8, -1.0],
        [2.5, -0.5, 1.5],
    ]
    print("\nquery-类别 top-k 候选 [分数, query索引, 类别索引]：")
    for candidate in query_class_topk(class_logits, max_count=4):
        print([round(candidate[0], 4), candidate[1], candidate[2]])
