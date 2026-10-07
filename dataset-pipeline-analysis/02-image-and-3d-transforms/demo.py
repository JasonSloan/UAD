"""用最小数值示例演示 VAD 图像和 3D 标注变换。

本示例从完整的 MMDetection3D pipeline 中抽出相关几何关系：图像缩放必须
同步反映到 lidar2img；图像缩放本身不改变 LiDAR 坐标系下的 3D 框及其标签。
示例只依赖 NumPy，不需要 nuScenes 数据或模型权重。
"""

import numpy as np


def pad_to_divisor(size, divisor):
    """返回不小于 size 且能被 divisor 整除的最小尺寸。"""
    return ((size + divisor - 1) // divisor) * divisor


def project(matrix, point_xyz1):
    """使用列向量形式的矩阵投影一个齐次 LiDAR 点。"""
    image_homogeneous = matrix @ point_xyz1
    return image_homogeneous[:2] / image_homogeneous[2]


def main():
    # nuScenes 相机图像示例为 1600×900，VAD 将图像固定缩放到 0.4 倍。
    original_h, original_w = 900, 1600
    scale = 0.4
    resized_h, resized_w = int(original_h * scale), int(original_w * scale)
    padded_h = pad_to_divisor(resized_h, 32)
    padded_w = pad_to_divisor(resized_w, 32)

    print("图像尺寸（高, 宽）：")
    print(f"  原始尺寸：{original_h} × {original_w}")
    print(f"  缩放之后：{resized_h} × {resized_w}")
    print(f"  填充之后：{padded_h} × {padded_w}")

    # 构造合成针孔相机投影矩阵。按列向量约定，P 将 [x,y,z,1] 映射为 [u*z,v*z,z,1]。
    fx, fy = 1000.0, 1000.0
    cx, cy = original_w / 2, original_h / 2
    lidar2img = np.array(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image_scale = np.diag([scale, scale, 1.0, 1.0])
    scaled_lidar2img = image_scale @ lidar2img

    point = np.array([1.0, 0.5, 10.0, 1.0])
    uv_original = project(lidar2img, point)
    uv_after_resize = project(scaled_lidar2img, point)
    np.testing.assert_allclose(uv_after_resize, scale * uv_original)
    print("\n同一个 LiDAR 点在缩放前后的图像坐标：")
    print(f"  原图像素坐标：{uv_original}")
    print(f"  缩放后像素坐标：{uv_after_resize}")
    print("  校验：缩放后的像素坐标等于原坐标乘以 0.4")

    # 范围或类别筛选必须对 3D 框和对应的目标属性使用同一个掩码。
    boxes_xy = np.array([[1.0, 0.0], [20.0, 0.0], [-2.0, 3.0]])
    labels = np.array([0, 1, 8])
    future_attr = np.array([[10, 11], [20, 21], [30, 31]])
    keep = (np.abs(boxes_xy[:, 0]) <= 15) & (np.abs(boxes_xy[:, 1]) <= 30)
    filtered_boxes = boxes_xy[keep]
    filtered_labels = labels[keep]
    filtered_attr = future_attr[keep]
    np.testing.assert_equal(len(filtered_boxes), len(filtered_labels))
    np.testing.assert_equal(len(filtered_boxes), len(filtered_attr))
    print("\n同步筛选后的 GT：")
    print(f"  保留的类别标签：{filtered_labels.tolist()}")
    print(f"  对应的未来属性：{filtered_attr.tolist()}")
    print("\n所有演示校验均已通过。")


if __name__ == "__main__":
    main()
