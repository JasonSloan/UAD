"""用 tiny 配置演示米制位移如何换算为归一化 BEV shift。"""

import numpy as np


def calculate_shift(delta_x, delta_y, absolute_yaw_rad):
    """按 VADPerceptionTransformer 中的公式计算 BEV shift。"""
    bev_h = 100
    bev_w = 100
    range_x = (-15.0, 15.0)
    range_y = (-30.0, 30.0)

    grid_length_y = (range_y[1] - range_y[0]) / bev_h
    grid_length_x = (range_x[1] - range_x[0]) / bev_w

    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle_deg = np.degrees(np.arctan2(delta_y, delta_x))
    ego_angle_deg = np.degrees(absolute_yaw_rad)
    bev_angle_rad = np.radians(ego_angle_deg - translation_angle_deg)

    shift_y = (
        translation_length * np.cos(bev_angle_rad)
        / grid_length_y / bev_h
    )
    shift_x = (
        translation_length * np.sin(bev_angle_rad)
        / grid_length_x / bev_w
    )

    return shift_x, shift_y, grid_length_x, grid_length_y


def main():
    # 假设相邻帧全局位置差为 (1.2, 0) 米，当前 ego 绝对朝向为 30 度。
    yaw_rad = np.deg2rad(30.0)
    shift_x, shift_y, cell_x, cell_y = calculate_shift(
        delta_x=1.2,
        delta_y=0.0,
        absolute_yaw_rad=yaw_rad,
    )

    # 把归一化偏移还原成 BEV 局部轴上的米制分量，检查尺度是否一致。
    local_x_m = shift_x * 100 * cell_x
    local_y_m = shift_y * 100 * cell_y

    print("tiny 配置的 BEV 网格尺寸:")
    print("x 方向每格: {:.3f} 米".format(cell_x))
    print("y 方向每格: {:.3f} 米".format(cell_y))
    print("\n归一化 shift [x, y]:")
    print("[{:.6f}, {:.6f}]".format(shift_x, shift_y))
    print("\n按网格尺度还原后的局部位移分量:")
    print("x: {:.3f} 米".format(local_x_m))
    print("y: {:.3f} 米".format(local_y_m))


if __name__ == "__main__":
    main()
