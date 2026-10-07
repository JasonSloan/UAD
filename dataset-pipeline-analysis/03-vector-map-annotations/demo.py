"""演示 VAD 地图 GT 生成，并支持用真实 nuScenes 数据保存可视化图。"""

import argparse
from pathlib import Path
import sys

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, box


CLASS2LABEL = {
    "road_divider": 0,
    "lane_divider": 0,
    "ped_crossing": 1,
    "contours": 2,
}

LABEL2CLASS = {
    0: "divider",
    1: "ped_crossing",
    2: "boundary",
}

POINT_CLOUD_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
DEFAULT_DATAROOT = (
    "/mnt/nfs_docker_volume/training-container-space/mnt/datasets/"
    "object-detection-datasets/open-source/nuScenes/v1.0-trainval-related"
)
DEFAULT_SAMPLE_TOKEN = "e93e98b63d3b40209056d129dc53ceee"


def clip_and_localize(global_line, lidar_global_xy, lidar_global_yaw_deg, patch_size):
    """模拟 VAD 的地图 patch 裁剪、反向 yaw 旋转和原点平移。"""
    patch_h, patch_w = patch_size
    # lidar_global_xy：LiDAR 原点在 global 坐标系下的 x、y 平移。
    center_x, center_y = lidar_global_xy
    # local_patch_at_origin：以 LiDAR 原点为中心、轴与 LiDAR 对齐的局部矩形。
    local_patch_at_origin = box(-patch_w / 2, -patch_h / 2,
                                patch_w / 2, patch_h / 2)
    # global_patch：将局部矩形旋转到 global 朝向，再平移到 LiDAR 的 global 位置。
    global_patch = affinity.rotate(
        local_patch_at_origin, lidar_global_yaw_deg, origin=(0, 0), use_radians=False
    )
    global_patch = affinity.translate(
        global_patch, xoff=center_x, yoff=center_y
    )

    # global_line 和 clipped_global 都在 nuScenes 地图使用的 global 坐标系下。
    clipped_global = global_line.intersection(global_patch)
    # local_line：将裁剪结果反向旋转并平移，转换到以 LiDAR 为原点的局部坐标系。
    local_line = affinity.rotate(
        clipped_global, -lidar_global_yaw_deg, origin=(center_x, center_y),
        use_radians=False
    )
    return affinity.translate(local_line, xoff=-center_x, yoff=-center_y)


def sample_fixed_num(line, fixed_num):
    """按照 LiDARInstanceLines 的弧长插值思路采样固定点数。"""
    # line 的坐标系由调用方决定；本例传入的是 LiDAR 局部坐标系下的 local_line。
    distances = np.linspace(0, line.length, fixed_num)
    # points 沿用 line 的坐标系；本例中每个点都是 LiDAR 局部 xy 坐标。
    return np.asarray(
        [line.interpolate(distance).coords[0] for distance in distances],
        dtype=np.float64,
    )


def main():
    # 使用 VAD 配置的局部范围；patch_size 顺序为（高, 宽），所以
    # 在 LiDAR 局部坐标系下，x 范围是 [-15, 15]，y 范围是 [-30, 30]。
    # patch_size 本身是尺寸，不是某个坐标系中的坐标点。
    patch_size = (60.0, 30.0)
    # lidar_global_xy：LiDAR 原点在 global 坐标系下的位置，单位为米。
    lidar_global_xy = np.array([100.0, 200.0])
    # lidar_global_yaw_deg：LiDAR 局部坐标轴相对 global 坐标轴的 yaw 角。
    lidar_global_yaw_deg = 90.0

    # global_line：直接在 global 地图坐标系中构造道路分隔线，模拟 nuScenes 地图输入。
    # LiDAR 朝向为全局 yaw=90 度，因此这条全局纵向线对应局部 x 方向。
    global_line = LineString([(100.0, 180.0), (100.0, 220.0)])

    # local_line：global 裁剪结果变换到 LiDAR 局部坐标系后的折线。
    local_line = clip_and_localize(
        global_line, lidar_global_xy, lidar_global_yaw_deg, patch_size
    )
    assert local_line.geom_type == "LineString"

    # points：LiDAR 局部坐标系下采样出的 GT polyline，形状为 [20, 2]。
    points = sample_fixed_num(local_line, fixed_num=20)
    assert points.shape == (20, 2)
    np.testing.assert_allclose(points[0], [-15.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(points[-1], [15.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(points[:, 1], 0.0, atol=1e-6)

    print(
        "合成 LiDAR 全局位姿："
        f"全局平移 xy={lidar_global_xy.tolist()}，"
        f"yaw={lidar_global_yaw_deg:.1f} 度"
    )
    print("局部地图范围：x=[-15, 15]，y=[-30, 30]")
    print(f"全局地图输入折线：{list(global_line.coords)}")
    print(f"裁剪并转到 LiDAR 局部坐标后的折线长度：{local_line.length:.1f} 米")
    print(f"road_divider 对应的类别 ID：{CLASS2LABEL['road_divider']}")
    print(f"采样后的折线形状：{points.shape}（20 个点，每点含 x、y）")
    print(f"局部坐标下的首尾点：{points[0]} / {points[-1]}")
    print("所有地图几何校验均已通过。")


def get_lidar_global_pose(nusc, sample):
    """从真实 nuScenes 表中组合 LiDAR 到 global 的位姿。"""
    from pyquaternion import Quaternion

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    calibration = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
    ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

    ego_rotation = Quaternion(ego_pose["rotation"])
    lidar_rotation = ego_rotation * Quaternion(calibration["rotation"])
    lidar_translation = (
        np.asarray(ego_rotation.rotate(calibration["translation"]), dtype=np.float64)
        + np.asarray(ego_pose["translation"], dtype=np.float64)
    )
    return np.asarray(lidar_translation), lidar_rotation


def extract_real_map_sample(nusc, vector_map, sample):
    """使用 VAD 自己的地图转换器，生成一帧真实的局部地图 GT。"""
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    lidar_translation, lidar_rotation = get_lidar_global_pose(nusc, sample)
    annotations = vector_map.gen_vectorized_samples(
        log["location"], lidar_translation, lidar_rotation.elements
    )

    labels = np.asarray(annotations["gt_vecs_label"], dtype=np.int64)
    line_instances = annotations["gt_vecs_pts_loc"]
    if len(labels) == 0:
        points = np.empty((0, 20, 2), dtype=np.float32)
    else:
        points = line_instances.fixed_num_sampled_points.cpu().numpy()
    return log["location"], labels, points


def select_real_sample(nusc, vector_map, sample_token, search_limit):
    """选指定帧，或查找一帧同时包含三类地图要素的真实样本。"""
    if sample_token is not None:
        sample = nusc.get("sample", sample_token)
        location, labels, points = extract_real_map_sample(nusc, vector_map, sample)
        return sample, location, labels, points

    best_result = None
    required_labels = set(LABEL2CLASS)
    candidate_count = min(search_limit, len(nusc.sample))
    if candidate_count < 1:
        raise ValueError("--search-limit 必须大于 0，且 nuScenes 样本表不能为空。")
    candidate_indices = np.linspace(
        0, len(nusc.sample) - 1, candidate_count, dtype=np.int64
    )
    for checked_count, sample_index in enumerate(candidate_indices, start=1):
        sample = nusc.sample[int(sample_index)]
        location, labels, points = extract_real_map_sample(nusc, vector_map, sample)
        present_labels = set(labels.tolist())
        result = (sample, location, labels, points)
        if best_result is None or len(present_labels) > len(set(best_result[2].tolist())):
            best_result = result
        if required_labels.issubset(present_labels):
            return result
        if checked_count % 10 == 0:
            print(
                f"已检查 {checked_count}/{candidate_count} 个分布于数据集的样本，"
                "继续寻找同时含三类地图要素的帧……",
                flush=True,
            )

    if best_result is None:
        raise RuntimeError("nuScenes 样本表为空，无法选择可视化帧。")
    best_classes = sorted(set(best_result[2].tolist()))
    raise RuntimeError(
        "在抽查样本中没有找到同时包含 divider、ped_crossing、boundary 的帧；"
        f"覆盖最全的样本为 {best_result[0]['token']}，类别 ID 为 {best_classes}。"
        "请提高 --search-limit，或传入包含三类要素的 --sample-token。"
    )


def visualize_real_map(dataroot, version, output_path, sample_token, search_limit):
    """提取真实 VAD 地图 GT，将每条二十点向量绘图并保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from nuscenes.nuscenes import NuScenes

    # 脚本从子目录运行时，将仓库根目录加入模块搜索路径。
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from projects.mmdet3d_plugin.datasets.nuscenes_vad_dataset import (
        VectorizedLocalMap,
    )

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    patch_height = POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]
    patch_width = POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]
    vector_map = VectorizedLocalMap(
        dataroot=dataroot,
        patch_size=(patch_height, patch_width),
        map_classes=["divider", "ped_crossing", "boundary"],
        fixed_ptsnum_per_line=20,
    )
    sample, location, labels, points = select_real_sample(
        nusc, vector_map, sample_token, search_limit
    )

    colors = {
        0: "#e68613",
        1: "#2678b2",
        2: "#2a9d55",
    }
    fig, ax = plt.subplots(figsize=(8, 12), constrained_layout=True)
    counts = {label: int(np.count_nonzero(labels == label)) for label in LABEL2CLASS}

    for label, line_points in zip(labels, points):
        color = colors[int(label)]
        ax.plot(line_points[:, 0], line_points[:, 1], color=color, linewidth=1.5, alpha=0.85)
        ax.scatter(line_points[:, 0], line_points[:, 1], color=color, s=8, alpha=0.85)

    legend_handles = [
        Line2D([0], [0], color=colors[label], marker="o", markersize=4,
               linewidth=1.5, label=f"{LABEL2CLASS[label]} ({counts[label]})")
        for label in LABEL2CLASS
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_xlim(POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[3])
    ax.set_ylim(POINT_CLOUD_RANGE[1], POINT_CLOUD_RANGE[4])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("LiDAR-local x (m)")
    ax.set_ylabel("LiDAR-local y (m)")
    ax.set_title(
        "Real nuScenes map GT (20 points per vector)\n"
        f"location={location}, sample={sample['token'][:8]}..."
    )
    ax.grid(True, linestyle="--", alpha=0.25)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"真实 nuScenes 样本：{sample['token']}")
    print(f"地图位置：{location}")
    print("各类二十点向量数量：")
    for label, class_name in LABEL2CLASS.items():
        print(f"  {class_name}: {counts[label]}")
    print(f"可视化图片已保存：{output_path}")


def run_cli():
    """保留原合成示例，并提供真实数据可视化命令行入口。"""
    parser = argparse.ArgumentParser(description="VAD 地图 GT 示例")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="使用 nuScenes 地图和 VAD 数据转换器绘图",
    )
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument(
        "--output",
        default="dataset-pipeline-analysis/03-vector-map-annotations/real_map_gt.png",
        help="保存图片的路径",
    )
    parser.add_argument("--sample-token", default=DEFAULT_SAMPLE_TOKEN)
    parser.add_argument(
        "--find-sample",
        action="store_true",
        help="忽略默认 sample token，均匀抽查样本寻找同时含三类地图要素的帧",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=100,
        help="未指定样本时，在整个 split 中均匀抽查多少帧寻找三类要素",
    )
    args = parser.parse_args()

    if args.real_data:
        visualize_real_map(
            args.dataroot,
            args.version,
            args.output,
            None if args.find_sample else args.sample_token,
            args.search_limit,
        )
    else:
        main()


if __name__ == "__main__":
    run_cli()
