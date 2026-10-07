"""不依赖外部数据，演示 VAD 时序队列的关键行为。"""

from copy import deepcopy


def make_frame(token, scene, frame_idx):
    """模拟一帧经过数据处理后的样本。"""
    return {
        "sample_token": token,
        "scene_token": scene,
        "frame_idx": frame_idx,
        "image_marker": f"image-from-{token}",
    }


def build_queue(target, candidates):
    """用已排序的候选帧模拟 prepare_train_data 中的关键循环。"""
    queue = [deepcopy(target)]
    example = target
    last_accepted_frame_idx = target["frame_idx"]

    for candidate in candidates:
        if (
            candidate["frame_idx"] < last_accepted_frame_idx
            and candidate["scene_token"] == target["scene_token"]
        ):
            # 模拟依次执行 pre_pipeline、pipeline 和 vectormap_pipeline。
            example = candidate
            last_accepted_frame_idx = candidate["frame_idx"]

        # 即使候选帧没有通过检查，实际代码仍会执行这次入队。
        queue.insert(0, deepcopy(example))

    return queue


def main():
    target = make_frame("A_t10", "scene_A", 10)
    candidates = [
        make_frame("A_t9", "scene_A", 9),
        make_frame("B_t8", "scene_B", 8),
        make_frame("B_t7", "scene_B", 7),
    ]

    queue = build_queue(target, candidates)
    tokens = [frame["sample_token"] for frame in queue]
    expected = ["A_t9", "A_t9", "A_t9", "A_t10"]
    assert tokens == expected, (tokens, expected)

    print("候选帧按从新到旧的顺序检查：")
    for frame in candidates:
        print(
            f"  候选帧={frame['sample_token']}，"
            f"场景={frame['scene_token']}，帧号={frame['frame_idx']}"
        )
    print("\n最终队列按从旧到新的顺序排列：")
    for slot, frame in enumerate(queue):
        print(
            f"  槽位 {slot}：样本={frame['sample_token']}，"
            f"场景={frame['scene_token']}，帧号={frame['frame_idx']}"
        )
    print(f"\n4 个队列槽位实际来自 {len(set(tokens))} 个不同的原始帧。")
    print("队列行为校验通过。")


if __name__ == "__main__":
    main()
