#!/usr/bin/env python3
"""
SAM2MOT-lite: 1シーケンスに対する最小推論スクリプト (Milestone 4).

ワークフロー:
  1. detections.txt から高信頼検出を読み込む
  2. 最初のフレームの検出を SAM2 に box prompt として追加
  3. SAM2 で全フレームにマスクを伝播
  4. mask 由来 bbox を trajectories.txt に保存
  5. (オプション) mask npz を保存

使用例:
  cd /path/to/repo_root
  PYTHONPATH=./sam2mot_lite .venv-sam2mot/bin/python \\
    sam2mot_lite/scripts/run_sequence.py \\
    --frames_dir /path/to/frames \\
    --detections /path/to/detections.txt \\
    --output_dir /path/to/output
"""
import argparse
import os
import sys

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: str) -> dict:
    """YAML 設定ファイルを読み込む."""
    if yaml is None:
        raise ImportError(
            "PyYAML が必要です.  pip install pyyaml でインストールしてください."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_sequence(
    frames_dir: str,
    detections_file: str,
    config: dict,
    sam2_config: str,
    checkpoint: str,
    output_dir: str,
    device: str = "cuda",
) -> None:
    """1シーケンス分の最小 SAM2MOT-lite 推論を実行する."""
    from tracker.detection import read_detections
    from tracker.result_writer import write_trajectories
    from tracker.sam2_wrapper import (
        SAM2Wrapper,
        mot_to_sam2_frame,
        sam2_to_mot_frame,
    )
    from tracker.trajectory_manager import TrajectoryManager

    det_conf_thr = config.get("det_conf_thr", 0.5)
    save_masks = config.get("save_masks", True)
    max_frames = config.get("max_frames", None)

    # ------------------------------------------------------------------ #
    # 1. 検出読み込み
    # ------------------------------------------------------------------ #
    all_dets = read_detections(detections_file, score_thr=det_conf_thr)
    if not all_dets:
        print("[run_sequence] 閾値を超える検出がありません. 終了します.")
        return

    dets_by_frame: dict = {}
    for det in all_dets:
        dets_by_frame.setdefault(det.frame_id, []).append(det)

    first_frame = min(dets_by_frame.keys())
    first_dets = dets_by_frame[first_frame]

    print(
        f"[run_sequence] {len(all_dets)} 検出を読み込み "
        f"({len(dets_by_frame)} フレーム)"
    )
    print(
        f"[run_sequence] 最初のフレーム: {first_frame} "
        f"({len(first_dets)} 検出)"
    )

    # ------------------------------------------------------------------ #
    # 2. SAM2 初期化
    # ------------------------------------------------------------------ #
    print("[run_sequence] SAM2 を初期化中...")
    wrapper = SAM2Wrapper(sam2_config, checkpoint, device=device)
    wrapper.init_video(frames_dir)

    # ------------------------------------------------------------------ #
    # 3. 最初のフレームからトラック作成
    # ------------------------------------------------------------------ #
    tm = TrajectoryManager(config)
    created = tm.init_tracks_from_detections(first_dets, wrapper)
    print(
        f"[run_sequence] フレーム {first_frame} から "
        f"{len(created)} トラックを初期化"
    )

    if not created:
        print("[run_sequence] トラックを作成できませんでした. 終了します.")
        return

    # ------------------------------------------------------------------ #
    # 4. 伝播して結果を収集
    # ------------------------------------------------------------------ #
    trajectories: list = []
    masks_to_save: dict = {}  # frame_id -> {track_id: mask}

    start_idx = mot_to_sam2_frame(first_frame)
    current_idx = start_idx
    frame_count = 0
    total_frames = wrapper.inference_state["num_frames"]

    print("[run_sequence] マスクを伝播中...")
    while current_idx < total_frames:
        if max_frames is not None and frame_count >= max_frames:
            break

        added_new_object = False

        gen = wrapper.propagate_in_video(start_frame_idx=current_idx)
        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in gen:
                mot_frame_id = sam2_to_mot_frame(out_frame_idx)

                # 1. Update existing tracks with propagation results
                for track in tm.get_active_tracks():
                    mask, bbox, score = wrapper.extract_result(
                        out_obj_ids, out_mask_logits, track.track_id
                    )
                    tm.update_track(track, mot_frame_id, mask, bbox, score)

                tm.resolve_conflicts()

                # 2. Check for object addition and quality reconstruction
                dets = dets_by_frame.get(mot_frame_id, [])
                triggered_restart = tm.associate_and_update(
                    frame_id=mot_frame_id,
                    frame_idx=out_frame_idx,
                    detections=dets,
                    wrapper=wrapper
                )

                if triggered_restart:
                    added_new_object = True
                    current_idx = out_frame_idx  # Restart propagation from this frame!
                    gen.close()  # Close the generator immediately to free VRAM
                    break  # Break inner generator loop

                # 3. If no new objects were added, write trajectories for this frame
                if not added_new_object:
                    from tracker.track import STATE_RELIABLE, STATE_PENDING
                    for track in tm.get_active_tracks():
                        if track.bbox is not None and track.state in (STATE_RELIABLE, STATE_PENDING):
                            x1, y1, x2, y2 = track.bbox
                            trajectories.append(
                                {
                                    "frame_id": mot_frame_id,
                                    "track_id": track.track_id,
                                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                                    "score": track.score,
                                }
                            )
                            if save_masks and track.mask is not None:
                                masks_to_save.setdefault(mot_frame_id, {})[
                                    track.track_id
                                ] = track.mask

                    frame_count += 1

                    # 4. Prune old non-conditioning memory to prevent Out of Memory
                    prune_horizon = 48
                    if out_frame_idx > prune_horizon:
                        prune_idx = out_frame_idx - prune_horizon
                        # Pop from global non_cond_frame_outputs
                        wrapper.inference_state["output_dict"]["non_cond_frame_outputs"].pop(prune_idx, None)
                        # Pop from per-object non_cond_frame_outputs
                        for obj_out_dict in wrapper.inference_state["output_dict_per_obj"].values():
                            obj_out_dict["non_cond_frame_outputs"].pop(prune_idx, None)

                    if max_frames is not None and frame_count >= max_frames:
                        break
        finally:
            gen.close()

        if not added_new_object:
            # Reached end of video or propagation completed without restarts
            break

    # ------------------------------------------------------------------ #
    # 5. 結果書き出し
    # ------------------------------------------------------------------ #
    if output_dir.endswith(".txt"):
        traj_path = output_dir
        os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        traj_path = os.path.join(output_dir, "trajectories.txt")

    write_trajectories(traj_path, trajectories)
    print(
        f"[run_sequence] {len(trajectories)} エントリを "
        f"{traj_path} に書き出しました"
    )

    if save_masks and masks_to_save:
        if output_dir.endswith(".txt"):
            base_dir = os.path.dirname(output_dir)
            base_name = os.path.splitext(os.path.basename(output_dir))[0]
            masks_dir = os.path.join(base_dir, f"{base_name}_masks")
        else:
            masks_dir = os.path.join(output_dir, "masks")
            
        os.makedirs(masks_dir, exist_ok=True)
        for frame_id, track_masks in masks_to_save.items():
            save_dict = {
                f"track_{tid}": m for tid, m in track_masks.items()
            }
            np.savez_compressed(
                os.path.join(masks_dir, f"frame_{frame_id:06d}.npz"),
                **save_dict,
            )
        print(
            f"[run_sequence] {len(masks_to_save)} フレーム分の mask を "
            f"{masks_dir} に保存しました"
        )

    print("[run_sequence] 完了.")

    # Clean up GPU memory
    del wrapper
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser(
        description="SAM2MOT-lite: 1シーケンスの最小推論"
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="動画フレーム画像が入ったディレクトリ (例: 00000.jpg, 00001.jpg, ...)",
    )
    parser.add_argument(
        "--detections",
        type=str,
        required=True,
        help="detections.txt のパス (MOT 形式)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        ),
        help="SAM2MOT-lite 設定ファイルのパス (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 config 名 (Hydra config search path 内)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sam2/checkpoints/sam2.1_hiera_tiny.pt",
        help="SAM2 checkpoint ファイルのパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="デバイス (cuda または cpu)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    run_sequence(
        frames_dir=args.frames_dir,
        detections_file=args.detections,
        config=config,
        sam2_config=args.sam2_config,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
