#!/usr/bin/env python3
"""
Smoke test for Milestone 4: run_sequence パイプライン全体の動作確認.

GPU + SAM2 checkpoint が必要です.

使い方:
  cd /path/to/repo_root
  PYTHONPATH=./sam2mot_lite .venv-sam2mot/bin/python \\
    sam2mot_lite/tests/smoke_m4_sequence.py \\
    --checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt
"""
import argparse
import os
import shutil
import tempfile

import numpy as np
from PIL import Image


def create_dummy_data(base_dir: str):
    """ダミーの動画フレームと detections.txt を作成する."""
    frames_dir = os.path.join(base_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(5):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Object 1: 赤い正方形 (右下へ移動)
        x1 = 100 + i * 20
        y1 = 100 + i * 10
        img[y1 : y1 + 50, x1 : x1 + 50] = [255, 0, 0]
        # Object 2: 緑の長方形 (左下へ移動)
        x2 = 400 - i * 15
        y2 = 200 + i * 5
        img[y2 : y2 + 80, x2 : x2 + 40] = [0, 255, 0]

        Image.fromarray(img).save(os.path.join(frames_dir, f"{i:05d}.jpg"))

    det_file = os.path.join(base_dir, "detections.txt")
    with open(det_file, "w") as f:
        # Frame 1 (MOT 1-indexed) — 2つの検出
        f.write("1,-1,95.0,95.0,60.0,60.0,0.9,1,1.0\n")
        f.write("1,-1,395.0,195.0,50.0,90.0,0.85,1,1.0\n")

    return frames_dir, det_file


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for M4: run_sequence pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 config 名",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sam2/checkpoints/sam2.1_hiera_tiny.pt",
        help="SAM2 checkpoint のパス",
    )
    args = parser.parse_args()

    # 前提チェック
    try:
        import sam2  # noqa: F401
    except ImportError:
        print("SAM2 がインストールされていません. スキップします.")
        return

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint が見つかりません: {args.checkpoint}")
        return

    try:
        import yaml  # noqa: F401
    except ImportError:
        print("PyYAML がインストールされていません. スキップします.")
        return

    # テスト用一時ディレクトリ
    temp_dir = tempfile.mkdtemp()
    try:
        frames_dir, det_file = create_dummy_data(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )

        from scripts.run_sequence import load_config, run_sequence

        config = load_config(config_path)

        # パイプライン実行
        run_sequence(
            frames_dir=frames_dir,
            detections_file=det_file,
            config=config,
            sam2_config=args.config,
            checkpoint=args.checkpoint,
            output_dir=output_dir,
        )

        # ---- 結果検証 ---- #
        traj_path = os.path.join(output_dir, "trajectories.txt")
        assert os.path.exists(traj_path), "trajectories.txt が生成されていません"

        with open(traj_path) as f:
            lines = f.readlines()

        track_ids = set()
        frame_ids = set()
        for line in lines:
            parts = line.strip().split(",")
            frame_ids.add(int(parts[0]))
            track_ids.add(int(parts[1]))

        print(f"\n=== 検証結果 ===")
        print(f"trajectory エントリ数: {len(lines)}")
        print(f"Track ID: {sorted(track_ids)}")
        print(f"Frame ID: {sorted(frame_ids)}")

        assert len(track_ids) >= 2, (
            f"2つ以上の track が必要ですが {len(track_ids)} しかありません"
        )
        assert len(frame_ids) >= 2, (
            f"2フレーム以上の出力が必要ですが {len(frame_ids)} しかありません"
        )

        # 出力形式チェック: frame,track_id,x,y,w,h,score,-1,-1,-1
        sample = lines[0].strip().split(",")
        assert len(sample) == 10, f"列数が 10 でなく {len(sample)} です"
        assert sample[7] == "-1", f"8列目が -1 ではありません: {sample[7]}"

        # mask ファイルチェック
        masks_dir = os.path.join(output_dir, "masks")
        if os.path.exists(masks_dir):
            mask_files = os.listdir(masks_dir)
            print(f"mask ファイル数: {len(mask_files)}")
            assert len(mask_files) > 0, "mask ファイルが 0 件です"

        print("\nSmoke test PASSED ✅")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
