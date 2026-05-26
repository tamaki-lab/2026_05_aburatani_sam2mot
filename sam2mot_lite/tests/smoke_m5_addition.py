#!/usr/bin/env python3
"""
Smoke test for Milestone 5: Dynamic mid-sequence object addition verification.

GPU + SAM2 checkpoint is required.

Usage:
  cd /path/to/repo_root
  PYTHONPATH=./sam2mot_lite .venv-sam2mot/bin/python \
    sam2mot_lite/tests/smoke_m5_addition.py \
    --checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt
"""
import argparse
import os
import shutil
import tempfile

import numpy as np
from PIL import Image


def create_dummy_data_with_addition(base_dir: str):
    """Creates dummy video frames and a detections.txt where object 2 appears on frame 3."""
    frames_dir = os.path.join(base_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(5):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Object 1: Red square (moving down-right, active all 5 frames)
        x1 = 100 + i * 20
        y1 = 100 + i * 10
        img[y1 : y1 + 50, x1 : x1 + 50] = [255, 0, 0]

        # Object 2: Green rectangle (moving down-left, only active from frame 3 onwards)
        if i >= 2:  # 0-indexed frame 2 = MOT 1-indexed frame 3
            x2 = 400 - i * 15
            y2 = 200 + i * 5
            img[y2 : y2 + 80, x2 : x2 + 40] = [0, 255, 0]

        Image.fromarray(img).save(os.path.join(frames_dir, f"{i:05d}.jpg"))

    det_file = os.path.join(base_dir, "detections.txt")
    with open(det_file, "w") as f:
        # Frame 1: only Object 1
        f.write("1,-1,95.0,95.0,60.0,60.0,0.9,1,1.0\n")
        # Frame 2: only Object 1
        f.write("2,-1,115.0,105.0,60.0,60.0,0.9,1,1.0\n")
        # Frame 3: Object 1 and Object 2 (appears here!)
        f.write("3,-1,135.0,115.0,60.0,60.0,0.9,1,1.0\n")
        f.write("3,-1,365.0,205.0,50.0,90.0,0.85,1,1.0\n")
        # Frame 4: Object 1 and Object 2
        f.write("4,-1,155.0,125.0,60.0,60.0,0.9,1,1.0\n")
        f.write("4,-1,350.0,210.0,50.0,90.0,0.85,1,1.0\n")
        # Frame 5: Object 1 and Object 2
        f.write("5,-1,175.0,135.0,60.0,60.0,0.9,1,1.0\n")
        f.write("5,-1,335.0,215.0,50.0,90.0,0.85,1,1.0\n")

    return frames_dir, det_file


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for M5: Dynamic Object Addition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 config name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sam2/checkpoints/sam2.1_hiera_tiny.pt",
        help="SAM2 checkpoint path",
    )
    args = parser.parse_args()

    # Pre-flight checks
    try:
        import sam2  # noqa: F401
    except ImportError:
        print("SAM2 is not installed or importable. Skipping smoke test.")
        return

    if not os.path.exists(args.checkpoint):
        print(f"SAM2 checkpoint not found: {args.checkpoint}")
        return

    # Temporary testing directory
    temp_dir = tempfile.mkdtemp()
    try:
        frames_dir, det_file = create_dummy_data_with_addition(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )

        from scripts.run_sequence import load_config, run_sequence

        config = load_config(config_path)

        # Run pipeline
        run_sequence(
            frames_dir=frames_dir,
            detections_file=det_file,
            config=config,
            sam2_config=args.config,
            checkpoint=args.checkpoint,
            output_dir=output_dir,
        )

        # ---- Verification ---- #
        traj_path = os.path.join(output_dir, "trajectories.txt")
        assert os.path.exists(traj_path), "trajectories.txt was not generated"

        with open(traj_path) as f:
            lines = f.readlines()

        track_entries = {}  # track_id -> list of frame_ids
        for line in lines:
            parts = line.strip().split(",")
            f_id = int(parts[0])
            t_id = int(parts[1])
            track_entries.setdefault(t_id, []).append(f_id)

        print("\n=== Validation Results ===")
        print(f"Total trajectory entries: {len(lines)}")
        for t_id, f_ids in sorted(track_entries.items()):
            print(f"Track {t_id}: frames {sorted(f_ids)}")

        # Verify that Track 1 starts from frame 1
        assert 1 in track_entries, "Track 1 should exist"
        assert min(track_entries[1]) == 1, "Track 1 should start at frame 1"
        assert len(track_entries[1]) == 5, "Track 1 should be tracked for 5 frames"

        # Verify that Track 2 starts from frame 3 (added dynamically mid-sequence)
        assert 2 in track_entries, "Track 2 should exist (dynamically added)"
        assert min(track_entries[2]) == 3, "Track 2 should start at frame 3 (mid-sequence addition)"
        assert len(track_entries[2]) == 3, "Track 2 should be tracked for 3 frames (3, 4, 5)"

        # Check formatting of first line
        sample = lines[0].strip().split(",")
        assert len(sample) == 10, f"Expected 10 columns, got {len(sample)}"
        assert sample[7] == "-1", "Column 8 should be -1"

        print("\nMilestone 5 Smoke Test PASSED ✅")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
