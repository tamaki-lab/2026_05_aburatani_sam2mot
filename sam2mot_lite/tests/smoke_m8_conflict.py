#!/usr/bin/env python3
"""
Smoke test for Milestone 8: Cross-object Interaction Approximation verification.

GPU + SAM2 checkpoint is required.

Usage:
  cd /path/to/repo_root
  PYTHONPATH=./sam2mot_lite .venv-sam2mot/bin/python \
    sam2mot_lite/tests/smoke_m8_conflict.py \
    --checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt
"""
import argparse
import os
import shutil
import tempfile

import numpy as np
from PIL import Image


def create_dummy_data_with_conflict(base_dir: str):
    """Creates dummy video frames where two objects collide/perfectly overlap on frame 3."""
    frames_dir = os.path.join(base_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(5):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        if i < 2:
            # Frame 0 and 1: Separate objects
            # Object 1: Red square
            x1 = 100 + i * 10
            y1 = 100
            img[y1 : y1 + 50, x1 : x1 + 50] = [255, 0, 0]

            # Object 2: Green square
            x2 = 200 - i * 10
            y2 = 100
            img[y2 : y2 + 50, x2 : x2 + 50] = [0, 255, 0]
        else:
            # Frame 2, 3, 4: Objects perfectly overlap (drawn as Yellow square)
            x_overlap = 150
            y_overlap = 100
            img[y_overlap : y_overlap + 50, x_overlap : x_overlap + 50] = [255, 255, 0]

        Image.fromarray(img).save(os.path.join(frames_dir, f"{i:05d}.jpg"))

    det_file = os.path.join(base_dir, "detections.txt")
    with open(det_file, "w") as f:
        # Frame 1: Separate detections for both objects
        f.write("1,-1,95.0,95.0,60.0,60.0,0.9,1,1.0\n")
        f.write("1,-1,195.0,95.0,60.0,60.0,0.9,1,1.0\n")
        # Frame 2: Separate detections
        f.write("2,-1,105.0,95.0,60.0,60.0,0.9,1,1.0\n")
        f.write("2,-1,185.0,95.0,60.0,60.0,0.9,1,1.0\n")
        # Frame 3: Overlapping detections
        f.write("3,-1,145.0,95.0,60.0,60.0,0.9,1,1.0\n")
        f.write("3,-1,145.0,95.0,60.0,60.0,0.85,1,1.0\n")

    return frames_dir, det_file


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for M8: Cross-Object Interaction Approximation"
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
        frames_dir, det_file = create_dummy_data_with_conflict(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )

        from scripts.run_sequence import load_config, run_sequence

        config = load_config(config_path)
        # Enable cross object interaction conflict detection and resolution
        config["enable_cross_object_interaction"] = True
        config["coi_miou_thr"] = 0.8
        config["coi_score_gap_thr"] = 2.0

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

        # Group track output by frame
        frame_tracks = {}  # frame_id -> list of track_ids
        for line in lines:
            parts = line.strip().split(",")
            f_id = int(parts[0])
            t_id = int(parts[1])
            frame_tracks.setdefault(f_id, []).append(t_id)

        print("\n=== Validation Results ===")
        print(f"Total trajectory entries: {len(lines)}")
        for f_id, t_ids in sorted(frame_tracks.items()):
            print(f"Frame {f_id}: Track IDs {sorted(t_ids)}")

        # Verify that for Frame 1 and 2, both objects (1 and 2) are active
        assert 1 in frame_tracks, "Frame 1 should have tracks"
        assert len(frame_tracks[1]) == 2, f"Expected 2 tracks on frame 1, got {frame_tracks[1]}"
        assert len(frame_tracks[2]) == 2, f"Expected 2 tracks on frame 2, got {frame_tracks[2]}"

        # Verify that on frame 3, due to overlap, conflict resolution was triggered, 
        # marking one track corrupted and degrading it to STATE_SUSPICIOUS, thus suppressing its output.
        # So we should have at most 1 track in the output of frame 3.
        assert 3 in frame_tracks, "Frame 3 should have at least one active track"
        assert len(frame_tracks[3]) <= 1, f"Conflict resolution failed: expected <= 1 track on frame 3, but got {frame_tracks[3]}"

        print("\nMilestone 8 Smoke Test PASSED ✅")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
