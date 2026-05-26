#!/usr/bin/env python3
"""
Smoke test for Milestone 7: Quality Reconstruction (Re-prompting pending tracks) verification.

GPU + SAM2 checkpoint is required.

Usage:
  cd /path/to/repo_root
  PYTHONPATH=./sam2mot_lite .venv-sam2mot/bin/python \
    sam2mot_lite/tests/smoke_m7_reconstruction.py \
    --checkpoint sam2/checkpoints/sam2.1_hiera_tiny.pt
"""
import argparse
import os
import shutil
import tempfile

import numpy as np
from PIL import Image


def create_dummy_data(base_dir: str):
    """Creates dummy video frames and a detections.txt."""
    frames_dir = os.path.join(base_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    for i in range(5):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Object 1: Red square (moving down-right)
        x1 = 100 + i * 20
        y1 = 100 + i * 10
        img[y1 : y1 + 50, x1 : x1 + 50] = [255, 0, 0]
        Image.fromarray(img).save(os.path.join(frames_dir, f"{i:05d}.jpg"))

    det_file = os.path.join(base_dir, "detections.txt")
    with open(det_file, "w") as f:
        # Frame 1 to 5: high confidence detections for Object 1
        f.write("1,-1,95.0,95.0,60.0,60.0,0.9,1,1.0\n")
        f.write("2,-1,115.0,105.0,60.0,60.0,0.9,1,1.0\n")
        f.write("3,-1,135.0,115.0,60.0,60.0,0.9,1,1.0\n")
        f.write("4,-1,155.0,125.0,60.0,60.0,0.9,1,1.0\n")
        f.write("5,-1,175.0,135.0,60.0,60.0,0.9,1,1.0\n")

    return frames_dir, det_file


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for M7: Quality Reconstruction"
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
        frames_dir, det_file = create_dummy_data(temp_dir)
        output_dir = os.path.join(temp_dir, "output")

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        )

        from scripts.run_sequence import load_config, run_sequence

        config = load_config(config_path)
        
        # Set reliable_thr artificially high (e.g. 20.0) so the track's score (usually ~15.0) 
        # naturally falls into STATE_PENDING on frame 2 (frame_idx = 1), triggering quality reconstruction!
        config["reliable_thr"] = 20.0
        config["enable_quality_reconstruction"] = True

        from tracker.trajectory_manager import TrajectoryManager
        # We will subclass TrajectoryManager to spy on the reconstruction events
        reconstruction_triggered = []
        original_associate = TrajectoryManager.associate_and_update
        
        def spy_associate_and_update(self, frame_id, frame_idx, detections, wrapper):
            res = original_associate(self, frame_id, frame_idx, detections, wrapper)
            # Check if any track was re-prompted at frame_idx > 0
            if frame_idx > 0:
                for t in self.get_active_tracks():
                    if t.keyframe_idx == frame_idx:
                        reconstruction_triggered.append((frame_id, t.track_id))
            return res
            
        TrajectoryManager.associate_and_update = spy_associate_and_update

        # Run pipeline
        run_sequence(
            frames_dir=frames_dir,
            detections_file=det_file,
            config=config,
            sam2_config=args.config,
            checkpoint=args.checkpoint,
            output_dir=output_dir,
        )

        # Restore original method
        TrajectoryManager.associate_and_update = original_associate

        # ---- Verification ---- #
        traj_path = os.path.join(output_dir, "trajectories.txt")
        assert os.path.exists(traj_path), "trajectories.txt was not generated"

        print("\n=== Validation Results ===")
        print(f"Reconstruction events detected: {reconstruction_triggered}")
        
        # Verify that quality reconstruction was indeed triggered at least once
        assert len(reconstruction_triggered) > 0, "Quality reconstruction was not triggered"
        print(f"Successfully re-prompted Track {reconstruction_triggered[0][1]} on Frame {reconstruction_triggered[0][0]}!")

        print("\nMilestone 7 Smoke Test PASSED ✅")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
