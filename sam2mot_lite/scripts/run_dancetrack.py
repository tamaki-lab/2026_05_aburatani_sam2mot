#!/usr/bin/env python3
"""
DanceTrack Batch Evaluator for SAM2MOT-lite.

Runs the SAM2MOT-lite pipeline on one or more DanceTrack sequences.
"""
import argparse
import os
import sys

# Ensure repository root is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_sequence import load_config, run_sequence


def main():
    parser = argparse.ArgumentParser(
        description="SAM2MOT-lite: DanceTrack Evaluation script"
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        default="data/DanceTrack/val",
        help="DanceTrack split directory containing sequences (e.g. data/DanceTrack/val)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default="",
        help="Specific sequence to run (e.g. dancetrack0004). If empty, runs testing_set or all sequences.",
    )
    parser.add_argument(
        "--testing_set",
        type=str,
        default="",
        help="Optional txt file listing sequences to run (one per line).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "configs", "default.yaml"
        ),
        help="SAM2MOT-lite configuration file path.",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 model configuration name.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sam2/checkpoints/sam2.1_hiera_tiny.pt",
        help="SAM2 checkpoint file path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sam2mot_lite",
        help="Base output directory for saving tracking results.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Max frames to track per sequence (useful for fast testing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu).",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.max_frames is not None:
        config["max_frames"] = args.max_frames

    # Determine sequences to run
    sequences = []
    if args.sequence:
        sequences = [args.sequence.strip()]
    elif args.testing_set:
        if not os.path.exists(args.testing_set):
            raise FileNotFoundError(f"Testing set file not found: {args.testing_set}")
        with open(args.testing_set, "r", encoding="utf-8") as f:
            sequences = sorted([line.strip() for line in f if line.strip()])
    else:
        # Scan all directories in video_folder
        if not os.path.exists(args.video_folder):
            raise FileNotFoundError(f"Video folder not found: {args.video_folder}")
        sequences = sorted([
            d for d in os.listdir(args.video_folder)
            if os.path.isdir(os.path.join(args.video_folder, d))
        ])

    print(f"[run_dancetrack] Starting evaluation on {len(sequences)} sequences.")

    for i, seq in enumerate(sequences, 1):
        print(f"\n==========================================")
        print(f"[{i}/{len(sequences)}] Processing sequence: {seq}")
        print(f"==========================================")

        frames_dir = os.path.join(args.video_folder, seq, "img1")
        detections_file = os.path.join(args.video_folder, seq, "gt", "gt.txt")
        seq_output_dir = os.path.join(args.output_dir, seq)

        if not os.path.exists(frames_dir):
            print(f"[WARNING] Frames directory not found: {frames_dir}. Skipping.")
            continue
        if not os.path.exists(detections_file):
            print(f"[WARNING] Detections file not found: {detections_file}. Skipping.")
            continue

        try:
            run_sequence(
                frames_dir=frames_dir,
                detections_file=detections_file,
                config=config,
                sam2_config=args.sam2_config,
                checkpoint=args.checkpoint,
                output_dir=seq_output_dir,
                device=args.device,
            )
            print(f"[SUCCESS] Sequence {seq} completed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process sequence {seq}: {e}")


if __name__ == "__main__":
    main()
