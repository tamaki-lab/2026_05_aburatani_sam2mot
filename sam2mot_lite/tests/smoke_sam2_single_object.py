import os
import numpy as np
import tempfile
from PIL import Image
from tracker.sam2_wrapper import SAM2Wrapper

def create_dummy_video(frames_dir: str, num_frames: int = 5):
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(num_frames):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a moving square
        x = 100 + i * 20
        y = 100 + i * 10
        img[y:y+50, x:x+50] = [255, 0, 0]
        im = Image.fromarray(img)
        im.save(os.path.join(frames_dir, f"{i:05d}.jpg"))

import argparse
from tracker.sam2_wrapper import SAM2Wrapper, mot_to_sam2_frame

def main():
    parser = argparse.ArgumentParser(description="Smoke test for SAM2 single object.")
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml",
                        help="SAM2 config name (relative to sam2 package config search path)")
    parser.add_argument("--checkpoint", type=str, default="sam2/checkpoints/sam2.1_hiera_tiny.pt",
                        help="Path to SAM2 checkpoint file")
    args = parser.parse_args()

    try:
        import sam2
    except ImportError:
        print("SAM2 not installed. Skipping smoke test.")
        return

    config_path = args.config
    checkpoint_path = args.checkpoint
    
    if not os.path.exists(checkpoint_path):
        print(f"SAM2 checkpoint missing. Checked {checkpoint_path}")
        return

    temp_dir = tempfile.TemporaryDirectory()
    frames_dir = os.path.join(temp_dir.name, "frames")
    create_dummy_video(frames_dir)

    print("Initializing SAM2Wrapper...")
    wrapper = SAM2Wrapper(config_path, checkpoint_path)
    wrapper.init_video(frames_dir)

    # Frame 1 prompt (MOT format is 1-indexed)
    mot_frame_id = 1
    frame_idx = mot_to_sam2_frame(mot_frame_id)
    obj_id = 1
    # Square is at 100,100 -> 150,150
    box_prompt = [95.0, 95.0, 155.0, 155.0]
    
    print(f"Adding prompt at frame {frame_idx} for obj {obj_id}...")
    mask, box, score = wrapper.add_box_prompt(frame_idx, obj_id, box_prompt)
    
    if mask is not None:
        print(f"Success! Mask shape: {mask.shape}, BBox: {box}, Score: {score}")
        np.savez_compressed(os.path.join(temp_dir.name, "mask_0.npz"), mask=mask)
        print("Saved mask to mask_0.npz")
    else:
        print("Failed to generate mask.")
        
    temp_dir.cleanup()

if __name__ == "__main__":
    main()
