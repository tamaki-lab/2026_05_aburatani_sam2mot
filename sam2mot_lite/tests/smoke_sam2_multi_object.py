import os
import numpy as np
import tempfile
from PIL import Image
from tracker.sam2_wrapper import SAM2Wrapper
from tracker.result_writer import write_trajectories

def create_dummy_video(frames_dir: str, num_frames: int = 5):
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(num_frames):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Obj 1: moving square
        x1 = 100 + i * 20
        y1 = 100 + i * 10
        img[y1:y1+50, x1:x1+50] = [255, 0, 0]
        
        # Obj 2: moving rectangle
        x2 = 300 - i * 15
        y2 = 200 + i * 5
        img[y2:y2+80, x2:x2+40] = [0, 255, 0]
        
        im = Image.fromarray(img)
        im.save(os.path.join(frames_dir, f"{i:05d}.jpg"))

import argparse
from tracker.sam2_wrapper import SAM2Wrapper, mot_to_sam2_frame, sam2_to_mot_frame
from tracker.result_writer import write_trajectories

def main():
    parser = argparse.ArgumentParser(description="Smoke test for SAM2 multi object.")
    parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml", help="SAM2 config name")
    parser.add_argument("--checkpoint", type=str, default="sam2/checkpoints/sam2.1_hiera_tiny.pt", help="Path to SAM2 checkpoint")
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

    mot_frame_id = 1
    frame_idx = mot_to_sam2_frame(mot_frame_id)
    # Obj 1
    obj1_id = 1
    box1 = [95.0, 95.0, 155.0, 155.0]
    wrapper.add_box_prompt(frame_idx, obj1_id, box1)
    
    # Obj 2
    obj2_id = 2
    box2 = [295.0, 195.0, 345.0, 285.0]
    mask2, out_box2, score2 = wrapper.add_box_prompt(frame_idx, obj2_id, box2)

    trajectories = []
    
    print("Propagating in video...")
    start_idx = mot_to_sam2_frame(mot_frame_id)
    for out_frame_idx, out_obj_ids, out_mask_logits in wrapper.propagate_in_video(start_frame_idx=start_idx):
        mot_frame_id_out = sam2_to_mot_frame(out_frame_idx)
        
        for tgt_id in out_obj_ids:
            mask, box, score = wrapper.extract_result(out_obj_ids, out_mask_logits, tgt_id)
            if box is not None:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                trajectories.append({
                    'frame_id': mot_frame_id_out,
                    'track_id': tgt_id,
                    'bbox_xywh': [x1, y1, w, h],
                    'score': score
                })
                
    out_file = os.path.join(temp_dir.name, "trajectories.txt")
    write_trajectories(out_file, trajectories)
    
    print(f"Wrote {len(trajectories)} trajectories to {out_file}")
    with open(out_file, 'r') as f:
        print(f.read())
        
    temp_dir.cleanup()

if __name__ == "__main__":
    main()
