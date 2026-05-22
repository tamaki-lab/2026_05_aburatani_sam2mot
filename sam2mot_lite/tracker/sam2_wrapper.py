"""
SAM2 wrapper for MOT.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None

def mot_to_sam2_frame(mot_frame_id: int) -> int:
    """Converts 1-indexed MOT frame ID to 0-indexed SAM2 frame index."""
    return mot_frame_id - 1

def sam2_to_mot_frame(sam2_frame_idx: int) -> int:
    """Converts 0-indexed SAM2 frame index to 1-indexed MOT frame ID."""
    return sam2_frame_idx + 1

class SAM2Wrapper:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        if build_sam2_video_predictor is None:
            raise ImportError("SAM2 is not installed or importable.")
            
        self.device = device
        self.predictor = build_sam2_video_predictor(
            config_path, checkpoint_path, device=device
        )
        self.inference_state = None
        self.frames_dir = None
        
    def init_video(self, frames_dir: str):
        """
        Initialize the predictor for a new video sequence.
        """
        self.frames_dir = frames_dir
        self.inference_state = self.predictor.init_state(video_path=frames_dir)
        
    def add_box_prompt(self, frame_idx: int, obj_id: int, box_xyxy: List[float]) -> Tuple[Optional[np.ndarray], Optional[List[float]], float]:
        """
        Adds a box prompt for a specific object ID at a given frame.
        frame_idx: 0-indexed frame index (SAM2 uses 0-based indexing)
        obj_id: 1-indexed object ID
        
        Returns:
            mask: (H, W) numpy array if object exists, else None
            bbox_xyxy: list of 4 floats if mask exists, else None
            score: float, confidence score of the prediction
        """
        if self.inference_state is None:
            raise ValueError("Call init_video first.")
            
        box_np = np.array(box_xyxy, dtype=np.float32)
        
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            box=box_np
        )
        
        return self._extract_mask_and_box(out_obj_ids, out_mask_logits, obj_id)

    def propagate_in_video(self, start_frame_idx: int):
        """
        Propagates masks for all objects from the current frame.
        Yields (frame_idx, out_obj_ids, out_mask_logits)
        """
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=start_frame_idx):
            yield out_frame_idx, out_obj_ids, out_mask_logits

    def extract_result(self, out_obj_ids: List[int], out_mask_logits: torch.Tensor, obj_id: int) -> Tuple[Optional[np.ndarray], Optional[List[float]], float]:
        return self._extract_mask_and_box(out_obj_ids, out_mask_logits, obj_id)

    def _extract_mask_and_box(self, out_obj_ids: List[int], out_mask_logits: torch.Tensor, target_obj_id: int) -> Tuple[Optional[np.ndarray], Optional[List[float]], float]:
        """Helper to extract mask, box, score for a target object ID."""
        from tracker.mask_utils import mask_to_box
        
        if target_obj_id not in out_obj_ids:
            return None, None, 0.0
            
        idx = list(out_obj_ids).index(target_obj_id)
        
        mask_logit = out_mask_logits[idx, 0].cpu().numpy()
        mask_binary = (mask_logit > 0.0).astype(np.uint8)
        
        if mask_binary.sum() == 0:
            return None, None, 0.0
            
        box_xyxy = mask_to_box(mask_binary)
        
        # Provisional score: mean of positive logits
        positive_logits = mask_logit[mask_logit > 0.0]
        score = float(positive_logits.mean()) if len(positive_logits) > 0 else 0.0
        
        return mask_binary, box_xyxy, score
