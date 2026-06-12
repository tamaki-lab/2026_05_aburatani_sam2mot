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
        self.inference_state = self.predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False
        )
        
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
        
        # Bypass tracking started check and frames_already_tracked if active
        was_tracking = self.inference_state.get("tracking_has_started", False)
        old_tracked = self.inference_state.get("frames_already_tracked", {})
        
        if was_tracking:
            self.inference_state["tracking_has_started"] = False
        # Treat this prompt frame as the initial conditioning frame for this object
        self.inference_state["frames_already_tracked"] = {}
            
        try:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box_np
            )
            
            # Pad historical frame outputs to match the new batch size if a new object was registered
            new_batch_size = len(self.inference_state["obj_ids"])
            self._pad_historical_states(new_batch_size)
        finally:
            if was_tracking:
                self.inference_state["tracking_has_started"] = True
            self.inference_state["frames_already_tracked"] = old_tracked
        
        return self._extract_mask_and_box(out_obj_ids, out_mask_logits, obj_id)

    def propagate_in_video(self, start_frame_idx: int):
        """
        Propagates masks for all objects from the current frame.
        """
        return self.predictor.propagate_in_video(self.inference_state, start_frame_idx=start_frame_idx)

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

    def _pad_historical_states(self, new_batch_size: int):
        """
        Pads all historical frame outputs in inference_state['output_dict'] and 
        inference_state['output_dict_per_obj'] to match the new batch size.
        This resolves shape mismatch errors when retrieving memory features 
        for frames prior to the introduction of a new object.
        """
        import torch
        from sam2.modeling.sam2_base import NO_OBJ_SCORE
        
        state = self.inference_state
        if state is None:
            return
            
        output_dict = state.get("output_dict", {})
        for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
            frames_dict = output_dict.get(storage_key, {})
            for frame_idx, out in list(frames_dict.items()):
                # Check the current shape of pred_masks
                if out.get("pred_masks") is None:
                    continue
                current_b = out["pred_masks"].shape[0]
                if current_b >= new_batch_size:
                    continue
                
                pad_size = new_batch_size - current_b
                
                # 1. Pad pred_masks
                H, W = out["pred_masks"].shape[-2:]
                dev_masks = out["pred_masks"].device
                dtype_masks = out["pred_masks"].dtype
                pad_masks = torch.full((pad_size, 1, H, W), NO_OBJ_SCORE, dtype=dtype_masks, device=dev_masks)
                out["pred_masks"] = torch.cat([out["pred_masks"], pad_masks], dim=0)
                
                # 2. Pad obj_ptr with zeros (since it is a feature embedding, not a logit/score)
                hidden_dim = out["obj_ptr"].shape[-1]
                dev_ptr = out["obj_ptr"].device
                dtype_ptr = out["obj_ptr"].dtype
                pad_ptr = torch.zeros((pad_size, hidden_dim), dtype=dtype_ptr, device=dev_ptr)
                out["obj_ptr"] = torch.cat([out["obj_ptr"], pad_ptr], dim=0)
                
                # 3. Pad object_score_logits
                dev_logits = out["object_score_logits"].device
                dtype_logits = out["object_score_logits"].dtype
                pad_logits = torch.full((pad_size, 1), NO_OBJ_SCORE, dtype=dtype_logits, device=dev_logits)
                out["object_score_logits"] = torch.cat([out["object_score_logits"], pad_logits], dim=0)
                
                # 4. Pad maskmem_features
                if out.get("maskmem_features") is not None:
                    C, H_m, W_m = out["maskmem_features"].shape[-3:]
                    dev_feat = out["maskmem_features"].device
                    dtype_feat = out["maskmem_features"].dtype
                    pad_feat = torch.zeros((pad_size, C, H_m, W_m), dtype=dtype_feat, device=dev_feat)
                    out["maskmem_features"] = torch.cat([out["maskmem_features"], pad_feat], dim=0)
                    
                # 5. Pad maskmem_pos_enc
                if out.get("maskmem_pos_enc") is not None:
                    padded_pos_enc = []
                    for pos in out["maskmem_pos_enc"]:
                        C_pos, H_pos, W_pos = pos.shape[-3:]
                        dev_pos = pos.device
                        dtype_pos = pos.dtype
                        pad_pos = torch.zeros((pad_size, C_pos, H_pos, W_pos), dtype=dtype_pos, device=dev_pos)
                        padded_pos_enc.append(torch.cat([pos, pad_pos], dim=0))
                    out["maskmem_pos_enc"] = padded_pos_enc
                
                # Update output_dict_per_obj slices for all objects (including the new ones)
                for obj_idx in range(new_batch_size):
                    obj_slice = slice(obj_idx, obj_idx + 1)
                    obj_out = {
                        "maskmem_features": None,
                        "maskmem_pos_enc": None,
                        "pred_masks": out["pred_masks"][obj_slice],
                        "obj_ptr": out["obj_ptr"][obj_slice],
                        "object_score_logits": out["object_score_logits"][obj_slice],
                    }
                    if out.get("maskmem_features") is not None:
                        obj_out["maskmem_features"] = out["maskmem_features"][obj_slice]
                    if out.get("maskmem_pos_enc") is not None:
                        obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in out["maskmem_pos_enc"]]
                        
                    if obj_idx not in state["output_dict_per_obj"]:
                        state["output_dict_per_obj"][obj_idx] = {
                            "cond_frame_outputs": {},
                            "non_cond_frame_outputs": {}
                        }
                    state["output_dict_per_obj"][obj_idx][storage_key][frame_idx] = obj_out
