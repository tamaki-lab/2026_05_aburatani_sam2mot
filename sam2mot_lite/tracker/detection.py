"""
Detection utilities.
"""
from dataclasses import dataclass
from typing import Optional, List, Any
import os

@dataclass
class Detection:
    frame_id: int
    box_xyxy: List[float]
    score: float
    cls: Optional[int] = None
    raw: Optional[List[Any]] = None

    @classmethod
    def from_xywh(cls, frame_id: int, x: float, y: float, w: float, h: float, score: float, class_id: Optional[int] = None, raw: Optional[List[Any]] = None):
        box_xyxy = [x, y, x + w, y + h]
        return cls(frame_id=frame_id, box_xyxy=box_xyxy, score=score, cls=class_id, raw=raw)

    def to_xywh(self) -> List[float]:
        x1, y1, x2, y2 = self.box_xyxy
        return [x1, y1, x2 - x1, y2 - y1]

def read_detections(file_path: str, score_thr: float = 0.0) -> List[Detection]:
    """
    Reads detections from a standard MOT-format detections.txt file.
    Format: frame, id_or_minus1, x, y, w, h, score, class, visibility, ...
    """
    detections = []
    if not os.path.exists(file_path):
        return detections
        
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            try:
                frame_id = int(parts[0])
                
                # Check column length to determine format
                if len(parts) == 6:
                    # Format: frame, x, y, w, h, score
                    x, y, w, h = map(float, parts[1:5])
                    score = float(parts[5])
                    class_id = None
                else:
                    # Format: frame, id_or_minus1, x, y, w, h, score, class, visibility...
                    # Note: We do NOT use the id column as track_id here.
                    x, y, w, h = map(float, parts[2:6])
                    score = float(parts[6]) if len(parts) > 6 else 1.0
                    class_id = int(float(parts[7])) if len(parts) > 7 else None
                
                if score >= score_thr:
                    det = Detection.from_xywh(
                        frame_id=frame_id,
                        x=x, y=y, w=w, h=h,
                        score=score,
                        class_id=class_id,
                        raw=parts
                    )
                    detections.append(det)
            except ValueError:
                continue
                
    return detections

def filter_detections_by_score(detections: List[Detection], score_thr: float) -> List[Detection]:
    return [d for d in detections if d.score >= score_thr]
