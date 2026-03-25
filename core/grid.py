import cv2
import numpy as np
from typing import Tuple, List, Dict

class GridOverlay:
    def __init__(self, grid_size: int = 4, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1):
        self.grid_size = grid_size
        self.color = color
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1

    def apply_grid(self, image: np.ndarray, timestamp: float = None, frame_idx: int = None) -> Tuple[np.ndarray, Dict[str, Tuple[int, int, int, int]]]:
        """
        Burn in timestamp on the image. 
        Note: Spatial grid drawing has been removed as per user request.
        Returns:
            - processed_image
            - empty dict (for compatibility)
        """
        img_h, img_w = image.shape[:2]
        output = image.copy()
        
        # Burn in timestamp if provided
        if timestamp is not None:
            ts_text = f"T: {timestamp:.2f}s"
            if frame_idx is not None:
                ts_text += f" | F: {frame_idx}"
            
            # Bottom-right corner
            ts_size, _ = cv2.getTextSize(ts_text, self.font, 1.0, 2)
            bg_rect_start = (img_w - ts_size[0] - 10, img_h - ts_size[1] - 10)
            bg_rect_end = (img_w, img_h)
            
            cv2.rectangle(output, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
            cv2.putText(output, ts_text, (img_w - ts_size[0] - 5, img_h - 5), self.font, 1.0, (0, 255, 255), 2)

        return output, {}
