import cv2
import numpy as np
from typing import List

class MontageGenerator:
    def __init__(self, grid_size: int = 4):
        """
        grid_size: number of frames per side (NxN). 
        Total frames K = grid_size * grid_size.
        """
        self.grid_size = grid_size

    def create_montage(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Stitches K frames into an NxN grid.
        """
        K = len(frames)
        N = self.grid_size
        
        if K == 0:
            return None
            
        # Target size for each sub-image (to keep montage size reasonable)
        # Assuming source is ~480p or 1080p, let's target ~256x256 or similar for each tile
        # Total image size ~1024 or 2048
        tile_h, tile_w = frames[0].shape[:2]
        # target_tile_size = (tile_w // 2, tile_h // 2) 
        # Actually, let's just use original or slightly downsampled
        target_tile_w = 400
        target_tile_h = int(tile_h * (target_tile_w / tile_w))
        
        resized_frames = [cv2.resize(f, (target_tile_w, target_tile_h)) for f in frames]
        
        # Create empty canvas
        montage_h = target_tile_h * N
        montage_w = target_tile_w * N
        montage = np.zeros((montage_h, montage_w, 3), dtype=np.uint8)
        
        for idx, frame in enumerate(resized_frames):
            if idx >= N * N:
                break
            r = idx // N
            c = idx % N
            
            y1 = r * target_tile_h
            y2 = y1 + target_tile_h
            x1 = c * target_tile_w
            x2 = x1 + target_tile_w
            
            montage[y1:y2, x1:x2] = frame
            
            # Label the tile index (1..K)
            cv2.putText(montage, f"idx:{idx+1}", (x1+5, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        return montage
