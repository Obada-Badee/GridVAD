import cv2
import numpy as np
from typing import List, Dict, Optional
import os

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def draw_annotations(self, image: np.ndarray, 
                         anomaly_instances: List[Dict], 
                         frame_idx: int, 
                         grid_overlay: Optional['GridOverlay'] = None) -> np.ndarray:
        """
        Draws boxes, masks, and text on a frame.
        """
        out_img = image.copy()
        
        # Draw anomalies active in this frame
        for instance in anomaly_instances:
            # Check if instance is active in this frame
            interval = instance.get('interval', {})
            # We assume the caller filters instances, but we can check if we have frame-specific data
            
            # Draw bbox if available for this frame
            # (In a real pipeline, we'd look up the track for this frame_idx)
            track = instance.get('tracks', [])
            for track_item in track:
                for frame_data in track_item.get('frames', []):
                    if frame_data['frame_idx'] == frame_idx:
                        bbox = frame_data.get('bbox')
                        if bbox:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            label = f"{instance['label']} ({instance['confidence']:.2f})"
                            cv2.putText(out_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return out_img

    def save_clip_debug(self, clip_id: str, frames: List[np.ndarray], fps: float = 25.0):
        """Saves a list of frames as a video file."""
        if not frames:
            return
            
        h, w = frames[0].shape[:2]
        out_path = os.path.join(self.output_dir, f"debug_clip_{clip_id}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
