import cv2
import numpy as np
import os
from typing import List, Dict, Any
from tqdm import tqdm

class Annotator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_annotated_video(self, video_path: str, anomalies: List[Dict[str, Any]], output_name: str = "annotated.mp4"):
        """
        Reads source video, overlays anomalies, and saves to output_path.
        video_path: Path to source video (or image pattern).
        anomalies: List of anomaly dicts from pipeline. 
                   Expected format:
                   {
                       'start_frame': ..., 'end_frame': ...,
                       'description': ...,
                       'tracks': [
                           {'frames': [{'frame_idx': ..., 'bbox': [x1, y1, x2, y2]}, ...]}
                       ]
                   }
        """
        # Initialize reader using our VideoReader for consistency
        from .video_io import VideoReader
        reader = VideoReader(video_path)
        
        # Prepare writer
        out_path = os.path.join(self.output_dir, output_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, reader.fps, (reader.width, reader.height))
        
        # Pre-process tracks for fast lookup: frame_idx -> list of (bbox, label, score)
        frame_map = {}
        
        for instance in anomalies:
            label = instance.get('description_for_grounding', 'Anomaly')
            tracks = instance.get('tracks', [])
            
            # 1. Overlay tracks ( SAM2 / Propagation )
            for track in tracks:
                for tf in track['frames']:
                    f_idx = tf['frame_idx']
                    bbox = tf['bbox']
                    score = tf.get('score', 1.0)
                    
                    if f_idx not in frame_map:
                        frame_map[f_idx] = []
                    frame_map[f_idx].append({
                        'bbox': bbox,
                        'label': label,
                        'score': score,
                        'type': 'track'
                    })
                    
            # 2. Overlay GroundingDINO box (if available and not in track?)
            # Usually detection is start of track, so it matches.
            # But we can show it distinctly if needed.
            if 'grounding_dino' in instance:
                 gd = instance['grounding_dino']
                 boxes = gd.get('boxes', [])
                 # We don't know exact frame for GD unless we stored it.
                 # Pipeline stores it in 'grounding' or we assume middle.
                 # Let's skip explicitly drawing GD if we have tracks covering it.
        
        print(f"Generating annotated video: {out_path}...")
        
        # Iterate all frames
        # Use iterator
        for frame_idx, frame, timestamp in tqdm(reader.iter_frames()):
            # Draw
            if frame_idx in frame_map:
                fname = os.path.basename(video_path)
                # Draw all objects
                for obj in frame_map[frame_idx]:
                    x1, y1, x2, y2 = map(int, obj['bbox'])
                    label_text = f"{obj['label'][:20]}"
                    
                    # Color: Red for anomaly
                    color = (0, 0, 255) 
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Label background
                    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add timestamp/frame counter
            cv2.putText(frame, f"F:{frame_idx} T:{timestamp:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            writer.write(frame)
            
        reader.close()
        writer.release()
        print("Done.")
