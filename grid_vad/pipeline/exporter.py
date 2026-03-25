import json
import os
import csv
import numpy as np
from typing import Dict, List, Any


class _ResultsEncoder(json.JSONEncoder):
    """
    Custom encoder for pipeline result dicts.
    - Drops 'mask' keys (large binary uint8 ndarrays — in-memory only, not persisted).
    - Converts numpy scalars / arrays to native Python types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

    def encode(self, obj):
        return super().encode(self._strip(obj))

    def _strip(self, obj):
        if isinstance(obj, dict):
            return {k: self._strip(v) for k, v in obj.items() if k != 'mask'}
        if isinstance(obj, list):
            return [self._strip(v) for v in obj]
        return obj


class Exporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_json(self, data: Dict[str, Any], filename: str = "results.json"):
        path = os.path.join(self.output_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, cls=_ResultsEncoder)
            
    def save_tao_txt(self, anomalies: List[Dict], video_id: str, fps: float = 25.0):
        """
        Saves anomalies in TAO/StreetScene format:
        frame_id, x_min, y_min, x_max, y_max, anomaly_score
        
        frame_id is 0-indexed or 1-indexed? TAO usually 0-indexed matches frame_idx? 
        The provided `compute_tbdc` script reads:
        detected_anomaly = [frame_id, x_min, y_min, x_max, y_max, anomaly_score]
        
        We need to unroll the tracks/intervals into this format.
        """
        lines = []
        
        for instance in anomalies:
            score = instance.get('confidence', 0.5)
            
            # 1. If we have explicit tracks (SAM2 output)
            if 'tracks' in instance:
                for track in instance['tracks']:
                    for frame_data in track.get('frames', []):
                        fid = frame_data['frame_idx']
                        bbox = frame_data.get('bbox') # [x1, y1, x2, y2]
                        if bbox:
                            lines.append(f"{fid},{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f},{score:.4f}")

            # 2. Fallback: If we only have start/end and an anchor box (Prototype mode)
            elif 'interval' in instance and 'grounding' in instance:
                # Interpolate or repeat anchor? 
                # Repeating anchor is better than nothing for static objects.
                # Use interval start/end
                start_t = instance['interval']['start_t']
                end_t = instance['interval']['end_t']
                start_f = int(start_t * fps)
                end_f = int(end_t * fps)
                
                # Anchor box
                anchor_box = instance['grounding']['box']
                
                for fid in range(start_f, end_f + 1):
                    # In a real implementation we might clamp to video duration, but here we just emit
                    lines.append(f"{fid},{anchor_box[0]:.1f},{anchor_box[1]:.1f},{anchor_box[2]:.1f},{anchor_box[3]:.1f},{score:.4f}")

        # Write to file
        # Filename: usually {video_id}.txt
        out_path = os.path.join(self.output_dir, f"{video_id}.txt")
        with open(out_path, 'w') as f:
            for line in lines:
                f.write(line + "\n")
