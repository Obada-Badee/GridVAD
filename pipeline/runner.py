import os
import json
import cv2
import base64
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm

from ..data.video_io import VideoReader, ClipGenerator
from ..core.grid import GridOverlay
from ..core.vlm import LocalVLMClient
from ..core.scc import SCC
from ..core.grounding import GroundingModule
from ..core.propagate import PropagationModule

class GridVADPipeline:
    def __init__(self, config: Dict[str, Any], vlm_client: LocalVLMClient, grounding: GroundingModule, propagator: PropagationModule):
        self.config = config
        self.vlm_client = vlm_client
        self.grounding = grounding
        self.propagator = propagator
        self.grid_overlay = GridOverlay(grid_size=config.get('grid_size', 4))
        self.scc = SCC(vlm_client=vlm_client)
        self.clip_gen = ClipGenerator(config.get('clip_len', 60.0), config.get('overlap', 2.0))
        from ..core.montage import MontageGenerator
        self.montage_gen = MontageGenerator(grid_size=config.get('grid_size', 4))

    def _encode_frame(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    def run_on_video(self, video_path: str) -> Dict[str, Any]:
        reader = VideoReader(video_path)
        clips = self.clip_gen.generate_clips(reader)
        
        all_anomalies = []

        print(f"Processing {len(clips)} clips for video {os.path.basename(video_path)}...")
        
        grid_dim = self.config.get('grid_size', 4)
        grid_dim = self.config.get('grid_size', 4)
        K = grid_dim * grid_dim 

        for clip in tqdm(clips):
            grid_dim = self.config.get('grid_size', 4)
            K = grid_dim * grid_dim
            M = self.config.get('M', 1)
            
            proposals = []
            
            # 1 & 2. Stratified Sampling & VLM Proposals (M times)
            for m_iter in range(M):
                # Stratified Sampling: Divide clip into K bins, pick one random frame per bin
                start_f = clip['start_frame']
                end_f = clip['end_frame']
                
                # Create K bins
                bins = np.linspace(start_f, end_f, K + 1)
                frame_indices = []
                for i in range(K):
                    # Pick a random frame in this bin
                    low = int(bins[i])
                    high = int(bins[i+1])
                    if high > low:
                        idx = np.random.randint(low, high)
                    else:
                        idx = low
                    frame_indices.append(idx)
                
                sampled_frames = []
                timestamps = []
                
                for idx in frame_indices:
                    frame = reader.get_frame(idx)
                    ts = idx / reader.fps
                    # Apply spatial grid
                    grid_frame, _ = self.grid_overlay.apply_grid(frame, timestamp=ts, frame_idx=idx)
                    sampled_frames.append(grid_frame)
                    timestamps.append(ts)
                
                # Create Temporal Montage (Grid of frames)
                montage = self.montage_gen.create_montage(sampled_frames)
                encoded_montage = self._encode_frame(montage)
                
                # VLM Inference
                resp = self.vlm_client.analyze_clip([encoded_montage], timestamps, grid_dim)
                if 'anomalies' in resp:
                    for p in resp['anomalies']:
                        # Filter by score
                        score = float(p.get('anomaly_score', p.get('confidence', 0.5)))
                        thresh = self.config.get('confidence_threshold', 0.5)
                        if score < thresh:
                            continue
                            
                        # Map relative evidence indices (1..K) to global frame indices
                        rel_indices = p.get('evidence_frame_indices', [])
                        global_ev = []
                        for r_idx in rel_indices:
                            if 1 <= r_idx <= len(frame_indices):
                                global_ev.append(frame_indices[r_idx - 1])
                        p['evidence_global_frames'] = global_ev
                        
                        proposals.append(p)
            
            # 3. SCC (Self-Consistency Consolidation)
            consolidated = self.scc.consolidate(proposals)
            
            # 4. Processing Targets
            # If we have confirmed anomalies, we run grounding/tracking on the CLIP interval
            if consolidated:
                # Extract clip frames for SAM3/Grounding
                # Clip indices
                start_f = clip['start_frame']
                end_f = clip['end_frame']
                
                # Read ALL frames in clip for SAM3 tracking
                # (Assuming 10s clip @ 25fps = 250 frames, reasonable for RAM)
                clip_rgb_frames = []
                for f_idx in range(start_f, end_f):
                    clip_rgb_frames.append(cv2.cvtColor(reader.get_frame(f_idx), cv2.COLOR_BGR2RGB))
                
                # Get cell coordinates from one frame (assume static grid)
                _, cell_coords = self.grid_overlay.apply_grid(clip_rgb_frames[0])

                for instance in consolidated:
                    if instance['support_count'] < self.config.get('min_support', 2):
                        continue
                        
                    description = instance['description_for_grounding']
                    
                    # 1. Calculate Crop Box
                    # User requested to remove spatial grid logic and run on full frame.
                    crop_box = None

                    # 2. R-tries (Anchor Selection)
                    # Pick R frames from the interval
                    R = self.config.get('R', 3)
                    
                    # Add padding to start/end time
                    pad = self.config.get('pad_sec', 1.0) # Using pad_sec arg for this purpose
                    start_t = max(0, instance['start_t'] - pad)
                    # We don't know total video length here easily, but we know clip end time roughly.
                    # Actually valid timestamps are within the clip roughly.
                    end_t = instance['end_t'] + pad
                    
                    # Target frames inside [start_t, end_t]
                    f_start = max(start_f, int(start_t * reader.fps))
                    f_end = min(end_f - 1, int(end_t * reader.fps))
                    
                    if f_end <= f_start:
                        anchor_indices = [f_start]
                    else:
                        anchor_indices = np.linspace(f_start, f_end, R, dtype=int).tolist()
                    
                    # Also include evidence frames if they are within this clip
                    evidence_global = instance.get('evidence_global_frames', [])
                    for g_idx in evidence_global:
                        if start_f <= g_idx < end_f:
                            if g_idx not in anchor_indices:
                                anchor_indices.append(g_idx)

                    best_box = None
                    best_score = -1.0
                    best_fid = -1
                    
                    # 3. Running Grounding on R-tries
                    box_thresh = self.config.get('box_threshold', 0.01)
                    text_thresh = self.config.get('text_threshold', 0.01)
                    
                    for fid in anchor_indices:
                        local_idx = fid - start_f
                        if 0 <= local_idx < len(clip_rgb_frames):
                            test_frame = clip_rgb_frames[local_idx]
                            # Use config thresholds
                            res = self.grounding.run_on_frame(test_frame, description, 
                                                            box_threshold=box_thresh, 
                                                            text_threshold=text_thresh,
                                                            crop_box=crop_box)
                            
                            boxes = res.get('boxes', [])
                            scores = res.get('scores', [])
                            
                            if boxes:
                                # Pick highest score in this frame
                                max_idx = np.argmax(scores)
                                if scores[max_idx] > best_score:
                                    best_score = scores[max_idx]
                                    best_box = boxes[max_idx]
                                    best_fid = fid
                    
                    # B. SAM2 Video (Propagation/Tracking)
                    if best_box:
                        # Closure for re-initialization during propagation
                        def reinit_cb(frame):
                            # Use config thresholds
                            res = self.grounding.run_on_frame(frame, description, 
                                                            box_threshold=box_thresh, 
                                                            text_threshold=text_thresh,
                                                            crop_box=crop_box)
                            boxes = res.get('boxes', [])
                            scores = res.get('scores', [])
                            if boxes and scores:
                                # Return box with highest score
                                max_idx = np.argmax(scores)
                                return boxes[max_idx]
                            return None

                        # ── VLM CPU offload ──────────────────────────────────
                        # VLM (~64 GB) and SAM2 (~9 GB) don't fit together on
                        # one 80 GB A100. Temporarily move VLM to CPU so SAM2
                        # has the headroom it needs.
                        vlm_model = getattr(self.vlm_client, 'model', None)
                        offloaded = False
                        try:
                            if vlm_model is not None:
                                vlm_model.to('cpu')
                                torch.cuda.empty_cache()
                                offloaded = True

                            tracks = self.propagator.propagate_with_box(
                                frames=clip_rgb_frames,
                                start_frame_idx=best_fid - start_f,
                                box=best_box,
                                max_frames=len(clip_rgb_frames),
                                area_thresh=self.config.get('sam_area_thresh', 50),
                                reinit_func=reinit_cb
                            )
                        finally:
                            # Always restore VLM to GPU (even if SAM2 crashed)
                            if offloaded and vlm_model is not None:
                                vlm_model.to(self.vlm_client.device)
                                torch.cuda.empty_cache()
                        # ────────────────────────────────────────────────────
                        
                        # Adjust frame indices to global
                        for track in tracks:
                            for fd in track['frames']:
                                fd['frame_idx'] = start_f + fd['frame_idx']
                        
                        instance['tracks'] = tracks
                        instance['grounding'] = {
                            "anchor_frame_idx": best_fid,
                            "anchor_box": best_box,
                            "score": best_score
                        }
                        all_anomalies.append(instance)

        reader.close()
        
        return {
            "video_id": os.path.basename(video_path),
            "anomalies": all_anomalies
        }
