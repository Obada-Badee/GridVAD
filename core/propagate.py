import numpy as np
from typing import List, Dict, Any
import torch
from PIL import Image

class PropagationModule:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large", device: str = "cuda", cache_dir: str = None):
        print(f"Loading SAM2 Video: {model_id}...")
        try:
            from transformers import Sam2VideoModel, Sam2VideoProcessor
            
            self.device = device
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
            
            self.model = Sam2VideoModel.from_pretrained(model_id, torch_dtype=self.dtype, cache_dir=cache_dir)
            self.model.to(self.device)
            
            self.processor = Sam2VideoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            
        except Exception as e:
            print(f"Failed to load SAM2: {e}")
            self.model = None

    def propagate_with_box(self, frames: List[np.ndarray], start_frame_idx: int, box: List[float], 
                           max_frames: int = 250, 
                           area_thresh: int = 100,
                           reinit_func: Any = None) -> List[Dict]:
        """
        Uses SAM2 to track objects starting from a box. Supports failure recovery.
        reinit_func: callable(frame_numpy) -> [x1, y1, x2, y2] or None
        """
        if self.model is None:
            return []

        pil_frames = [Image.fromarray(f) for f in frames]
        h, w = frames[0].shape[:2]

        try:
            inference_session = self.processor.init_video_session(
                video=pil_frames,
                inference_device=self.device,
                dtype=self.dtype
            )
            
            # Initial prompt
            self.processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=start_frame_idx,
                obj_ids=[1],
                input_boxes=[[box]] 
            )
            
            # First inference
            _ = self.model(inference_session=inference_session, frame_idx=start_frame_idx)
            
            obj_tracks = {1: {'id': 1, 'frames': []}}
            consecutive_small = 0
            MAX_SMALL = 5 # N frames to trigger re-init
            
            # Helper for processing propagation stream
            def process_stream(iterator):
                consecutive_small = 0
                for model_outputs in iterator:
                    f_idx = model_outputs.frame_idx
                    
                    masks_batch = self.processor.post_process_masks(
                        [model_outputs.pred_masks], 
                        original_sizes=[[h, w]], 
                        binarize=True
                    )[0]
                    
                    if len(masks_batch) > 0:
                         mask = masks_batch[0]
                         if mask.ndim == 3: mask = mask[0]
                         if hasattr(mask, 'cpu'): mask = mask.cpu().numpy()
                         
                         area = np.sum(mask)
                         if area < area_thresh:
                             consecutive_small += 1
                         else:
                             consecutive_small = 0
                         
                         # RECOVERY LOGIC
                         if consecutive_small >= MAX_SMALL and reinit_func is not None:
                             # print(f"Track lost at F:{f_idx}. Attempting re-init...")
                             new_box = reinit_func(frames[f_idx])
                             if new_box:
                                 # Add new prompt to SAME session
                                 self.processor.add_inputs_to_inference_session(
                                     inference_session=inference_session,
                                     frame_idx=f_idx,
                                     obj_ids=[1],
                                     input_boxes=[[new_box]] 
                                 )
                                 # Re-run inference on this frame to update state
                                 _ = self.model(inference_session=inference_session, frame_idx=f_idx)
                                 consecutive_small = 0
                         
                         rows, cols = np.where(mask > 0)
                         if len(rows) > 0:
                             p_box = [float(np.min(cols)), float(np.min(rows)), float(np.max(cols)), float(np.max(rows))]
                             # Avoid duplicate frames if overlapping
                             exists = any(t['frame_idx'] == f_idx for t in obj_tracks[1]['frames'])
                             if not exists:
                                 obj_tracks[1]['frames'].append({
                                     'frame_idx': f_idx,
                                     'bbox':      p_box,
                                     'mask':      mask.astype(np.uint8),  # full-res binary mask (H x W)
                                 })

            # 1. Forward Propagation
            process_stream(self.model.propagate_in_video_iterator(
                inference_session=inference_session,
                start_frame_idx=start_frame_idx,
                reverse=False
            ))

            # 2. Backward Propagation
            if start_frame_idx > 0:
                # Reset consecutive small for backward
                process_stream(self.model.propagate_in_video_iterator(
                    inference_session=inference_session,
                    start_frame_idx=start_frame_idx,
                    reverse=True
                ))
            
            # Sort frames by index
            obj_tracks[1]['frames'].sort(key=lambda x: x['frame_idx'])
            
            return list(obj_tracks.values())
            
        except Exception as e:
            print(f"SAM2 Propagation Error: {e}")
            import traceback
            traceback.print_exc()
            return []
