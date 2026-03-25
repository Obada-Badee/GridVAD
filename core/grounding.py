import numpy as np
from typing import Dict, Any, List
import torch
from PIL import Image

class GroundingModule:
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str = "cuda", cache_dir: str = None):
        print(f"Loading GroundingDINO: {model_id}...")
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            self.device = device
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=cache_dir)
            self.model.to(self.device)
        except Exception as e:
            print(f"Failed to load GroundingDINO: {e}")
            self.model = None

    def run_on_frame(self, frame: np.ndarray, text_prompt: str, box_threshold: float = 0.25, text_threshold: float = 0.25, crop_box: List[int] = None) -> Dict[str, Any]:
        """
        frame: RGB numpy array
        text_prompt: Description of object to find.
        crop_box: [x1, y1, x2, y2] to focus search.
        """
        if self.model is None:
            return {}

        # Handle crop
        if crop_box:
            cx1, cy1, cx2, cy2 = map(int, crop_box)
            # Ensure within bounds
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(w, cx2), min(h, cy2)
            
            if cx2 > cx1 and cy2 > cy1:
                inference_frame = frame[cy1:cy2, cx1:cx2]
                offset_x, offset_y = cx1, cy1
            else:
                inference_frame = frame
                offset_x, offset_y = 0, 0
        else:
            inference_frame = frame
            offset_x, offset_y = 0, 0

        # Convert to PIL
        image_pil = Image.fromarray(inference_frame)
        
        # GroundingDINO prompts should typically end with a dot if not present
        if not text_prompt.endswith("."):
            text_prompt = text_prompt + "."
        text_prompt = text_prompt.lower()
        
        try:
            inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image_pil.size[::-1]]
            )
            
            if not results:
                return {}
                
            res = results[0]
            
            # Format output and map back offsets
            boxes = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy().tolist()
            labels = res["labels"] 
            
            mapped_boxes = []
            for box in boxes:
                # [x1, y1, x2, y2]
                mapped_boxes.append([
                    float(box[0] + offset_x),
                    float(box[1] + offset_y),
                    float(box[2] + offset_x),
                    float(box[3] + offset_y)
                ])
            
            return {
                "boxes": mapped_boxes,
                "scores": scores,
                "labels": labels
            }
        except Exception as e:
            print(f"Grounding Inference Error: {e}")
            return {}
