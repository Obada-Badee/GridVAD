import json
import time
import re
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalVLMClient:
    def __init__(self, model_id: str, device: str = "cuda", cache_dir: str = None):
        print(f"Loading local VLM: {model_id}...")
        self.device = device
        # Try specific Qwen3 class if available
        from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
        import torch
        
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype="bfloat16", 
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        print("Loaded using Qwen3VLMoeForConditionalGeneration")

    def _clean_and_parse(self, text: str) -> Dict[str, Any]:
        """Extracts JSON from markdown code blocks or raw text."""
        try:
            # Try to find JSON block
            import re
            match = re.search(r"```json\s+(.*?)\s+```", text, re.DOTALL)
            if match:
                text = match.group(1)
            else:
                # Try simple brace matching if no markdown
                # match first { to last }
                match = re.search(r"(\{.*\})", text, re.DOTALL)
                if match:
                    text = match.group(0)
                else:
                    # Maybe it returned a list directly?
                    match = re.search(r"(\[.*\])", text, re.DOTALL)
                    if match:
                        arr_text = match.group(0)
                        return {"anomalies": json.loads(arr_text), "unique_anomalies": []}

            data = json.loads(text)
            if isinstance(data, list):
                 return {"anomalies": data, "unique_anomalies": []}
            return data
        except Exception as e:
            logger.warning(f"Local VLM JSON parse failed: {e}. Text: {text[:100]}...")
            return {"anomalies": [], "unique_anomalies": []}

    def analyze_clip(self, encoded_images: List[str], timestamps: List[float], grid_size: int = 4) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info
        import torch
        import json
        
        start_t = timestamps[0] if timestamps else 0.0
        end_t = timestamps[-1] if timestamps else 0.0
        
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        # Add images
        for b64 in encoded_images:
            messages[0]["content"].append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{b64}"
            })
            
        system_prompt = f"""You are an advanced Video Anomaly Detection assistant. 
You are analyzing a video clip where a {grid_size}x{grid_size} grid of frames (Temporal Montage) is stitched into a single image.
Each frame in the grid has an index (1..K) and burned-in timestamps.
Your task is to identify anomalous events based on visual content.

**Definition of Anomaly**:
An anomaly is any action, or object that is unexpected, unusual, or out-of-place within the specific context of the scene (e.g., suspicious movements, objects in restricted areas, sudden appearance of an abnormal object).

**Output Format**:
You must return a strictly valid JSON object. Do not include markdown formatting or explanations outside the JSON.
The JSON structure must be:
{{
  "anomalies": [
    {{
      "label": "short_label_string",
      "description_for_grounding": "concise noun phrase describing the object (e.g. 'something is fallen')",
      "location_description": "brief spatial location in the frame (e.g. 'center', 'top-left', 'background')",
      "start_t": <float>,
      "end_t": <float>,
      "confidence": <float>,
      "evidence_frame_indices": [1, 2] 
    }}
  ]
}})

If no anomaly is found, return {{"anomalies": []}}.
"""
        messages[0]["content"].append({"type": "text", "text": system_prompt})
        
        # Inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False,           # greedy — reproducible
                                                                        temperature=None,          # must be None when do_sample=False
                                                                        top_k=None,
                                                                        top_p=None,
                                                                        remove_invalid_values=True )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return self._clean_and_parse(output_text)

    def consolidate_anomalies(self, proposals: List[Dict]) -> List[Dict]:
        """
        Uses the VLM (text-only mode) to merge duplicate or overlapping anomalies 
        based on their descriptions and time intervals.
        """
        if not proposals:
            return []
            
        # Serialize proposals for the prompt
        # We only need key info
        simplified = []
        for i, p in enumerate(proposals):
            simplified.append({
                "id": i,
                "label": p.get("label"),
                "description": p.get("description_for_grounding"),
                "location": p.get("location_description", ""),
                "start": p.get("start_t"),
                "end": p.get("end_t")
            })
            
        prompt = f"""You are an intelligent video analysis assistant.
Below is a list of anomaly proposals detected in a video clip. Many of them refer to the same event but are fragmented across time or have slightly different descriptions.
Your task is to consolidate them into a list of unique, distinct anomalous events.

Input Proposals:
{json.dumps(simplified, indent=2)}

Instructions:
1. Group proposals that refer to the EXACT SAME object instance, even if it moves or changes location slightly over time.
2. If an object (e.g. a car) moves from 'left' to 'center', treat it as ONE continuous event and merge them.
3. DO NOT merge different objects (e.g., a car and a person) into one entry. Keep distinct objects separate.
4. For each unique event group, create a SINGLE merged entry with the min start and max end time of that specific group.
5. Select the most descriptive description for that specific group.
6. Return a JSON object with the list of unique anomalies.

Output Format:
{{
  "unique_anomalies": [
    {{
      "label": "merged_label",
      "description": "best_description",
      "location": "location_description",
      "start": <float_min>,
      "end": <float_max>,
      "original_ids": [id1, id2...]
    }}
  ]
}}
"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant for video anomaly consolidation."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Prepare inputs for Qwen
            from transformers import AutoProcessor
            from qwen_vl_utils import process_vision_info
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False,           # greedy — reproducible
                                                                        temperature=None,          # must be None when do_sample=False
                                                                        top_k=None,
                                                                        top_p=None,
                                                                        remove_invalid_values=True )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            parsed = self._clean_and_parse(output_text)
            unique = parsed.get("unique_anomalies", [])
            
            final_list = []
            for u in unique:
                 orig_ids = u.get("original_ids", [])
                 confs = []
                 support = 0
                 evidence_frames = set()
                 
                 for oid in orig_ids:
                     if 0 <= oid < len(proposals):
                         p = proposals[oid]
                         confs.append(float(p.get("confidence", 0.5)))
                         support += p.get("support_count", 1)
                         
                         ev = p.get("evidence_global_frames", [])
                         for f in ev: evidence_frames.add(f)
                 
                 avg_conf = sum(confs)/len(confs) if confs else 0.5
                 
                 final_list.append({
                    "label": u.get("label", "anomaly"),
                    "description_for_grounding": u.get("description", ""),
                    "location_description": u.get("location", ""),
                    "start_t": float(u.get("start", 0)),
                    "end_t": float(u.get("end", 0)),
                    "confidence": avg_conf,
                    "support_count": support,
                    "cluster_size": len(orig_ids),
                    "evidence_global_frames": list(evidence_frames)
                 })
                 
            return final_list

        except Exception as e:
            print(f"Consolidation Error: {e}")
            return proposals
