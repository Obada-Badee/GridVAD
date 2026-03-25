from typing import List, Dict, Any
import numpy as np

def merge_global_anomalies(anomalies: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    """
    Merges anomalies from overlapping clips that refer to the same event.
    """
    if not anomalies:
        return []

    # Sort by start time
    sorted_anomalies = sorted(anomalies, key=lambda x: x['start_t'])
    
    merged = []
    
    while sorted_anomalies:
        current = sorted_anomalies.pop(0)
        cluster = [current]
        
        # Find overlaps in remaining
        remaining = []
        for other in sorted_anomalies:
            # Check temporal overlap
            t1 = (current['start_t'], current['end_t'])
            t2 = (other['start_t'], other['end_t'])
            
            start = max(t1[0], t2[0])
            end = min(t1[1], t2[1])
            inter = max(0, end - start)
            union = (t1[1] - t1[0]) + (t2[1] - t2[0]) - inter
            iou = inter / union if union > 0 else 0
            
            # Semantic check
            # For now exact label match. In future use LLM or Embeddings.
            same_label = current.get('label') == other.get('label')
            
            if (iou > iou_thresh or inter > 1.0) and same_label:
                cluster.append(other)
            else:
                remaining.append(other)
        
        sorted_anomalies = remaining
        
        # Merge cluster
        best_candidate = max(cluster, key=lambda x: x.get('confidence', 0.0))
        merged_instance = best_candidate.copy()
        
        # Min/Max interval
        merged_instance['start_t'] = min(c['start_t'] for c in cluster)
        merged_instance['end_t'] = max(c['end_t'] for c in cluster)
        merged_instance['confidence'] = max(c.get('confidence', 0.0) for c in cluster)
        merged_instance['support_count'] = sum(c.get('support_count', 1) for c in cluster)
        merged_instance['merge_count'] = len(cluster)
        
        # Merge Tracks
        all_tracks = []
        for c in cluster:
            if 'tracks' in c:
                all_tracks.extend(c['tracks'])
        
        if all_tracks:
            # Group frames by frame_idx across all tracks in the cluster
            frame_map = {}
            for t in all_tracks:
                for f_data in t.get('frames', []):
                    f_idx = f_data['frame_idx']
                    if f_idx not in frame_map:
                        frame_map[f_idx] = f_data
                    # else: keep existing (could average bboxes but pick highest confidence clip instead)
            
            # Sort back by index
            sorted_frames = [frame_map[idx] for idx in sorted(frame_map.keys())]
            merged_instance['tracks'] = [{'track_id': 1, 'frames': sorted_frames}]
        
        merged.append(merged_instance)
        
    return merged
