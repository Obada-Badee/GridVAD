from typing import List, Dict, Any, Tuple
import numpy as np

class SCC:
    def __init__(self, vlm_client=None):
        self.vlm_client = vlm_client

    def consolidate(self, proposals_list: List[Dict]) -> List[Dict]:
        """
        Consolidates proposals. 
        If VLM client is available, uses LLM-based merging.
        Otherwise falls back to Interval Union (min/max).
        """
        if not proposals_list:
            return []


        merged = self.vlm_client.consolidate_anomalies(proposals_list)
        return merged
        