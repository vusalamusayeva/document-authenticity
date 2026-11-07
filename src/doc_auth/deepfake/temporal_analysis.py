import numpy as np
from typing import List, Dict, Any

class TemporalAnalyzer:
    def __init__(self):
        pass

    def compute_temporal_score(self, frame_analyses: List[Dict[str, Any]]) -> float:

        if not frame_analyses or len(frame_analyses) < 2:
            return 0.0

        scores = [f.get("max_deepfake_score", 0.0) for f in frame_analyses]
        
        std_dev = np.std(scores)
        
        diffs = np.abs(np.diff(scores))
        max_jump = np.max(diffs) if len(diffs) > 0 else 0.0

        raw_score = float(std_dev * 2.0 + max_jump * 1.5)
        temporal_score = min(max(raw_score, 0.0), 1.0)

        return round(temporal_score, 4)