import cv2
import numpy as np
import os
from typing import List
from .schemas import Candidate

def visualize_results(image: np.ndarray, candidates: List[Candidate], output_path: str) -> str:

    output_img = image.copy()
    
    for cand in candidates:
        bbox = cand.bbox
        ymin, xmin, ymax, xmax = bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax
        
        if cand.is_suspicious:
            color = (0, 0, 255)  
            label_prefix = "Subheli"
        elif cand.type == "stamp":
            color = (0, 255, 0)  
            label_prefix = "Mohur"
        else:
            color = (255, 0, 0)  
            label_prefix = "Imza"
            
        cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
        
        label = f"{label_prefix}: {cand.confidence:.2f}"
        cv2.putText(output_img, label, (xmin, max(20, ymin - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, output_img)
    return output_path