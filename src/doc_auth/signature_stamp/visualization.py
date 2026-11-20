import os
from typing import List

import cv2
import numpy as np

from .schemas import Candidate


def visualize_results(image: np.ndarray, candidates: List[Candidate], output_path: str) -> str:
    output_img = image.copy()
    h, w = output_img.shape[:2]

    for cand in candidates:
        raw_bbox = cand.bbox
        if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            x1 = max(0, min(w - 1, int(round(float(raw_bbox[0])))))
            y1 = max(0, min(h - 1, int(round(float(raw_bbox[1])))))
            x2 = max(0, min(w, int(round(float(raw_bbox[2])))))
            y2 = max(0, min(h, int(round(float(raw_bbox[3])))))
        elif hasattr(raw_bbox, "as_int_list"):
            bx1, by1, bx2, by2 = raw_bbox.as_int_list()
            x1, y1 = max(0, bx1), max(0, by1)
            x2, y2 = min(w, bx2), min(h, by2)
        else:
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        is_suspicious = bool(cand.metadata.get("is_suspicious", False))
        elem_type = cand.element_type or cand.label or "item"

        if is_suspicious:
            color = (0, 0, 255)
            label_prefix = "Subheli"
        elif elem_type == "stamp":
            color = (0, 255, 0)
            label_prefix = "Mohur"
        else:
            color = (255, 0, 0)
            label_prefix = "Imza"

        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)

        label = f"{label_prefix}: {cand.confidence:.2f}"
        cv2.putText(
            output_img,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    cv2.imwrite(output_path, output_img)
    return output_path