import numpy as np
import os
from typing import List, Dict, Any, Optional
from .detector import SignatureStampDetector
from .signature_preprocess import preprocess_signature
from .signature_similarity import SignatureSimilarity
from .stamp_analysis import analyze_stamp
from .texture_analysis import analyze_texture
from .visualization import visualize_results
from .schemas import SignatureStampResult, Candidate
from .exceptions import SignatureStampError

WEIGHT_TEXTURE = 0.3
WEIGHT_SIGNATURE = 0.4
WEIGHT_STAMP = 0.3
SUSPICIOUS_RISK_THRESHOLD = 0.5

def _sanitize_file_id(file_id: str) -> str:
    if not isinstance(file_id, str):
        file_id = str(file_id)
    if ".." in file_id or "/" in file_id or "\\" in file_id:
        raise SignatureStampError("Path traversal təhlükəsi.")
    return os.path.basename(file_id)

def analyze_signature_stamp(
    image: np.ndarray,
    file_id: str,
    reference_signatures: Optional[List[np.ndarray]] = None,
    output_dir: str = "outputs"
) -> SignatureStampResult:
    """
    Sənəd şəkli üzərində imza və möhür analizi həyata keçirir.
    Qeyd: Pipeline input olaraq birbaşa NumPy massivi (np.ndarray) qəbul edir.
    """
    errors = []
    candidates = []
    signature_verifications = []
    stamp_analyses = []

    try:
        if image is None or image.size == 0:
            raise SignatureStampError("Giriş şəkli etibarsızdır.")

        safe_file_id = _sanitize_file_id(file_id)
        h, w = image.shape[:2]

        detector = SignatureStampDetector()
        detection_results = detector.detect_stamps_and_signatures(image)

        processed_sigs = []
        max_sig_texture_risk = 0.0

        for sig in detection_results.get("signatures", []):
            bbox = sig["bbox"]
            box_tuple = (bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax)
            sig_crop = image[max(0, bbox.ymin):min(h, bbox.ymax), max(0, bbox.xmin):min(w, bbox.xmax)]
            
            proc_sig = preprocess_signature(image, box_tuple)
            processed_sigs.append(proc_sig)

            sig_tex_risk = 0.0
            if sig_crop.size > 0:
                tex_res = analyze_texture(sig_crop)
                if isinstance(tex_res, dict):
                    sig_tex_risk = tex_res.get("digital_manipulation_probability", 0.0)

            max_sig_texture_risk = max(max_sig_texture_risk, sig_tex_risk)
            is_sig_suspicious = sig_tex_risk > 0.5

            candidates.append(Candidate(
                type="signature",
                bbox=bbox,
                confidence=sig["confidence"],
                is_suspicious=is_sig_suspicious
            ))

        max_stamp_risk = 0.0
        for stp in detection_results.get("stamps", []):
            bbox = stp["bbox"]
            stamp_crop = image[max(0, bbox.ymin):min(h, bbox.ymax), max(0, bbox.xmin):min(w, bbox.xmax)]
            
            stamp_tex_risk = 0.0
            if stamp_crop.size > 0:
                tex_res = analyze_texture(stamp_crop)
                if isinstance(tex_res, dict):
                    stamp_tex_risk = tex_res.get("digital_manipulation_probability", 0.0)

            stamp_res = analyze_stamp(stamp_crop)
            stamp_analyses.append(stamp_res.dict() if hasattr(stamp_res, "dict") else dict(stamp_res))

            is_stamp_failed = not stamp_res.detected
            is_stamp_suspicious = is_stamp_failed or (stamp_tex_risk > 0.5)
            
            item_stamp_risk = 1.0 if is_stamp_failed else stamp_tex_risk
            max_stamp_risk = max(max_stamp_risk, item_stamp_risk)

            candidates.append(Candidate(
                type="stamp",
                bbox=bbox,
                confidence=stp["confidence"],
                is_suspicious=is_stamp_suspicious
            ))

        raw_signature_match_risk = 0.0
        if reference_signatures and processed_sigs:
            for sig_idx, proc_sig in enumerate(processed_sigs):
                best_sim = 0.0
                best_metrics = {}
                for ref_img in reference_signatures:
                    ref_proc = preprocess_signature(ref_img)
                    sim_res = SignatureSimilarity.compare(proc_sig, ref_proc)
                    if sim_res["total_similarity"] > best_sim:
                        best_sim = sim_res["total_similarity"]
                        best_metrics = sim_res

                if best_sim >= 0.7:
                    status = "match"
                    sig_risk_item = 0.0
                elif 0.4 <= best_sim < 0.7:
                    status = "uncertain"
                    sig_risk_item = 0.5
                else:
                    status = "non_match"
                    sig_risk_item = 1.0

                raw_signature_match_risk = max(raw_signature_match_risk, sig_risk_item)
                signature_verifications.append({
                    "signature_index": sig_idx,
                    "similarity_score": best_sim,
                    "metrics": best_metrics,
                    "status": status
                })
        else:
            raw_signature_match_risk = 0.2
            signature_verifications.append({
                "similarity_score": 0.0,
                "metrics": {},
                "status": "not_verified"
            })

        risk_score = (WEIGHT_TEXTURE * max_sig_texture_risk) + \
                     (WEIGHT_SIGNATURE * raw_signature_match_risk) + \
                     (WEIGHT_STAMP * max_stamp_risk)
        risk_score = float(min(1.0, max(0.0, risk_score)))
        
        is_suspicious = (risk_score > SUSPICIOUS_RISK_THRESHOLD) or \
                        (raw_signature_match_risk == 1.0) or \
                        (max_stamp_risk == 1.0) or \
                        any(c.is_suspicious for c in candidates)

        os.makedirs(output_dir, exist_ok=True)
        vis_path = os.path.join(output_dir, f"visualized_{safe_file_id}.png")
        visualization_path = visualize_results(image, candidates, vis_path)

        return SignatureStampResult(
            file_id=safe_file_id,
            image_width=w,
            image_height=h,
            candidates=candidates,
            signature_verifications=signature_verifications,
            stamp_analyses=stamp_analyses,
            risk_score=risk_score,
            is_suspicious=is_suspicious,
            visualization_path=visualization_path,
            status="success",
            errors=errors
        )

    except Exception as e:
        errors.append(str(e))
        return SignatureStampResult(
            file_id="safe_failed_doc",
            image_width=image.shape[1] if image is not None and image.ndim >= 2 else 0,
            image_height=image.shape[0] if image is not None and image.ndim >= 2 else 0,
            risk_score=1.0,
            is_suspicious=True,
            status="failed",
            errors=errors
        )