import os
from typing import Any

import numpy as np

from .detector import SignatureStampDetector
from .exceptions import SignatureStampError
from .schemas import Candidate, SignatureStampItem, SignatureStampResult
from .signature_preprocess import preprocess_signature
from .signature_similarity import SignatureSimilarity
from .stamp_analysis import analyze_stamp
from .texture_analysis import analyze_texture
from .visualization import visualize_results

WEIGHT_TEXTURE = 0.3
WEIGHT_SIGNATURE = 0.4
WEIGHT_STAMP = 0.3
SUSPICIOUS_RISK_THRESHOLD = 0.5


def _sanitize_file_id(file_id: str) -> str:
    if not isinstance(file_id, str):
        file_id = str(file_id)
    if ".." in file_id or "/" in file_id or "\\" in file_id:
        raise SignatureStampError("Path traversal təhlükəsi.")
    cleaned = os.path.basename(file_id).strip()
    if not cleaned:
        raise SignatureStampError("Etibarlı file_id tələb olunur.")
    return cleaned


def _extract_risk_value(tex_res: Any, keys: tuple[str, ...]) -> float:
    if not isinstance(tex_res, dict):
        return 0.0
    for key in keys:
        val = tex_res.get(key)
        if isinstance(val, (int, float, np.number)):
            return float(min(1.0, max(0.0, val)))
    return 0.0


def analyze_signature_stamp(
    image: np.ndarray,
    file_id: str,
    reference_signatures: list[np.ndarray] | None = None,
    output_dir: str = "outputs",
) -> SignatureStampResult:
    """
    Sənəd şəkli üzərində imza və möhür analizi həyata keçirir.
    """
    errors: list[str] = []
    candidates: list[Candidate] = []
    items: list[SignatureStampItem] = []
    signature_verifications: list[dict[str, Any]] = []

    try:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise SignatureStampError("Giriş şəkli etibarsızdır.")

        safe_file_id = _sanitize_file_id(file_id)
        h, w = image.shape[:2]

        detector = SignatureStampDetector()
        detection_results = detector.detect_stamps_and_signatures(image)

        processed_sigs: list[np.ndarray] = []
        max_sig_texture_risk = 0.0

        for idx, sig in enumerate(detection_results.get("signatures", [])):
            bbox = sig["bbox"]
            x1, y1, x2, y2 = bbox.as_int_list()
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)

            sig_crop = image[y1_c:y2_c, x1_c:x2_c] if (x2_c > x1_c and y2_c > y1_c) else np.empty((0, 0), dtype=image.dtype)
            proc_sig = preprocess_signature(image, (x1_c, y1_c, x2_c, y2_c))
            processed_sigs.append(proc_sig)

            sig_tex_risk = 0.0
            if sig_crop.size > 0:
                tex_res = analyze_texture(sig_crop)
                sig_tex_risk = _extract_risk_value(
                    tex_res,
                    ("digital_manipulation_probability", "forgery_probability", "score"),
                )

            max_sig_texture_risk = max(max_sig_texture_risk, sig_tex_risk)
            is_sig_suspicious = sig_tex_risk > 0.5
            conf = float(sig.get("confidence", 0.75))

            cand = Candidate(
                candidate_id=f"sig_{idx}",
                element_type="signature",
                label="signature",
                bbox=bbox.as_list(),
                score=conf,
                confidence=conf,
                is_authentic=not is_sig_suspicious,
                metadata={"is_suspicious": is_sig_suspicious, "texture_risk": sig_tex_risk},
            )
            candidates.append(cand)
            items.append(
                SignatureStampItem(
                    element_type="signature",
                    bbox=bbox.as_list(),
                    confidence=conf,
                    is_authentic=not is_sig_suspicious,
                    extra={"candidate_id": cand.candidate_id, "is_suspicious": is_sig_suspicious},
                )
            )

        max_stamp_risk = 0.0
        for idx, stp in enumerate(detection_results.get("stamps", [])):
            bbox = stp["bbox"]
            x1, y1, x2, y2 = bbox.as_int_list()
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)

            stamp_crop = image[y1_c:y2_c, x1_c:x2_c] if (x2_c > x1_c and y2_c > y1_c) else np.empty((0, 0), dtype=image.dtype)

            stamp_tex_risk = 0.0
            if stamp_crop.size > 0:
                tex_res = analyze_texture(stamp_crop)
                stamp_tex_risk = _extract_risk_value(
                    tex_res,
                    ("digital_manipulation_probability", "forgery_probability", "score"),
                )

            stamp_res = analyze_stamp(stamp_crop if stamp_crop.size > 0 else image)
            stamp_detected = bool(getattr(stamp_res, "detected", True))
            is_stamp_failed = not stamp_detected
            is_stamp_suspicious = is_stamp_failed or (stamp_tex_risk > 0.5)

            item_stamp_risk = 1.0 if is_stamp_failed else stamp_tex_risk
            max_stamp_risk = max(max_stamp_risk, item_stamp_risk)
            conf = float(stp.get("confidence", 0.5))

            cand = Candidate(
                candidate_id=f"stamp_{idx}",
                element_type="stamp",
                label="stamp",
                bbox=bbox.as_list(),
                score=conf,
                confidence=conf,
                is_authentic=not is_stamp_suspicious,
                metadata={
                    "is_suspicious": is_stamp_suspicious,
                    "stamp_detected": stamp_detected,
                    "texture_risk": stamp_tex_risk,
                },
            )
            candidates.append(cand)
            items.append(
                SignatureStampItem(
                    element_type="stamp",
                    bbox=bbox.as_list(),
                    confidence=conf,
                    is_authentic=not is_stamp_suspicious,
                    extra={
                        "candidate_id": cand.candidate_id,
                        "is_suspicious": is_stamp_suspicious,
                        "stamp_detected": stamp_detected,
                    },
                )
            )

        raw_signature_match_risk = 0.0
        if reference_signatures and processed_sigs:
            for sig_idx, proc_sig in enumerate(processed_sigs):
                best_sim = 0.0
                best_metrics: dict[str, Any] = {}
                for ref_img in reference_signatures:
                    ref_proc = preprocess_signature(ref_img)
                    sim_res = SignatureSimilarity.compare(proc_sig, ref_proc)
                    sim_val = float(sim_res.get("total_similarity", 0.0))
                    if sim_val > best_sim:
                        best_sim = sim_val
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
                    "status": status,
                })
        else:
            raw_signature_match_risk = 0.2
            signature_verifications.append({
                "similarity_score": 0.0,
                "metrics": {},
                "status": "not_verified",
            })

        risk_score = (
            (WEIGHT_TEXTURE * max_sig_texture_risk)
            + (WEIGHT_SIGNATURE * raw_signature_match_risk)
            + (WEIGHT_STAMP * max_stamp_risk)
        )
        risk_score = float(min(1.0, max(0.0, risk_score)))

        is_suspicious = bool(
            (risk_score > SUSPICIOUS_RISK_THRESHOLD)
            or (raw_signature_match_risk == 1.0)
            or (max_stamp_risk == 1.0)
            or any(bool(c.metadata.get("is_suspicious")) for c in candidates)
        )

        os.makedirs(output_dir, exist_ok=True)
        vis_path = os.path.join(output_dir, f"visualized_{safe_file_id}.png")
        visualization_path = visualize_results(image, candidates, vis_path)

        return SignatureStampResult(
            file_id=safe_file_id,
            status="success",
            overall_score=float(round(1.0 - risk_score, 4)),
            risk_score=risk_score,
            is_suspicious=is_suspicious,
            is_tampered=is_suspicious,
            visualization_path=visualization_path,
            signature_verifications=signature_verifications,
            items=items,
            errors=errors,
        )

    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(str(exc))
        return SignatureStampResult(
            file_id="safe_failed_doc",
            status="failed",
            overall_score=0.0,
            risk_score=1.0,
            is_suspicious=True,
            is_tampered=True,
            visualization_path=None,
            signature_verifications=signature_verifications,
            items=items,
            errors=errors,
        )