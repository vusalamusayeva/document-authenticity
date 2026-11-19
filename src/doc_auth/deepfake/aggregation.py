from typing import Any, Dict, List, Optional, Sequence


def aggregate_frame_scores(
    scores: Sequence[float],
    method: str = "mean",
    top_k: int = 3,
) -> Optional[float]:
    valid = [float(s) for s in scores if isinstance(s, (int, float)) and not isinstance(s, bool)]
    if not valid:
        return None

    mode = method.lower().strip()
    if mode == "max":
        return float(max(valid))
    if mode == "min":
        return float(min(valid))
    if mode == "top_k":
        k = max(1, min(int(top_k), len(valid)))
        top_vals = sorted(valid, reverse=True)[:k]
        return float(sum(top_vals) / len(top_vals))
    return float(sum(valid) / len(valid))


def aggregate_face_predictions(
    predictions: List[Dict[str, Any]],
    method: str = "max",
    top_k: int = 3,
) -> Dict[str, Any]:
    if not predictions:
        return {
            "status": "no_data",
            "aggregated_score": None,
            "count": 0,
            "faces_count": 0,
            "method": method,
        }

    scores = [p["score"] for p in predictions if isinstance(p.get("score"), (int, float))]
    agg = aggregate_frame_scores(scores, method=method, top_k=top_k)
    return {
        "status": "success" if agg is not None else "no_data",
        "aggregated_score": round(agg, 4) if agg is not None else None,
        "count": len(scores),
        "faces_count": len(predictions),
        "method": method,
    }


def aggregate_visual_predictions(
    predictions: List[Dict[str, Any]],
    method: str = "top_k",
    top_k: int = 3,
    frames_count: Optional[int] = None,
) -> Dict[str, Any]:
    items = predictions or []
    base = aggregate_face_predictions(predictions=items, method=method, top_k=top_k)
    scores = [
        float(p["score"])
        for p in items
        if isinstance(p.get("score"), (int, float)) and not isinstance(p.get("score"), bool)
    ]
    return {
        **base,
        "frames_count": frames_count if frames_count is not None else len(items),
        "faces_count": len(items),
        "max_score": round(max(scores), 4) if scores else None,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
    }