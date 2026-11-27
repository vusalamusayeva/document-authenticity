from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    if len(box_a) < 4 or len(box_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = float(box_a[0]), float(box_a[1]), float(box_a[2]), float(box_a[3])
    bx1, by1, bx2, by2 = float(box_b[0]), float(box_b[1]), float(box_b[2]), float(box_b[3])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def assign_face_tracks(
    frame_items: list[dict[str, Any]],
    iou_threshold: float = 0.3,
    max_time_gap: float = 1.0,
) -> list[dict[str, Any]]:
    active_tracks: list[dict[str, Any]] = []
    next_track_id = 1
    tracked_records: list[dict[str, Any]] = []
    prev_frame_ts: float | None = None

    for idx, item in enumerate(frame_items):
        frame_idx = item.get("frame_index", idx)
        ts = float(item.get("timestamp", float(idx)))
        faces = item.get("faces", [])

        scene_cut_here = False
        if prev_frame_ts is not None and (ts - prev_frame_ts) > max_time_gap:
            scene_cut_here = True
        prev_frame_ts = ts

        used_track_ids = set()
        for face in faces:
            bbox = face.get("bbox", [0, 0, 0, 0])
            score = face.get("score")

            best_track = None
            best_iou = 0.0
            for tr in active_tracks:
                if tr["track_id"] in used_track_ids:
                    continue
                if abs(ts - tr["last_ts"]) > max_time_gap:
                    continue
                iou = compute_iou(bbox, tr["bbox"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_track = tr

            if best_track is not None:
                track_id = best_track["track_id"]
                best_track["bbox"] = bbox
                best_track["last_ts"] = ts
                used_track_ids.add(track_id)
            else:
                track_id = next_track_id
                next_track_id += 1
                active_tracks.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "last_ts": ts,
                })
                used_track_ids.add(track_id)

            tracked_records.append({
                "track_id": track_id,
                "frame_index": frame_idx,
                "timestamp": ts,
                "bbox": bbox,
                "score": score,
                "scene_cut": scene_cut_here,
            })

    return tracked_records


class TemporalAnalyzer:
    def __init__(
        self,
        max_time_gap: float = 1.0,
        high_jump_threshold: float = 0.35,
    ) -> None:
        self.max_time_gap = float(max_time_gap)
        self.high_jump_threshold = float(high_jump_threshold)

    def analyze(self, tracked_faces: list[dict[str, Any]] | None) -> dict[str, Any]:
        if not tracked_faces:
            return {
                "status": "no_data",
                "temporal_score": None,
                "temporal_risk": None,
                "high_jumps": 0,
                "scene_cuts_ignored": 0,
                "tracks_analyzed": 0,
            }

        grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for rec in tracked_faces:
            grouped[rec.get("track_id", 0)].append(rec)

        deltas: list[float] = []
        high_jumps = 0
        scene_cuts_ignored = 0

        for items in grouped.values():
            ordered = sorted(
                items,
                key=lambda x: (float(x.get("timestamp", 0.0)), int(x.get("frame_index", 0))),
            )
            prev_valid: dict[str, Any] | None = None
            for cur in ordered:
                if cur.get("scene_cut") is True:
                    scene_cuts_ignored += 1
                    prev_valid = cur
                    continue
                if prev_valid is None:
                    prev_valid = cur
                    continue
                dt = float(cur.get("timestamp", 0.0)) - float(prev_valid.get("timestamp", 0.0))
                if dt < 0.0 or dt > self.max_time_gap:
                    scene_cuts_ignored += 1
                    prev_valid = cur
                    continue

                s1 = prev_valid.get("score")
                s2 = cur.get("score")
                if isinstance(s1, (int, float)) and isinstance(s2, (int, float)):
                    diff = abs(float(s2) - float(s1))
                    deltas.append(diff)
                    if diff >= self.high_jump_threshold:
                        high_jumps += 1
                prev_valid = cur

        temporal_score = round(sum(deltas) / len(deltas), 4) if deltas else 0.0

        return {
            "status": "success",
            "temporal_score": temporal_score,
            "temporal_risk": temporal_score,
            "high_jumps": high_jumps,
            "scene_cuts_ignored": scene_cuts_ignored,
            "tracks_analyzed": len(grouped),
        }