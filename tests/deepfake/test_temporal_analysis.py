import numpy as np

from doc_auth.deepfake.temporal_analysis import TemporalAnalyzer, assign_face_tracks


def test_same_face_track_temporal_consistency() -> None:
    frame_items = [
        {"frame_index": 0, "timestamp": 0.0, "faces": [{"bbox": [50, 50, 150, 150], "score": 0.2}]},
        {"frame_index": 1, "timestamp": 0.5, "faces": [{"bbox": [54, 52, 154, 152], "score": 0.25}]},
        {"frame_index": 2, "timestamp": 1.0, "faces": [{"bbox": [58, 54, 158, 154], "score": 0.22}]},
    ]
    tracked = assign_face_tracks(frame_items, iou_threshold=0.3, max_time_gap=1.0)
    analyzer = TemporalAnalyzer(max_time_gap=1.0)
    res = analyzer.analyze(tracked)
    assert res["status"] == "success"
    assert res["tracks_analyzed"] == 1
    assert isinstance(res["temporal_score"], float)


def test_multiple_faces_same_frame_and_scene_cut() -> None:
    frame_items = [
        {
            "frame_index": 0,
            "timestamp": 0.0,
            "faces": [
                {"bbox": [20, 20, 100, 100], "score": 0.1},
                {"bbox": [300, 20, 380, 100], "score": 0.8},
            ],
        },
        {
            "frame_index": 5,
            "timestamp": 6.0,
            "faces": [{"bbox": [20, 20, 100, 100], "score": 0.15}],
        },
    ]
    tracked = assign_face_tracks(frame_items, iou_threshold=0.3, max_time_gap=1.0)
    analyzer = TemporalAnalyzer(max_time_gap=1.0)
    res = analyzer.analyze(tracked)
    assert res["tracks_analyzed"] >= 2
    assert res["scene_cuts_ignored"] >= 1
    assert isinstance(res["temporal_score"], float)
from doc_auth.deepfake.temporal_analysis import TemporalAnalyzer, compute_iou


def test_compute_iou_short_box_and_zero_union() -> None:
    assert compute_iou([0, 0, 10], [0, 0, 10, 10]) == 0.0
    assert compute_iou([0, 0, 10, 10], [5, 5]) == 0.0
    assert compute_iou([10, 10, 10, 10], [10, 10, 10, 10]) == 0.0


def test_temporal_analyzer_no_data_time_gap_and_high_jumps() -> None:
    analyzer = TemporalAnalyzer(max_time_gap=1.0, high_jump_threshold=0.35)

    empty_res = analyzer.analyze([])
    assert empty_res["status"] == "no_data"
    assert empty_res["temporal_score"] is None

    none_res = analyzer.analyze(None)
    assert none_res["status"] == "no_data"

    records = [
        # Track 1: initial score 0.1 -> high jump to 0.6 at t=0.5 -> negative/large dt jumps
        {"track_id": 1, "frame_index": 0, "timestamp": 0.0, "score": 0.1},
        {"track_id": 1, "frame_index": 1, "timestamp": 0.5, "score": 0.6},
        {"track_id": 1, "frame_index": 2, "timestamp": 0.2, "score": 0.9},
        {"track_id": 1, "frame_index": 3, "timestamp": 5.0, "score": 0.1},
    ]
    res = analyzer.analyze(records)
    assert res["status"] == "success"
    assert res["high_jumps"] == 1
    assert res["scene_cuts_ignored"] == 1
from doc_auth.deepfake.temporal_analysis import compute_iou


class _ZeroUnionFloat(float):
    def __sub__(self, other: object) -> float:
        return 0.0


def test_compute_iou_returns_zero_when_union_area_is_zero() -> None:
    box_a = [_ZeroUnionFloat(0.0), _ZeroUnionFloat(0.0), _ZeroUnionFloat(2.0), _ZeroUnionFloat(2.0)]
    box_b = [0.0, 0.0, 2.0, 2.0]
    assert compute_iou(box_a, box_b) == 0.0
