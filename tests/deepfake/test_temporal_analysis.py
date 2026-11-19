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
    assert np.isfinite(res["temporal_score"])