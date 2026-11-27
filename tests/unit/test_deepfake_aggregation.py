import pytest

from doc_auth.deepfake.aggregation import (
    aggregate_face_predictions,
    aggregate_frame_scores,
    aggregate_visual_predictions,
)


def test_aggregate_frame_scores_modes_and_filtering() -> None:
    assert aggregate_frame_scores([]) is None
    assert aggregate_frame_scores([True, False, "bad", None]) is None  # type: ignore[list-item]

    scores = [0.2, 0.8, 0.5, True]  # type: ignore[list-item]
    assert aggregate_frame_scores(scores, method="mean") == pytest.approx(0.5)
    assert aggregate_frame_scores(scores, method="max") == pytest.approx(0.8)
    assert aggregate_frame_scores(scores, method="min") == pytest.approx(0.2)
    assert aggregate_frame_scores([0.1, 0.9, 0.6], method="top_k", top_k=2) == pytest.approx(0.75)


def test_aggregate_face_and_visual_predictions() -> None:
    empty_face = aggregate_face_predictions([])
    assert empty_face["status"] == "no_data"
    assert empty_face["aggregated_score"] is None
    assert empty_face["count"] == 0

    preds = [{"score": 0.2}, {"score": 0.8}, {"score": 0.4}]
    face_res = aggregate_face_predictions(preds, method="max")
    assert face_res["status"] == "success"
    assert face_res["aggregated_score"] == pytest.approx(0.8)
    assert face_res["count"] == 3
    assert face_res["faces_count"] == 3

    vis_res = aggregate_visual_predictions(preds, method="top_k", top_k=2, frames_count=10)
    assert vis_res["status"] == "success"
    assert vis_res["frames_count"] == 10
    assert vis_res["faces_count"] == 3
    assert vis_res["max_score"] == pytest.approx(0.8)
    assert vis_res["mean_score"] == pytest.approx(0.4667, abs=1e-4)

    vis_default_frames = aggregate_visual_predictions([])
    assert vis_default_frames["status"] == "no_data"
    assert vis_default_frames["frames_count"] == 0
    assert vis_default_frames["max_score"] is None
    assert vis_default_frames["mean_score"] is None
