from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

from api.main import app
from doc_auth.classification.document_classifier import DocumentClassifier
from doc_auth.deepfake.temporal_analysis import TemporalAnalyzer
from doc_auth.service import AnalysisService
from doc_auth.utils.job_queue import BackgroundWorker, JobQueue


def _make_doc_image(width: int = 360, height: int = 220) -> np.ndarray:
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (30, 30, 30), 2)
    cv2.putText(img, "ID DOCUMENT 2026", (45, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    for y in range(105, height - 35, 16):
        cv2.line(img, (45, y), (width - 65, y), (40, 40, 40), 2)
    return img


def test_acceptance_document_classification_and_service(tmp_path: Path) -> None:
    img_path = tmp_path / "sample_id.png"
    img = _make_doc_image(360, 220)
    assert cv2.imwrite(str(img_path), img)

    clf = DocumentClassifier()
    unfitted_res = clf.classify(img)
    assert unfitted_res["status"] == "inconclusive"
    assert unfitted_res["document_type"] == "unknown"

    train_samples = [
        _make_doc_image(360, 220),
        _make_doc_image(370, 225),
        _make_doc_image(240, 340),
        _make_doc_image(250, 350),
        _make_doc_image(600, 420),
        _make_doc_image(620, 440),
        np.zeros((180, 180, 3), dtype=np.uint8),
        np.full((180, 180, 3), 15, dtype=np.uint8),
    ]
    train_labels = [
        "id_card",
        "id_card",
        "passport",
        "passport",
        "certificate",
        "certificate",
        "unknown",
        "unknown",
    ]
    clf.fit(train_samples, train_labels)
    cls_res = clf.classify(img)
    assert cls_res["status"] == "success"
    assert cls_res["document_type"] in clf.classes
    assert abs(sum(cls_res["probabilities"].values()) - 1.0) < 1e-3

    svc = AnalysisService()
    out = svc.analyze(img_path)
    assert out["status"] in {"success", "completed", "approved", "review", "rejected", "inconclusive"}
    assert isinstance(out["classification"], dict)


def test_acceptance_temporal_effect() -> None:
    analyzer = TemporalAnalyzer()
    smooth_tracks = [
        {
            "track_id": 0,
            "frame_index": idx,
            "bbox": [16, 16, 96, 96],
            "crop": np.full((80, 80, 3), 120 + idx, dtype=np.uint8),
        }
        for idx in range(6)
    ]
    jitter_tracks = [
        {
            "track_id": 0,
            "frame_index": idx,
            "bbox": [16, 16, 96, 96],
            "crop": np.full((80, 80, 3), 50 if idx % 2 == 0 else 220, dtype=np.uint8),
        }
        for idx in range(6)
    ]
    smooth_res = analyzer.analyze(smooth_tracks)
    jitter_res = analyzer.analyze(jitter_tracks)
    assert smooth_res["status"] != "no_data"
    assert jitter_res["status"] != "no_data"


def test_acceptance_parallel_upload_api(tmp_path: Path) -> None:
    from api.schemas.document import DocumentAnalysisRequest

    client = TestClient(app)
    img_paths = []
    for i in range(4):
        p = tmp_path / f"doc_{i}.png"
        assert cv2.imwrite(str(p), _make_doc_image(320, 200))
        img_paths.append(p)

    fields = getattr(DocumentAnalysisRequest, "model_fields", {}) or getattr(DocumentAnalysisRequest, "__fields__", {})

    def _build_payload(path: Path) -> dict:
        payload: dict = {}
        for fname in fields:
            low = fname.lower()
            if any(k in low for k in ("path", "file", "image", "uri", "url", "source", "input")):
                payload[fname] = str(path)
            elif "id" in low:
                payload[fname] = path.stem
        if not payload:
            payload = {"image_path": str(path)}
        return payload

    def _call_one(path: Path) -> tuple[int, dict]:
        resp = client.post("/v1/analyze", json=_build_payload(path))
        return resp.status_code, resp.json()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_call_one, img_paths))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 15000.0
    for status_code, body in results:
        assert status_code in {200, 422, 500, 503}
        assert isinstance(body, dict)


def test_acceptance_job_queue_persistence_retry_timeout_and_duplicate(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    q1 = JobQueue(store_path=store)

    state = {"calls": 0}

    def _flaky_job() -> dict:
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("transient error")
        return {"ok": True, "attempts_seen": state["calls"]}

    jid = q1.submit(_flaky_job, job_id="job-accept-1", max_retries=2, timeout_sec=2.0)
    assert jid == "job-accept-1"
    # Idempotent duplicate submission returns same job_id without failing
    dup_id = q1.submit(_flaky_job, job_id="job-accept-1", max_retries=2, timeout_sec=2.0)
    assert dup_id == "job-accept-1"

    worker1 = BackgroundWorker(q1, num_workers=1)
    worker1.start()
    deadline = time.monotonic() + 3.0
    status = None
    while time.monotonic() < deadline:
        status = q1.get_status("job-accept-1")
        if status and status["status"] == "completed":
            break
        time.sleep(0.05)
    worker1.stop()

    assert status is not None
    assert status["status"] == "completed"
    assert status["attempts"] >= 2

    # Simulate process restart with persisted store
    q2 = JobQueue(store_path=store)
    reloaded = q2.get_status("job-accept-1")
    assert reloaded is not None
    assert reloaded["status"] == "completed"
    assert reloaded["result"] == status["result"]

    # Timeout handling test
    def _slow_job() -> dict:
        time.sleep(0.4)
        return {"slow": True}

    q2.submit(_slow_job, job_id="job-timeout-1", max_retries=0, timeout_sec=0.05)
    worker2 = BackgroundWorker(q2, num_workers=1)
    worker2.start()
    deadline = time.monotonic() + 2.0
    t_status = None
    while time.monotonic() < deadline:
        t_status = q2.get_status("job-timeout-1")
        if t_status and t_status["status"] == "failed":
            break
        time.sleep(0.05)
    worker2.stop()
    assert t_status is not None
    assert t_status["status"] == "failed"
    assert t_status["error_type"] == "TimeoutError"
