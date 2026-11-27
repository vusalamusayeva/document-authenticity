import json
import time
from pathlib import Path
from typing import Any

import pytest

from doc_auth.utils.job_queue import BackgroundWorker, JobQueue, JobRecord


def test_job_queue_submit_errors_idempotency_and_qsize(tmp_path: Path) -> None:
    store = tmp_path / "jobs.json"
    q = JobQueue(store_path=store)

    with pytest.raises(TypeError):
        q.submit("not-callable")  # type: ignore[arg-type]

    jid = q.submit(lambda: {"ok": 1}, job_id="job-1", max_retries=1, payload={"doc": "a.pdf"})
    assert jid == "job-1"
    assert q.qsize() == 1

    # Idempotent re-submit returns same job_id without duplicating queue entry
    same_jid = q.submit(lambda: {"ok": 2}, job_id="job-1", idempotent=True)
    assert same_jid == "job-1"
    assert q.qsize() == 1

    # Non-idempotent duplicate raises ValueError
    with pytest.raises(ValueError):
        q.submit(lambda: {"ok": 3}, job_id="job-1", idempotent=False)

    assert q.requeue_for_retry("missing-job", RuntimeError("boom")) is False


def test_job_queue_persistence_recovery_and_register_handler(tmp_path: Path) -> None:
    store = tmp_path / "state.json"
    data: dict[str, Any] = {
        "not-a-dict": "ignored",
        "job-retry": {
            "job_id": "job-retry",
            "status": "running",
            "attempts": 1,
            "max_retries": 2,
        },
        "job-exhausted": {
            "job_id": "job-exhausted",
            "status": "running",
            "attempts": 3,
            "max_retries": 2,
        },
    }
    store.write_text(json.dumps(data), encoding="utf-8")

    q = JobQueue(store_path=store, recover_running_on_load=True)
    retry_status = q.get_status("job-retry")
    exhausted_status = q.get_status("job-exhausted")
    assert retry_status is not None and retry_status["status"] == "queued"
    assert exhausted_status is not None and exhausted_status["status"] == "failed"
    assert exhausted_status["error_type"] == "WorkerInterruptedError"

    # Registering handler for an already queued job pushes it into internal queue
    q.register_handler("job-retry", lambda: {"recovered": True})
    popped = q.pop_next(timeout=0.1)
    assert popped is not None
    assert popped[0] == "job-retry"
    assert popped[1]() == {"recovered": True}


def test_background_worker_validation_success_and_failure() -> None:
    q = JobQueue()
    with pytest.raises(ValueError):
        BackgroundWorker(q, num_workers=0)

    worker = BackgroundWorker(q, num_workers=1)
    worker.start()
    # Calling start() while alive is a safe no-op
    worker.start()

    ok_id = q.submit(lambda: {"done": True}, max_retries=0)
    fail_id = q.submit(lambda: (_ for _ in ()).throw(RuntimeError("hard fail")), max_retries=0)

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        st_ok = q.get_status(ok_id)
        st_fail = q.get_status(fail_id)
        if st_ok and st_ok["status"] == "completed" and st_fail and st_fail["status"] == "failed":
            break
        time.sleep(0.02)

    worker.stop(timeout=2.0)

    final_ok = q.get_status(ok_id)
    final_fail = q.get_status(fail_id)
    assert final_ok is not None and final_ok["status"] == "completed" and final_ok["result"] == {"done": True}
    assert final_fail is not None and final_fail["status"] == "failed" and final_fail["error_type"] == "RuntimeError"

    rec = JobRecord.from_dict({"job_id": "rec-1"})
    assert rec.to_dict()["job_id"] == "rec-1"
