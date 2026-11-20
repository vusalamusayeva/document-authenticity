import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    status: str = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "error_type": self.error_type,
        }


class JobQueue:
    def __init__(self, maxsize: int = 0) -> None:
        self._queue: Queue[tuple[str, Callable[[], Any]]] = Queue(maxsize=maxsize)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[], Any], job_id: Optional[str] = None) -> str:
        if not callable(fn):
            raise TypeError("fn çağırıla bilən funksiya olmalıdır.")
        jid = job_id or str(uuid.uuid4())
        record = JobRecord(job_id=jid, status="queued")
        with self._lock:
            if jid in self._jobs:
                raise ValueError(f"Job ID artıq mövcuddur: {jid}")
            self._jobs[jid] = record
        self._queue.put((jid, fn))
        return jid

    def pop_next(self, timeout: float = 0.2) -> Optional[tuple[str, Callable[[], Any]]]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.status = "running"
                rec.started_at = _utc_now()

    def mark_completed(self, job_id: str, result: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.status = "completed"
                rec.result = result
                rec.finished_at = _utc_now()

    def mark_failed(self, job_id: str, exc: Exception) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.status = "failed"
                rec.error = str(exc)
                rec.error_type = type(exc).__name__
                rec.finished_at = _utc_now()

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._jobs.get(job_id)
            return rec.to_dict() if rec is not None else None

    def qsize(self) -> int:
        return self._queue.qsize()


class BackgroundWorker:
    def __init__(self, job_queue: JobQueue, num_workers: int = 1) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers müsbət tam ədəd olmalıdır.")
        self.job_queue = job_queue
        self.num_workers = int(num_workers)
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads and any(t.is_alive() for t in self._threads):
            return
        self._stop_event.clear()
        self._threads = []
        for idx in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"doc-auth-worker-{idx}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self.job_queue.pop_next(timeout=0.1)
            if item is None:
                continue
            job_id, fn = item
            self.job_queue.mark_running(job_id)
            try:
                res = fn()
                self.job_queue.mark_completed(job_id, res)
            except Exception as exc:
                self.job_queue.mark_failed(job_id, exc)
            finally:
                self.job_queue._queue.task_done()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        deadline = time.monotonic() + timeout
        for t in self._threads:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)


Path("src/doc_auth/utils/job_queue.py").write_text(
    Path(__file__).read_text(encoding="utf-8").split('\nPath("src/doc_auth/utils/job_queue.py")')[0],
    encoding="utf-8",
) if False else None
