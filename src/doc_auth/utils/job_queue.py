from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
    attempts: int = 0
    max_retries: int = 1
    timeout_sec: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None

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
            "attempts": self.attempts,
            "max_retries": self.max_retries,
            "timeout_sec": self.timeout_sec,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobRecord":
        return cls(
            job_id=str(data["job_id"]),
            status=str(data.get("status", "queued")),
            created_at=str(data.get("created_at") or _utc_now()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            result=data.get("result"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            attempts=int(data.get("attempts", 0)),
            max_retries=int(data.get("max_retries", 1)),
            timeout_sec=data.get("timeout_sec"),
            payload=data.get("payload"),
        )


class JobQueue:
    def __init__(
        self,
        maxsize: int = 0,
        store_path: Optional[str | Path] = None,
        recover_running_on_load: bool = True,
    ) -> None:
        self._queue: Queue[tuple[str, Callable[[], Any]]] = Queue(maxsize=maxsize)
        self._jobs: Dict[str, JobRecord] = {}
        self._handlers: Dict[str, Callable[[], Any]] = {}
        self._lock = threading.Lock()
        self.store_path = Path(store_path) if store_path is not None else None
        if self.store_path is not None:
            self._load_from_disk(recover_running=recover_running_on_load)

    def _persist_locked(self) -> None:
        if self.store_path is None:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {jid: rec.to_dict() for jid, rec in self._jobs.items()}
        tmp_file = self.store_path.with_suffix(self.store_path.suffix + f".{os.getpid()}.tmp")
        tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_file, self.store_path)

    def _load_from_disk(self, recover_running: bool = True) -> None:
        assert self.store_path is not None
        if not self.store_path.exists():
            return
        raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return
        for jid, item in raw.items():
            if not isinstance(item, dict):
                continue
            rec = JobRecord.from_dict(item)
            if rec.status == "running" and recover_running:
                if rec.attempts <= rec.max_retries:
                    rec.status = "queued"
                else:
                    rec.status = "failed"
                    rec.error = rec.error or "Worker dayanmas? s?b?bind?n i? yar?m??q qald?."
                    rec.error_type = rec.error_type or "WorkerInterruptedError"
                    rec.finished_at = _utc_now()
            self._jobs[str(jid)] = rec
            if rec.status == "queued":
                handler = self._handlers.get(str(jid))
                if handler is not None:
                    self._queue.put((str(jid), handler))
        self._persist_locked()

    def register_handler(self, job_id: str, fn: Callable[[], Any]) -> None:
        with self._lock:
            self._handlers[job_id] = fn
            rec = self._jobs.get(job_id)
            if rec is not None and rec.status == "queued":
                self._queue.put((job_id, fn))

    def submit(
        self,
        fn: Callable[[], Any],
        job_id: Optional[str] = None,
        max_retries: int = 1,
        timeout_sec: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
        idempotent: bool = True,
    ) -> str:
        if not callable(fn):
            raise TypeError("fn ?a??r?la bil?n funksiya olmal?d?r.")
        jid = job_id or str(uuid.uuid4())
        with self._lock:
            if jid in self._jobs:
                if idempotent:
                    self._handlers[jid] = fn
                    return jid
                raise ValueError(f"Job ID art?q m?vcuddur: {jid}")
            record = JobRecord(
                job_id=jid,
                status="queued",
                max_retries=max(0, int(max_retries)),
                timeout_sec=timeout_sec,
                payload=payload,
            )
            self._jobs[jid] = record
            self._handlers[jid] = fn
            self._persist_locked()
        self._queue.put((jid, fn))
        return jid

    def pop_next(self, timeout: float = 0.2) -> Optional[tuple[str, Callable[[], Any]]]:
        try:
            jid, fn = self._queue.get(timeout=timeout)
            with self._lock:
                handler = self._handlers.get(jid, fn)
            return jid, handler
        except Empty:
            return None

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.status = "running"
                rec.attempts += 1
                rec.started_at = _utc_now()
                self._persist_locked()

    def mark_completed(self, job_id: str, result: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.status = "completed"
                rec.result = result
                rec.error = None
                rec.error_type = None
                rec.finished_at = _utc_now()
                self._persist_locked()

    def requeue_for_retry(self, job_id: str, exc: Exception) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return False
            if rec.attempts <= rec.max_retries:
                rec.status = "queued"
                rec.error = str(exc)
                rec.error_type = type(exc).__name__
                handler = self._handlers.get(job_id)
                self._persist_locked()
                if handler is not None:
                    self._queue.put((job_id, handler))
                return True
            return False

    def mark_failed(self, job_id: str, exc: Exception) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None:
                rec.status = "failed"
                rec.error = str(exc)
                rec.error_type = type(exc).__name__
                rec.finished_at = _utc_now()
                self._persist_locked()

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rec = self._jobs.get(job_id)
            return rec.to_dict() if rec is not None else None

    def qsize(self) -> int:
        return self._queue.qsize()


class BackgroundWorker:
    def __init__(self, job_queue: JobQueue, num_workers: int = 1) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers m?sb?t tam ?d?d olmal?d?r.")
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

    def _run_with_timeout(self, fn: Callable[[], Any], timeout_sec: Optional[float]) -> Any:
        if timeout_sec is None:
            return fn()
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fn)
            return fut.result(timeout=float(timeout_sec))

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            item = self.job_queue.pop_next(timeout=0.1)
            if item is None:
                continue
            job_id, fn = item
            self.job_queue.mark_running(job_id)
            rec = self.job_queue.get_status(job_id) or {}
            timeout_sec = rec.get("timeout_sec")
            try:
                res = self._run_with_timeout(fn, timeout_sec=timeout_sec)
                self.job_queue.mark_completed(job_id, res)
            except FutureTimeoutError as exc:
                timeout_exc = TimeoutError(f"Job {timeout_sec}s limitini a?d?.")
                if not self.job_queue.requeue_for_retry(job_id, timeout_exc):
                    self.job_queue.mark_failed(job_id, timeout_exc)
            except Exception as exc:
                if not self.job_queue.requeue_for_retry(job_id, exc):
                    self.job_queue.mark_failed(job_id, exc)
            finally:
                self.job_queue._queue.task_done()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        deadline = time.monotonic() + timeout
        for t in self._threads:
            remaining = max(0.0, deadline - time.monotonic())
            t.join(timeout=remaining)
