"""Execução de runs do pipeline em background, com rastreamento de status."""
from __future__ import annotations

import contextlib
import io
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Job:
    run_id: str
    name: str
    status: str = "queued"  # queued | running | completed | failed | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    logs: str = ""
    result: dict | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "logs": self.logs,
            "result": self.result,
            "error": self.error,
        }


class JobManager:
    """Submete callables a um pool de threads e rastreia o ciclo de vida.

    Cada job captura ``stdout``/``stderr`` do callable (o pipeline usa
    ``print`` extensivamente), além de status, tempos, resultado e erro.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[], dict], name: str = "") -> str:
        """Enfileira ``fn`` e retorna o ``run_id`` (uuid hex)."""
        run_id = uuid.uuid4().hex
        job = Job(run_id=run_id, name=name or run_id)
        with self._lock:
            self._jobs[run_id] = job
        self._executor.submit(self._run, run_id, fn)
        return run_id

    def get(self, run_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            raise KeyError(f"Run não encontrado: {run_id!r}")
        return job

    def list(self) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at)

    def cancel(self, run_id: str) -> bool:
        """Cancela apenas jobs ainda na fila; retorna ``False`` se já rodando."""
        with self._lock:
            job = self._jobs.get(run_id)
            if job is None:
                raise KeyError(f"Run não encontrado: {run_id!r}")
            if job.status != "queued":
                return False
            job.status = "cancelled"
            job.finished_at = time.time()
            return True

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    # -- internos ---------------------------------------------------------

    def _run(self, run_id: str, fn: Callable[[], dict]) -> None:
        with self._lock:
            job = self._jobs[run_id]
            if job.status == "cancelled":
                return
            job.status = "running"
            job.started_at = time.time()

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                result = fn()
            with self._lock:
                job.status = "completed"
                job.result = result
        except Exception as exc:  # noqa: BLE001 — registra qualquer falha
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                job.finished_at = time.time()
                job.logs = buf.getvalue()
