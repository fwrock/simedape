"""Execução de runs do pipeline em background, com rastreamento de status."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

_FINAL = ("completed", "failed", "cancelled")


# --------------------------------------------------------------------------
# Captura de stdout/stderr por thread
# --------------------------------------------------------------------------
# ``contextlib.redirect_stdout`` troca ``sys.stdout`` para o processo inteiro:
# com mais de um worker, os runs sobrescrevem o redirecionamento um do outro.
# Em vez disso, instalamos um roteador que escolhe o destino pela thread
# corrente; threads sem destino registrado continuam no stream original.

class LiveLog:
    """Buffer de texto thread-safe, legível enquanto o run ainda escreve nele.

    Mantém só os últimos ``max_chars`` caracteres: o pipeline imprime muito
    (``verbose`` do KShape) e um run longo não deve crescer sem limite.
    """

    def __init__(self, max_chars: int = 2_000_000) -> None:
        self._parts: list[str] = []
        self._size = 0
        self._max = max_chars
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        with self._lock:
            self._parts.append(text)
            self._size += len(text)
            if self._size > 2 * self._max:
                joined = "".join(self._parts)[-self._max:]
                self._parts, self._size = [joined], len(joined)
        return len(text)

    def flush(self) -> None:  # protocolo de arquivo
        pass

    def getvalue(self) -> str:
        with self._lock:
            return "".join(self._parts)[-self._max:]


class _ThreadRouter:
    def __init__(self, fallback) -> None:
        self.fallback = fallback

    def write(self, text):
        sink = _sinks.get(threading.get_ident())
        return (sink or self.fallback).write(text)

    def flush(self):
        sink = _sinks.get(threading.get_ident())
        (sink or self.fallback).flush()

    def __getattr__(self, name):
        return getattr(self.fallback, name)


_sinks: dict[int, LiveLog] = {}
_router_lock = threading.Lock()


def _capture_current_thread(sink: LiveLog) -> None:
    with _router_lock:
        if not isinstance(sys.stdout, _ThreadRouter):
            sys.stdout = _ThreadRouter(sys.stdout)
        if not isinstance(sys.stderr, _ThreadRouter):
            sys.stderr = _ThreadRouter(sys.stderr)
        _sinks[threading.get_ident()] = sink


def _release_current_thread() -> None:
    with _router_lock:
        _sinks.pop(threading.get_ident(), None)
        if not _sinks:  # último run terminou: devolve os streams originais
            for name in ("stdout", "stderr"):
                stream = getattr(sys, name)
                if isinstance(stream, _ThreadRouter):
                    setattr(sys, name, stream.fallback)


# --------------------------------------------------------------------------
# Job
# --------------------------------------------------------------------------

def _empty_progress() -> dict:
    return {
        "total": 0,
        "completed": 0,
        "percent": 0.0,
        "current_index": None,
        "current_stage": None,
        "stages": [],
    }


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
    # Descrição do run (fonte de dados, output_dir, estágios...) — definida na submissão.
    meta: dict = field(default_factory=dict)
    progress: dict = field(default_factory=_empty_progress)
    _live: LiveLog | None = field(default=None, repr=False, compare=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def on_progress(self, event: dict) -> None:
        """Atualiza ``progress`` a partir de um evento de ``Pipeline.run``."""
        kind = event.get("event")
        with self._lock:
            prog = self.progress
            if kind == "pipeline_start":
                prog["stages"] = [
                    {"name": n, "status": "pending", "started_at": None, "elapsed": None}
                    for n in event["stages"]
                ]
                prog["total"] = len(prog["stages"])
                return
            stages = prog["stages"]
            idx = event.get("index")
            if kind in ("stage_start", "stage_end", "stage_failed") and idx:
                entry = stages[idx - 1] if idx - 1 < len(stages) else None
                if entry is None:
                    return
                if kind == "stage_start":
                    entry.update(status="running", started_at=time.time())
                    prog["current_index"], prog["current_stage"] = idx, entry["name"]
                elif kind == "stage_end":
                    entry.update(status="completed", elapsed=event.get("elapsed"))
                    prog["completed"] = sum(s["status"] == "completed" for s in stages)
                else:
                    entry.update(
                        status="failed",
                        elapsed=event.get("elapsed"),
                        error=event.get("error"),
                    )
            if prog["total"]:
                prog["percent"] = round(100.0 * prog["completed"] / prog["total"], 1)

    def progress_dict(self) -> dict:
        with self._lock:
            prog = json.loads(json.dumps(self.progress))
            if self.status == "completed" and prog["total"]:
                prog["percent"] = 100.0
            return prog

    def to_dict(self) -> dict:
        with self._lock:
            live = self._live is not None and self.status == "running"
            logs = self._live.getvalue() if live else self.logs
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "logs": logs,
            "result": self.result,
            "error": self.error,
            "meta": self.meta,
            "progress": self.progress_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        return cls(
            run_id=data["run_id"],
            name=data.get("name", data["run_id"]),
            status=data.get("status", "failed"),
            created_at=data.get("created_at", time.time()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            logs=data.get("logs", ""),
            result=data.get("result"),
            error=data.get("error"),
            meta=data.get("meta") or {},
            progress=data.get("progress") or _empty_progress(),
        )


class JobManager:
    """Submete callables a um pool de threads e rastreia o ciclo de vida.

    Cada job captura ``stdout``/``stderr`` do callable (o pipeline usa
    ``print`` extensivamente) — por thread, e legível enquanto o run executa —
    além de status, progresso, tempos, resultado e erro.

    Parâmetros
    ----------
    state_dir : str, optional
        Se dado, cada run é gravado em ``<state_dir>/<run_id>.json`` a cada
        mudança de estado e recarregado ao iniciar. Runs que estavam
        ``queued``/``running`` num restart são marcados como ``failed``.
    """

    def __init__(self, max_workers: int = 2, state_dir: str | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self.state_dir = state_dir
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
            self._load_all()

    def submit(
        self,
        fn: Callable[..., dict],
        name: str = "",
        meta: dict | None = None,
        pass_job: bool = False,
    ) -> str:
        """Enfileira ``fn`` e retorna o ``run_id`` (uuid hex).

        Com ``pass_job=True``, ``fn`` recebe o ``Job`` (para reportar progresso
        via ``job.on_progress``); caso contrário é chamado sem argumentos.
        """
        run_id = uuid.uuid4().hex
        job = Job(run_id=run_id, name=name or run_id, meta=dict(meta or {}))
        with self._lock:
            self._jobs[run_id] = job
        self._persist(job)
        self._executor.submit(self._run, run_id, fn, pass_job)
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
        self._persist(job)
        return True

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    # -- internos ---------------------------------------------------------

    def _run(self, run_id: str, fn: Callable[..., dict], pass_job: bool) -> None:
        with self._lock:
            job = self._jobs[run_id]
            if job.status == "cancelled":
                return
            job.status = "running"
            job.started_at = time.time()
            job._live = LiveLog()
        self._persist(job)

        _capture_current_thread(job._live)
        try:
            result = fn(job) if pass_job else fn()
            with self._lock:
                job.status = "completed"
                job.result = result
        except Exception as exc:  # noqa: BLE001 — registra qualquer falha
            with self._lock:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
        finally:
            _release_current_thread()
            with self._lock:
                job.finished_at = time.time()
                job.logs = job._live.getvalue()
            self._persist(job)

    def _file(self, run_id: str) -> str:
        return os.path.join(self.state_dir, f"{run_id}.json")

    def _persist(self, job: Job) -> None:
        if not self.state_dir:
            return
        tmp = self._file(job.run_id) + ".tmp"
        try:
            with open(tmp, "w") as fh:
                json.dump(job.to_dict(), fh)
            os.replace(tmp, self._file(job.run_id))
        except OSError:
            pass  # persistência é best-effort: nunca derruba o run

    def _load_all(self) -> None:
        for fname in sorted(os.listdir(self.state_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.state_dir, fname)) as fh:
                    job = Job.from_dict(json.load(fh))
            except (OSError, ValueError, KeyError):
                continue
            if job.status not in _FINAL:
                job.status = "failed"
                job.error = "Interrompido: o servidor foi reiniciado durante o run."
                job.finished_at = job.finished_at or time.time()
                self._persist(job)
            self._jobs[job.run_id] = job
