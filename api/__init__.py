"""API HTTP do SimEDaPE (Flask).

Gerencia configurações, lista estágios e executa runs do pipeline em background.
Os componentes que dependem do Flask são importados sob demanda, para que os
módulos leves (``store``/``jobs``) possam ser usados sem Flask instalado.
"""
from __future__ import annotations

__all__ = ["create_app", "JobManager", "Job", "ConfigStore"]


def __getattr__(name: str):
    if name == "create_app":
        from pipeline.api.app import create_app
        return create_app
    if name == "JobManager":
        from pipeline.api.jobs import JobManager
        return JobManager
    if name == "Job":
        from pipeline.api.jobs import Job
        return Job
    if name == "ConfigStore":
        from pipeline.api.store import ConfigStore
        return ConfigStore
    raise AttributeError(name)
