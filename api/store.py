"""Armazenamento de configurações nomeadas do pipeline."""
from __future__ import annotations

import json
import os
import re

from pipeline.config import SimEDaPEConfig

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigStore:
    """CRUD de configurações nomeadas, com persistência opcional em JSON.

    Parâmetros
    ----------
    storage_dir : str, optional
        Diretório onde cada config é salva como ``<name>.json``. Quando
        ``None``, o armazenamento é apenas em memória.
    """

    def __init__(self, storage_dir: str | None = None) -> None:
        self._configs: dict[str, SimEDaPEConfig] = {}
        self.storage_dir = storage_dir
        if storage_dir:
            os.makedirs(storage_dir, exist_ok=True)
            self._load_all()

    def save(self, name: str, config: SimEDaPEConfig) -> SimEDaPEConfig:
        """Cria ou substitui uma config pelo nome (retorna a config salva)."""
        self._validate_name(name)
        self._configs[name] = config
        if self.storage_dir:
            with open(self._file(name), "w") as fh:
                json.dump(config.to_dict(), fh, indent=2)
        return config

    def get(self, name: str) -> SimEDaPEConfig:
        try:
            return self._configs[name]
        except KeyError:
            raise KeyError(f"Configuração não encontrada: {name!r}") from None

    def list(self) -> list[str]:
        return sorted(self._configs)

    def delete(self, name: str) -> bool:
        if name not in self._configs:
            return False
        del self._configs[name]
        if self.storage_dir:
            path = self._file(name)
            if os.path.exists(path):
                os.remove(path)
        return True

    # -- internos ---------------------------------------------------------

    def _file(self, name: str) -> str:
        return os.path.join(self.storage_dir, f"{name}.json")

    def _load_all(self) -> None:
        for filename in sorted(os.listdir(self.storage_dir)):
            if not filename.endswith(".json"):
                continue
            name = filename[:-5]
            with open(os.path.join(self.storage_dir, filename)) as fh:
                data = json.load(fh)
            self._configs[name] = SimEDaPEConfig.from_dict(data)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ValueError(
                f"Nome de config inválido: {name!r} (use [A-Za-z0-9._-])."
            )
