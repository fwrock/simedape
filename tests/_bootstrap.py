"""Test bootstrap: expõe o diretório ``simedape/`` como o pacote ``pipeline``.

O projeto mapeia o próprio diretório ``simedape/`` para o pacote Python chamado
``pipeline`` (ver ``pyproject.toml``, ``[tool.setuptools.package-dir]``), e todos
os imports internos usam ``from pipeline.X import Y``. Reproduzimos esse
mapeamento aqui SEM executar o ``__init__`` do pacote (``simedape/__init__.py``),
que puxa dependências pesadas/opcionais (tslearn, sklearn, ...) de que os
módulos leves ``pipeline.io`` / ``pipeline.config`` não precisam.
"""
import os
import sys
import types

_SIMEDAPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _SIMEDAPE_DIR not in sys.path:
    sys.path.insert(0, _SIMEDAPE_DIR)

if "pipeline" not in sys.modules:
    _pkg = types.ModuleType("pipeline")
    _pkg.__path__ = [_SIMEDAPE_DIR]
    _pkg.__package__ = "pipeline"
    sys.modules["pipeline"] = _pkg
