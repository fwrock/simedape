"""Leitura dos artefatos de um run (JSON/PDF em ``output_dir``) para exportação.

Os estágios gravam seus resultados como arquivos em ``output_dir``; nada aqui
recalcula algo. Os helpers são independentes da fonte de dados (CSV ou
ClickHouse): depois de ``data_load`` os artefatos têm o mesmo formato.

Todo nome vindo do cliente (método, prefixo) é validado contra ``_NAME_RE`` e
resolvido dentro de ``output_dir`` — nenhum caminho do cliente é usado direto.
"""
from __future__ import annotations

import json
import math
import os
import re

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# Rótulos das métricas de ``stages/compare.py`` (não importado aqui para manter
# a API leve; métricas desconhecidas aparecem com o próprio nome).
METRIC_LABELS = {
    "average_speed": "Velocidade Média (m/s)",
    "percentage_of_track": "Ocupação (%)",
    "average_vehicles": "Veículos Médios",
    "km_traveled": "Km Trafegados",
}

_KIND_BY_PREFIX = (
    ("new_simulation", "real_metrics"),
    ("simulation_points", "simulation_points"),
    ("dataset_cluster_", "cluster_dataset"),
    ("selective_instances", "selective_instantiation"),
    ("time_series_labels", "labels"),
    ("compare_", "comparison"),
    ("sweep_", "sweep"),
    ("result_", "estimation"),
)


class ArtifactNotFound(KeyError):
    pass


def valid_name(name: str) -> bool:
    return bool(name) and bool(_NAME_RE.match(name)) and ".." not in name


def sanitize(obj):
    """Troca NaN/Infinity por ``None``: o JSON estrito não os aceita.

    ``save_json`` grava ``NaN`` literal (ex.: erro % com baseline zero).
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


def classify(name: str, clustering_file: str = "clustering") -> str:
    if name.endswith(".pdf"):
        return "report"
    stem = name[:-5] if name.endswith(".json") else name
    if stem == clustering_file:
        return "clustering"
    for prefix, kind in _KIND_BY_PREFIX:
        if stem.startswith(prefix):
            return kind
    return "other"


def list_artifacts(output_dir: str | None, clustering_file: str = "clustering") -> list[dict]:
    """Arquivos JSON/PDF de ``output_dir`` e de ``output_dir/plots``."""
    if not output_dir or not os.path.isdir(output_dir):
        return []
    found = []
    for sub in ("", "plots"):
        folder = os.path.join(output_dir, sub)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            if not os.path.isfile(path) or not fname.endswith((".json", ".pdf")):
                continue
            stat = os.stat(path)
            found.append({
                "name": f"{sub}/{fname}" if sub else fname,
                "kind": classify(fname, clustering_file),
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            })
    return found


def _read_json(output_dir: str | None, stem: str) -> dict:
    if not output_dir or not valid_name(stem):
        raise ArtifactNotFound(stem)
    path = os.path.join(output_dir, f"{stem}.json")
    if not os.path.isfile(path):
        raise ArtifactNotFound(stem)
    with open(path) as fh:
        return sanitize(json.load(fh))


def list_methods(output_dir: str | None, prefix: str, exclude: tuple = ()) -> list[str]:
    """Métodos com arquivo ``<prefix>_<método>.json`` em ``output_dir``."""
    if not valid_name(prefix) or not output_dir or not os.path.isdir(output_dir):
        return []
    head = f"{prefix}_"
    methods = []
    for fname in sorted(os.listdir(output_dir)):
        if fname.startswith(head) and fname.endswith(".json"):
            method = fname[len(head):-len(".json")]
            if method and method not in exclude:
                methods.append(method)
    return methods


# -- estimativas ------------------------------------------------------------

def _stats(values) -> dict:
    nums = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not nums:
        return {"count": len(values), "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(nums),
        "max": max(nums),
        "mean": sum(nums) / len(nums),
    }


def estimation_overview(output_dir: str | None, prefix: str = "result") -> dict:
    """Um resumo por método (séries e min/média/max por métrica), sem os vetores."""
    overview = {}
    for method in list_methods(output_dir, prefix):
        data = _read_json(output_dir, f"{prefix}_{method}")
        overview[method] = {
            "file": f"{prefix}_{method}.json",
            "metrics": {
                key: _stats(value)
                for key, value in data.items()
                if isinstance(value, list)
            },
            "scalars": {
                key: value for key, value in data.items() if not isinstance(value, (list, dict))
            },
        }
    return {"prefix": prefix, "methods": overview}


def estimation_detail(
    output_dir: str | None,
    method: str,
    prefix: str = "result",
    metric: str | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    """Resultado completo de um método; ``metric``/``offset``/``limit`` recortam as séries."""
    data = _read_json(output_dir, f"{prefix}_{method}")
    total = None
    out = {}
    for key, value in data.items():
        if metric and isinstance(value, list) and key != metric:
            continue
        if isinstance(value, list):
            total = len(value) if total is None else max(total, len(value))
            end = None if limit is None else offset + limit
            value = value[offset:end]
        out[key] = value
    return {
        "method": method,
        "prefix": prefix,
        "offset": offset,
        "limit": limit,
        "total_series": total,
        "data": out,
    }


# -- comparação -------------------------------------------------------------

def comparison_matrix(output_dir: str | None, prefix: str = "compare") -> dict:
    """Matriz método × métrica (MAPE e erro agregado) e cobertura por método."""
    methods = list_methods(output_dir, prefix, exclude=("summary",))
    results = {m: _read_json(output_dir, f"{prefix}_{m}") for m in methods}
    metrics = sorted({
        k for r in results.values() for k in (r.get("per_series_error") or {})
    })
    return {
        "prefix": prefix,
        "methods": methods,
        "metrics": [{"key": k, "label": METRIC_LABELS.get(k, k)} for k in metrics],
        "mape": {
            k: {m: results[m]["per_series_error"].get(k, {}).get("mape") for m in methods}
            for k in metrics
        },
        "aggregate_error_pct": {
            k: {m: (results[m].get("aggregate") or {}).get("error_pct", {}).get(k) for m in methods}
            for k in metrics
        },
        "coverage_pct": {m: results[m].get("coverage_pct") for m in methods},
        "n_real": {m: results[m].get("n_real") for m in methods},
        "n_estimated": {m: results[m].get("n_estimated") for m in methods},
    }


def comparison_detail(output_dir: str | None, method: str, prefix: str = "compare") -> dict:
    return _read_json(output_dir, f"{prefix}_{method}")
