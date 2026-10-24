"""Dados de gráficos para frontends, derivados dos mesmos JSONs dos relatórios PDF.

Cada gráfico é um dict independente de biblioteca de plotagem::

    {"id", "title", "type", "x": {...}, "y": {...}, "series": [...], "meta": {...}}

``type`` define o formato de ``series``:

* ``bar`` / ``line`` / ``radar`` — ``x.categories`` (rótulos) e, por série,
  ``data`` alinhado a elas (``null`` = sem valor). No radar, as categorias são os eixos.
* ``boxplot``  — ``x.categories`` (métodos); ``data`` = lista de
  ``{min, q1, median, q3, max, mean}`` alinhada às categorias.
* ``heatmap``  — ``x.categories`` (colunas), ``y.categories`` (linhas) e ``values``
  (matriz linhas × colunas); ``series`` vem vazio.
* ``scatter``  — ``data`` = lista de ``{x, y, label}``; ``meta.reference == "identity"``
  indica a diagonal y = x (estimado = real).

Nada é recalculado: só reorganiza o que ``compare``, ``sweep`` e ``clustering``
já gravaram em ``output_dir``. Valores não finitos viram ``null``.
"""
from __future__ import annotations

import math
import os

from pipeline.api import exports
from pipeline.api.exports import ArtifactNotFound, METRIC_LABELS, list_methods, valid_name

METRICS = list(METRIC_LABELS)

_CATALOG = (
    # id, título, tipo, fonte, parâmetros aceitos
    ("error_aggregate", "Erro percentual agregado (gmean)", "bar", "compare", ()),
    ("mape_by_method", "MAPE por método", "bar", "compare", ()),
    ("mape_radar", "MAPE por métrica", "radar", "compare", ()),
    ("coverage", "Cobertura: séries estimadas vs. reais", "bar", "compare", ()),
    ("error_distribution", "Distribuição de erros por série", "boxplot", "compare", ("metric",)),
    ("real_vs_estimated", "Real vs. estimado por série", "scatter", "estimation", ("method", "metric", "limit")),
    ("sweep_progression", "Progressão do MAPE por cenário parcial", "line", "sweep", ("metric", "methods")),
    ("sweep_heatmap", "Heatmap de MAPE (método × cenário)", "heatmap", "sweep", ("metric", "methods")),
    ("sweep_coverage", "Cobertura por cenário parcial", "line", "sweep", ("methods",)),
    ("cluster_sizes", "Séries por cluster", "bar", "clustering", ()),
    ("centroids", "Centróides dos clusters", "line", "clustering", ()),
)
CHART_IDS = tuple(c[0] for c in _CATALOG)


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def _check_metric(metric: str) -> str:
    if metric not in METRIC_LABELS:
        raise ValueError(f"Métrica inválida: {metric!r}. Válidas: {METRICS}")
    return metric


def _parse_methods(raw) -> list[str] | None:
    if not raw:
        return None
    names = [m.strip() for m in raw.split(",") if m.strip()]
    for name in names:
        if not valid_name(name):
            raise ValueError(f"Nome de método inválido: {name!r}")
    return names


def _chart(cid, title, ctype, **fields) -> dict:
    spec = next(c for c in _CATALOG if c[0] == cid)
    return {
        "id": cid, "title": title, "type": ctype, "source": spec[3],
        "x": fields.pop("x", {}), "y": fields.pop("y", {}),
        "series": fields.pop("series", []), "meta": fields.pop("meta", {}),
        **fields,
    }


# -- fontes -----------------------------------------------------------------

def _compare_results(output_dir, prefix) -> dict:
    methods = list_methods(output_dir, prefix, exclude=("summary",))
    if not methods:
        raise ArtifactNotFound(f"{prefix}_*")
    return {m: exports._read_json(output_dir, f"{prefix}_{m}") for m in methods}


def sweep_table(output_dir, prefix: str = "sweep", methods: list[str] | None = None) -> dict:
    """Tabela de progressão: cenários (por % crescente) × métodos, com MAPE/erro/cobertura."""
    data = exports._read_json(output_dir, f"{prefix}_summary")
    scenarios = sorted(data.get("scenarios", []), key=lambda s: s.get("percentage") or 0)
    if not scenarios:
        raise ArtifactNotFound(f"{prefix}_summary")
    found: list[str] = []
    for sc in scenarios:
        for m in sc.get("methods", {}):
            if m not in found:
                found.append(m)
    if methods is not None:
        unknown = [m for m in methods if m not in found]
        if unknown:
            raise ArtifactNotFound(f"método(s) fora do sweep: {unknown}")
        found = [m for m in found if m in methods]
    metrics = exports.ordered_metrics({
        k for sc in scenarios for md in sc.get("methods", {}).values() for k in md.get("mape", {})
    })

    def column(field, key=None):
        out = {}
        for m in found:
            row = []
            for sc in scenarios:
                md = sc.get("methods", {}).get(m)
                if md is None:
                    row.append(None)
                else:
                    row.append(_num(md.get(field) if key is None else (md.get(field) or {}).get(key)))
            out[m] = row
        return out

    return {
        "prefix": prefix,
        "scenarios": [{"label": s["label"], "percentage": s.get("percentage")} for s in scenarios],
        "methods": found,
        "metrics": [{"key": k, "label": _metric_label(k)} for k in metrics],
        "mape": {k: column("mape", k) for k in metrics},
        "error_pct": {k: column("error_pct", k) for k in metrics},
        "coverage_pct": column("coverage_pct"),
    }


# -- gráficos ---------------------------------------------------------------

def _compare_charts(cid, output_dir, prefix, params):
    results = _compare_results(output_dir, prefix)
    methods = list(results)
    metrics = exports.ordered_metrics({
        k for r in results.values() for k in (r.get("per_series_error") or {})
    }) or METRICS
    categories = [_metric_label(k) for k in metrics]

    def per_metric(getter):
        return [
            {"name": m, "data": [_num(getter(results[m], k)) for k in metrics]}
            for m in methods
        ]

    if cid == "error_aggregate":
        return _chart(cid, "Erro percentual agregado (gmean)", "bar",
                      x={"label": "Métrica", "categories": categories},
                      y={"label": "Erro (%)", "unit": "%"},
                      series=per_metric(lambda r, k: ((r.get("aggregate") or {}).get("error_pct") or {}).get(k)))
    if cid == "mape_by_method":
        return _chart(cid, "MAPE por método", "bar",
                      x={"label": "Métrica", "categories": categories},
                      y={"label": "MAPE (%)", "unit": "%"},
                      series=per_metric(lambda r, k: ((r.get("per_series_error") or {}).get(k) or {}).get("mape")))
    if cid == "mape_radar":
        return _chart(cid, "MAPE por métrica", "radar",
                      x={"label": "Métrica", "categories": categories},
                      y={"label": "MAPE (%)", "unit": "%"},
                      series=per_metric(lambda r, k: ((r.get("per_series_error") or {}).get(k) or {}).get("mape")))
    if cid == "coverage":
        return _chart(cid, "Cobertura: séries estimadas vs. reais", "bar",
                      x={"label": "Método", "categories": methods},
                      y={"label": "Cobertura (%)", "unit": "%"},
                      series=[{"name": "Cobertura", "data": [_num(results[m].get("coverage_pct")) for m in methods]}],
                      meta={"n_real": {m: results[m].get("n_real") for m in methods},
                            "n_estimated": {m: results[m].get("n_estimated") for m in methods}})
    # error_distribution
    metric = _check_metric(params.get("metric") or "average_speed")
    boxes = []
    for m in methods:
        st = (results[m].get("per_series_error") or {}).get(metric) or {}
        boxes.append({
            "min": _num(st.get("min")), "q1": _num(st.get("p25")), "median": _num(st.get("median")),
            "q3": _num(st.get("p75")), "max": _num(st.get("max")), "mean": _num(st.get("mean")),
        })
    return _chart(cid, f"Distribuição de erros — {_metric_label(metric)}", "boxplot",
                  x={"label": "Método", "categories": methods},
                  y={"label": "Erro por série (%)", "unit": "%"},
                  series=[{"name": _metric_label(metric), "data": boxes}],
                  meta={"metric": metric})


def _sweep_chart(cid, output_dir, prefix, params):
    table = sweep_table(output_dir, prefix, _parse_methods(params.get("methods")))
    labels = [s["label"] for s in table["scenarios"]]
    meta = {"percentages": [s["percentage"] for s in table["scenarios"]]}
    if cid == "sweep_coverage":
        return _chart(cid, "Cobertura por cenário parcial", "line",
                      x={"label": "Cenário parcial", "categories": labels},
                      y={"label": "Cobertura (%)", "unit": "%"},
                      series=[{"name": m, "data": table["coverage_pct"][m]} for m in table["methods"]],
                      meta=meta)
    metric = _check_metric(params.get("metric") or "average_speed")
    if metric not in table["mape"]:
        raise ArtifactNotFound(f"métrica {metric!r} ausente no sweep")
    if cid == "sweep_progression":
        return _chart(cid, f"Progressão do MAPE — {_metric_label(metric)}", "line",
                      x={"label": "Cenário parcial", "categories": labels},
                      y={"label": "MAPE (%)", "unit": "%"},
                      series=[{"name": m, "data": table["mape"][metric][m]} for m in table["methods"]],
                      meta={**meta, "metric": metric})
    return _chart(cid, f"Heatmap de MAPE — {_metric_label(metric)}", "heatmap",
                  x={"label": "Cenário parcial", "categories": labels},
                  y={"label": "Método", "categories": table["methods"], "unit": "%"},
                  values=[table["mape"][metric][m] for m in table["methods"]],
                  meta={**meta, "metric": metric})


def _clustering_chart(cid, output_dir, clustering_file):
    data = exports._read_json(output_dir, clustering_file)
    if cid == "cluster_sizes":
        sizes = data.get("time_series_by_cluster") or {}
        keys = sorted(sizes, key=lambda k: int(k) if str(k).isdigit() else k)
        return _chart(cid, "Séries por cluster", "bar",
                      x={"label": "Cluster", "categories": [str(k) for k in keys]},
                      y={"label": "Séries"},
                      series=[{"name": "Séries", "data": [sizes[k] for k in keys]}])
    centroids = data.get("centroids") or {}
    keys = sorted(centroids, key=lambda k: int(k) if str(k).isdigit() else k)
    width = max((len(centroids[k]) for k in keys), default=0)

    def flat(values):
        return [_num(v[0] if isinstance(v, list) and v else v) for v in values]

    return _chart(cid, "Centróides dos clusters", "line",
                  x={"label": "Ponto da série (normalizada)", "categories": list(range(width))},
                  y={"label": "Valor (z-score)"},
                  series=[{"name": f"cluster {k}", "data": flat(centroids[k])} for k in keys])


def _scatter_chart(output_dir, est_prefix, params):
    methods = list_methods(output_dir, est_prefix)
    if not methods:
        raise ArtifactNotFound(f"{est_prefix}_*")
    method = params.get("method") or methods[0]
    if not valid_name(method):
        raise ValueError(f"Método inválido: {method!r}")
    metric = _check_metric(params.get("metric") or "average_speed")
    try:
        limit = int(params.get("limit") or 500)
    except ValueError:
        raise ValueError("'limit' deve ser inteiro.") from None
    if limit < 1:
        raise ValueError("'limit' deve ser >= 1.")
    real = exports._read_json(output_dir, "new_simulation").get(metric) or []
    est = exports._read_json(output_dir, f"{est_prefix}_{method}").get(metric) or []
    try:
        labels = exports._read_json(output_dir, "time_series_labels")
    except ArtifactNotFound:
        labels = []
    n = min(len(real), len(est))
    step = max(1, math.ceil(n / limit))
    points = []
    for i in range(0, n, step):
        x, y = _num(real[i]), _num(est[i])
        if x is not None and y is not None:
            points.append({"x": x, "y": y, "label": labels[i] if i < len(labels) else str(i)})
    return _chart("real_vs_estimated", f"Real vs. estimado — {_metric_label(metric)} ({method})", "scatter",
                  x={"label": f"Real — {_metric_label(metric)}"},
                  y={"label": f"Estimado — {_metric_label(metric)}"},
                  series=[{"name": method, "data": points}],
                  meta={"reference": "identity", "metric": metric, "method": method,
                        "total_series": n, "returned": len(points), "step": step})


def build_chart(output_dir, chart_id, *, compare_prefix="compare", sweep_prefix="sweep",
                est_prefix="result", clustering_file="clustering", params=None) -> dict:
    """Monta um gráfico. ``ArtifactNotFound`` = dado ausente; ``ValueError`` = parâmetro inválido."""
    params = params or {}
    if chart_id not in CHART_IDS:
        raise ArtifactNotFound(chart_id)
    if not output_dir:
        raise ArtifactNotFound(chart_id)
    for prefix in (compare_prefix, sweep_prefix, est_prefix, clustering_file):
        if not valid_name(prefix):
            raise ValueError("Prefixo/arquivo inválido.")
    source = next(c[3] for c in _CATALOG if c[0] == chart_id)
    if source == "compare":
        return _compare_charts(chart_id, output_dir, compare_prefix, params)
    if source == "sweep":
        return _sweep_chart(chart_id, output_dir, sweep_prefix, params)
    if source == "clustering":
        return _clustering_chart(chart_id, output_dir, clustering_file)
    return _scatter_chart(output_dir, est_prefix, params)


def catalog(output_dir, *, compare_prefix="compare", sweep_prefix="sweep",
            est_prefix="result", clustering_file="clustering") -> list[dict]:
    """Gráficos disponíveis para o ``output_dir`` (``available`` indica se os dados existem)."""
    for name in (compare_prefix, sweep_prefix, est_prefix, clustering_file):
        if not valid_name(name):
            raise ValueError("Prefixo/arquivo inválido.")

    def has(stem):
        return bool(output_dir) and os.path.isfile(os.path.join(output_dir, f"{stem}.json"))

    ready = {
        "compare": bool(list_methods(output_dir, compare_prefix, exclude=("summary",))),
        "estimation": bool(list_methods(output_dir, est_prefix)) and has("new_simulation"),
        "sweep": has(f"{sweep_prefix}_summary"),
        "clustering": has(clustering_file),
    }
    return [
        {
            "id": cid, "title": title, "type": ctype, "source": source,
            "params": list(params), "available": ready[source],
            "metrics": METRICS if "metric" in params else None,
            "url": f"charts/{cid}",
        }
        for cid, title, ctype, source, params in _CATALOG
    ]
