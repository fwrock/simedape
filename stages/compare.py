"""
CompareStage — compara métricas estimadas com o cenário real (100%).

Produz relatórios detalhados com:
  - Erro percentual agregado (gmean) por métrica
  - MAPE e distribuição de erros por série (min, max, mediana, desvio padrão)
  - Cobertura (séries estimadas vs séries reais)
  - Tabela multi-método side-by-side no terminal
  - JSON estruturado por método + summary consolidado

Fluxo esperado:
    ScenarioMetricsStage  →  new_simulation.json   (100% real)
    EstimationStage       →  result_avg.json        (estimado)
    CompareStage          →  compare_avg.json + compare_summary.json
"""
import os
import sys
import numpy as np
from scipy.stats import gmean

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.writer import load_json, save_json, ensure_dir


_METRICS = ["average_speed", "percentage_of_track", "average_vehicles", "km_traveled"]
_METRIC_LABELS = {
    "average_speed": "Velocidade Média (m/s)",
    "percentage_of_track": "Ocupação (%)",
    "average_vehicles": "Veículos Médios",
    "km_traveled": "Km Trafegados",
}


def _safe_gmean(values):
    arr = [float(v) for v in values if v is not None and float(v) > 0]
    if not arr:
        return float("nan")
    return float(gmean(arr))


def _pct_error(real, est):
    if real == 0 or real != real:  # nan check
        return float("nan")
    return 100.0 * (est - real) / real


def _series_errors(real_list, est_list):
    """Calcula erro percentual por série (pareado por índice)."""
    n = min(len(real_list), len(est_list))
    errors = []
    for i in range(n):
        r, e = float(real_list[i]), float(est_list[i])
        if r > 0:
            errors.append(100.0 * (e - r) / r)
    return errors


def _dist_stats(values):
    """Estatísticas de distribuição de uma lista de erros."""
    if not values:
        return {"min": float("nan"), "max": float("nan"),
                "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "mape": float("nan")}
    arr = np.array(values)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "mape": float(np.mean(np.abs(arr))),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def _analyse_method(real, estimated, label):
    """Produz análise completa para um par real/estimado."""
    n_real = len(real.get("average_speed", []))
    n_est = len(estimated.get("average_speed", []))
    coverage = 100.0 * n_est / n_real if n_real > 0 else 0.0

    real_agg = {m: _safe_gmean(real[m]) for m in _METRICS}
    est_agg = {m: _safe_gmean(estimated[m]) for m in _METRICS}
    agg_error = {m: _pct_error(real_agg[m], est_agg[m]) for m in _METRICS}

    per_series = {}
    for m in _METRICS:
        errors = _series_errors(real.get(m, []), estimated.get(m, []))
        per_series[m] = _dist_stats(errors)

    return {
        "label": label,
        "n_real": n_real,
        "n_estimated": n_est,
        "coverage_pct": round(coverage, 2),
        "aggregate": {
            "real": real_agg,
            "estimated": est_agg,
            "error_pct": agg_error,
        },
        "per_series_error": per_series,
    }


def _print_method_report(result):
    label = result["label"]
    agg = result["aggregate"]
    ps = result["per_series_error"]

    print(f"\n  {'─'*72}")
    print(f"  Método: {label}   "
          f"(séries: real={result['n_real']}, estimado={result['n_estimated']}, "
          f"cobertura={result['coverage_pct']:.1f}%)")
    print(f"  {'─'*72}")
    print(f"  {'Métrica':<26} {'Real':>10} {'Estim.':>10} {'Err%':>8} "
          f"{'MAPE':>8} {'Mediana':>9} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*72}")
    for m in _METRICS:
        lbl = _METRIC_LABELS[m]
        r = agg["real"][m]
        e = agg["estimated"][m]
        err = agg["error_pct"][m]
        mape = ps[m]["mape"]
        med = ps[m]["median"]
        mn = ps[m]["min"]
        mx = ps[m]["max"]
        print(f"  {lbl:<26} {r:>10.4f} {e:>10.4f} {err:>+8.2f}% "
              f"{mape:>7.2f}% {med:>+8.2f}% {mn:>+7.2f}% {mx:>+7.2f}%")


def _print_summary_table(all_results):
    """Tabela comparativa de todos os métodos para cada métrica."""
    methods = list(all_results.keys())
    print(f"\n  {'='*72}")
    print(f"  RESUMO COMPARATIVO — MAPE por método")
    print(f"  {'='*72}")
    print(f"  {'Métrica':<26}", end="")
    for m in methods:
        print(f"  {m:>10}", end="")
    print()
    print(f"  {'─'*72}")
    for metric in _METRICS:
        print(f"  {_METRIC_LABELS[metric]:<26}", end="")
        for m in methods:
            mape = all_results[m]["per_series_error"][metric]["mape"]
            print(f"  {mape:>9.2f}%", end="")
        print()
    print(f"  {'─'*72}")
    print(f"  {'Cobertura':<26}", end="")
    for m in methods:
        cov = all_results[m]["coverage_pct"]
        print(f"  {cov:>9.1f}%", end="")
    print()


class CompareStage(Stage):
    """Compara métricas estimadas com o cenário real (100%) e gera relatórios.

    Parâmetros
    ----------
    real_file : str
        Nome do JSON com métricas reais (sem extensão). Default: 'new_simulation'.
    estimated_files : list[str]
        Nomes dos JSONs com métricas estimadas (sem extensão).
        Ex: ['result_avg', 'result_sample', 'result_wavg', 'result_base'].
    output_prefix : str
        Prefixo dos arquivos de saída. Ex: 'compare' → 'compare_avg.json'.
    """

    name = "compare"

    def __init__(
        self,
        real_file: str = "new_simulation",
        estimated_files: list = None,
        output_prefix: str = "compare",
        generate_plots: bool = True,
    ):
        self.real_file = real_file
        self.estimated_files = estimated_files or ["result_avg"]
        self.output_prefix = output_prefix
        self.generate_plots = generate_plots

    def execute(self, ctx: PipelineContext) -> None:
        ensure_dir(ctx.output_dir)

        real_path = os.path.join(ctx.output_dir, self.real_file)
        try:
            real = load_json(real_path)
        except FileNotFoundError:
            raise RuntimeError(
                f"Arquivo de métricas reais não encontrado: '{real_path}.json'\n"
                "Execute ScenarioMetricsStage primeiro."
            )

        real_agg = {m: _safe_gmean(real[m]) for m in _METRICS}
        print(f"\n[{self.name}] Cenário real ({self.real_file}): "
              f"{len(real.get('average_speed', []))} séries")
        print(f"  Velocidade Média (gmean): {real_agg['average_speed']:.6f} m/s")
        print(f"  Ocupação       (gmean): {real_agg['percentage_of_track']:.6f} %")
        print(f"  Veículos       (gmean): {real_agg['average_vehicles']:.6f}")
        print(f"  Km Trafegados  (gmean): {real_agg['km_traveled']:.6f} km")

        all_results = {}

        for est_file in self.estimated_files:
            est_path = os.path.join(ctx.output_dir, est_file)
            try:
                estimated = load_json(est_path)
            except FileNotFoundError:
                print(f"\n[{self.name}] Arquivo não encontrado, pulando: '{est_path}.json'")
                continue

            label = est_file.removeprefix("result_")
            result = _analyse_method(real, estimated, label)
            all_results[label] = result

            _print_method_report(result)

            out_path = os.path.join(ctx.output_dir, f"{self.output_prefix}_{label}")
            save_json(out_path, result)
            print(f"  → Salvo: {out_path}.json")

        if len(all_results) > 1:
            _print_summary_table(all_results)

        if all_results:
            summary_path = os.path.join(ctx.output_dir, f"{self.output_prefix}_summary")
            save_json(summary_path, all_results)
            print(f"\n[{self.name}] Resumo consolidado: {summary_path}.json")

        if self.generate_plots and all_results:
            from pipeline.stages.plots import save_individual_pdfs, save_consolidated_pdf, save_pca_pdf
            plots_dir = os.path.join(ctx.output_dir, "plots")
            scenario = ctx.config.scenario_name
            title = f" — {scenario}" if scenario else ""
            pca_models = getattr(ctx, "pca_models", None)
            print(f"\n[{self.name}] Gerando gráficos em: {plots_dir}")
            save_individual_pdfs(all_results, plots_dir, self.output_prefix, title, pca_models)
            save_consolidated_pdf(all_results, plots_dir, self.output_prefix, title, pca_models)
            if pca_models:
                save_pca_pdf(pca_models, plots_dir, self.output_prefix, title)
            print(f"[{self.name}] Gráficos gerados.")
