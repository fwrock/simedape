"""
SweepStage — executa estimação e comparação para múltiplos cenários parciais.

Permite comparar como o erro evolui à medida que mais dados do cenário
parcial estão disponíveis (ex: 5%, 10%, 20%, 50%).

Fluxo esperado:
    DataLoadStage → ProcessDatasetStage → ClusteringStage
    → SimulationPointsStage → WarpingPathsStage
    → PCATrainStage (opcional, se include_pca=True)
    → ScenarioMetricsStage (salva new_simulation.json)
    → SweepStage

Exemplo de configuração no YAML:

    - name: sweep
      params:
        real_file: new_simulation
        methods: [avg, base, sample, wavg]
        include_pca: true
        output_prefix: sweep
        generate_plots: true
        scenarios:
          - label: "5%"
            dir: input/carla-experiments/experiments/5p/
            percentage: 5.0
          - label: "10%"
            dir: input/carla-experiments/experiments/10p/
            percentage: 10.0
          - label: "50%"
            dir: input/carla-experiments/experiments/50p/
            percentage: 50.0
"""

import os
import time
import numpy as np

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.reader import read_single_scenario
from pipeline.io.writer import save_json, load_json, ensure_dir
from pipeline.domain.processor import process_dataset, pre_processing_estimation
from pipeline.stages.simulation_points import ensure_clustering_loaded
from pipeline.stages.estimation import (
    run_estimation_avg, run_estimation_base,
    run_estimation_sample, run_estimation_wavg,
)
from pipeline.stages.compare import _analyse_method, _print_method_report, _METRICS


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _run_dtw_methods(methods, ctx, second_dir, percentage, wps_folder, interp_method=None):
    """Carrega o cenário parcial e roda os métodos DTW indicados.

    Retorna dict {method: metrics_dict}.

    interp_method : str ou None
        Se None, usa TimeSeriesResampler padrão (linear via tslearn).
        Se definido, usa _stretch_to_length com o método especificado para
        reamostrar as séries parciais antes do warping.
    """
    from pipeline.io.reader import read_single_scenario
    from pipeline.domain.processor import process_dataset, pre_processing_estimation, process_dataset_interp

    cfg = ctx.config
    clustering = ctx.clustering_result

    _, ss2, _, _, labels_2, _ = read_single_scenario(
        second_dir, sep=cfg.csv_sep, intervals=cfg.intervals.as_dict()
    )
    centroid_size = len(clustering["centroids"]["0"])
    if interp_method is None:
        X_2, mean_2, std_2 = process_dataset(ss2, centroid_size)
    else:
        X_2, mean_2, std_2 = process_dataset_interp(ss2, centroid_size, interp_method=interp_method)

    ts_labels_first = ctx.time_series_labels
    ts_indexs_first, ts_dict_first = pre_processing_estimation(ts_labels_first)
    _, ts_dict_second = pre_processing_estimation(labels_2)

    mean = np.array(clustering["mean"])
    std = np.array(clustering["standard_deviation"])
    sp_indexs = ctx.simulation_points_indexs

    results = {}
    for method in methods:
        if method == "avg":
            results[method] = run_estimation_avg(
                ctx.streets_info, clustering, mean, std, mean_2, std_2, X_2,
                ts_indexs_first, ts_labels_first, ts_dict_first, ts_dict_second,
                sp_indexs, wps_folder,
            )
        elif method == "sample":
            results[method] = run_estimation_sample(
                ctx.streets_info, clustering, mean, std, mean_2, std_2, X_2,
                ts_indexs_first, ts_labels_first, ts_dict_first, ts_dict_second,
                sp_indexs, wps_folder,
            )
        elif method == "wavg":
            results[method] = run_estimation_wavg(
                ctx.streets_info, clustering, mean, std, mean_2, std_2, X_2,
                ts_indexs_first, ts_labels_first, ts_dict_first, ts_dict_second,
                sp_indexs, wps_folder,
                percentage_sample=percentage,
            )
        elif method == "base":
            from pipeline.stages.estimation import run_estimation_base
            results[method] = run_estimation_base(
                ctx.streets_info, clustering, mean, std, mean_2, std_2, X_2,
                ts_indexs_first, ts_labels_first, ts_dict_first, ts_dict_second,
                sp_indexs, wps_folder,
            )
        else:
            print(f"    [sweep] Método DTW desconhecido ignorado: {method}")
    return results


# Métodos de interpolação disponíveis para os modos 'sp_stretched' e 'stretched'
INTERP_METHODS = ["linear", "poly", "lagrange", "newton", "vandermonde", "spline"]


def _run_pca_method(ctx, second_dir, use_sp=None, mode="sp", interp_method=None, use_eig=False):
    """Carrega o cenário parcial e roda estimação PCA. Retorna metrics_dict.

    mode='sp'       : SP do cluster → lstsq com loadings truncados → W → reconstrói cluster
    mode='ind'      : por série → lstsq com loadings truncados (sem interp)
    mode='stretched': por série → interpola parcial → comprimento completo → projeção PCA
    mode='sp'  + interp_method : SP esticado com interp → projeção completa → W → cluster

    interp_method : str ou None
        Se None, modos sp/ind usam truncated lstsq.
        Se definido ('linear','poly','lagrange','newton','vandermonde','spline'),
        sp usa _weights_from_sp_stretched e stretched passa o método para
        _reconstruct_stretched.
    """
    # backward compat
    if use_sp is not None:
        mode = "sp" if use_sp else "ind"

    from pipeline.io.reader import read_single_scenario
    from pipeline.domain.processor import process_dataset, pre_processing_estimation
    from pipeline.domain.metrics import calc_metrics
    from pipeline.stages.pca import (
        _denormalize, _assign_cluster, _reconstruct, _reconstruct_stretched,
        _pruning, _weights_from_sp, _weights_from_sp_stretched,
        _reconstruct_from_weights, _apply_eigenvalue_filter,
        _ensure_train_weights,
    )

    cfg = ctx.config
    clustering = ctx.clustering_result
    n_clusters = clustering["number_of_clusters"]
    clusters_array = np.array(clustering["clusters"])

    pca_models = getattr(ctx, "pca_models", None)
    if pca_models is None:
        raise RuntimeError(
            "Modelos PCA não encontrados no contexto.\n"
            "Execute PCATrainStage antes de SweepStage com include_pca=True."
        )

    max_size_2, ss2, _, _, labels_2, _ = read_single_scenario(
        second_dir, sep=cfg.csv_sep, intervals=cfg.intervals.as_dict()
    )
    centroid_size = len(clustering["centroids"]["0"])

    # Para sp sem interp: reamostrar para centroid_size (séries comparáveis ao SP)
    # Para sp+interp / ind / stretched: usar tamanho natural (parcialidade preservada)
    if mode == "sp" and interp_method is None:
        X_2, mean_2, std_2 = process_dataset(ss2, centroid_size)
    else:
        X_2, mean_2, std_2 = process_dataset(ss2, max_size_2)

    X_2_denorm = _denormalize(X_2, mean_2, std_2)
    _, labels_2_dict = pre_processing_estimation(labels_2)

    metrics = {
        "average_speed": [],
        "percentage_of_track": [],
        "average_vehicles": [],
        "km_traveled": [],
        "missing_time_series": 0,
    }

    ts_full_labels = ctx.time_series_labels

    if mode == "sp":
        # ── Modo SP ──
        ts_indexs_first, _ = pre_processing_estimation(ts_full_labels)
        sp_indexs = ctx.simulation_points_indexs

        # Pré-calcula W para cada cluster a partir do SP parcial
        cluster_weights = {}
        for i in range(n_clusters):
            if i not in pca_models:
                continue
            sp_label = ts_full_labels[sp_indexs[i]]
            if sp_label not in labels_2_dict:
                continue
            sp_partial = X_2_denorm[labels_2_dict[sp_label]]
            if interp_method is None:
                weights, P_complete, complete_mean = _weights_from_sp(
                    sp_partial, pca_models[i]
                )
            else:
                weights, P_complete, complete_mean = _weights_from_sp_stretched(
                    sp_partial, pca_models[i], interp_method=interp_method
                )
            if use_eig:
                weights = _apply_eigenvalue_filter(weights, pca_models[i])
            cluster_weights[i] = (weights, P_complete, complete_mean)

        # Reconstrói todas as séries do cenário completo
        for k, label in enumerate(ts_full_labels):
            link_id = label.split("-")[0]
            if link_id not in (ctx.streets_info or {}):
                metrics["missing_time_series"] += 1
                continue
            cluster_id = int(clusters_array[k])
            if cluster_id not in cluster_weights:
                metrics["missing_time_series"] += 1
                continue
            weights, P_complete, complete_mean = cluster_weights[cluster_id]
            estimated = _reconstruct_from_weights(weights, P_complete, complete_mean, centroid_size)
            estimated = _pruning(estimated)
            calc = calc_metrics(estimated, link_id, ctx.streets_info, printer=False)
            metrics["average_speed"].append(calc["average_speed"])
            metrics["percentage_of_track"].append(calc["percentage_of_track"])
            metrics["average_vehicles"].append(calc["average_vehicles"])
            metrics["km_traveled"].append(calc["km_traveled"])

    elif mode == "stretched":
        # ── Modo stretched: interpola parcial → comprimento completo → PCA ──
        cluster_centers_raw = clustering["centroids"]
        cluster_centers_str = [
            np.array(cluster_centers_raw[str(k)]).ravel()
            for k in sorted(int(k) for k in cluster_centers_raw)
        ]
        for label in ts_full_labels:
            if label not in labels_2_dict:
                metrics["missing_time_series"] += 1
                continue
            idx_2 = labels_2_dict[label]
            partial_series = X_2_denorm[idx_2]
            cluster_id = _assign_cluster(partial_series, cluster_centers_str)
            if cluster_id not in pca_models:
                metrics["missing_time_series"] += 1
                continue
            reconstructed = _reconstruct_stretched(
                partial_series, pca_models[cluster_id], centroid_size,
                interp_method=interp_method or "linear",
                use_eig=use_eig,
            )
            reconstructed = _pruning(reconstructed)
            link_id = label.split("-")[0]
            calc = calc_metrics(reconstructed, link_id, ctx.streets_info, printer=False)
            metrics["average_speed"].append(calc["average_speed"])
            metrics["percentage_of_track"].append(calc["percentage_of_track"])
            metrics["average_vehicles"].append(calc["average_vehicles"])
            metrics["km_traveled"].append(calc["km_traveled"])

    else:
        # ── Modo individual (ind) ──
        _cr = clustering["centroids"]
        cluster_centers = [np.array(_cr[str(k)]) for k in sorted(int(k) for k in _cr)]
        for label in ts_full_labels:
            if label not in labels_2_dict:
                metrics["missing_time_series"] += 1
                continue
            idx_2 = labels_2_dict[label]
            partial_series = X_2_denorm[idx_2]
            cluster_id = _assign_cluster(partial_series, cluster_centers)
            if cluster_id not in pca_models:
                metrics["missing_time_series"] += 1
                continue
            reconstructed = _reconstruct(partial_series, pca_models[cluster_id], centroid_size,
                                          use_eig=use_eig)
            reconstructed = _pruning(reconstructed)
            link_id = label.split("-")[0]
            calc = calc_metrics(reconstructed, link_id, ctx.streets_info, printer=False)
            metrics["average_speed"].append(calc["average_speed"])
            metrics["percentage_of_track"].append(calc["percentage_of_track"])
            metrics["average_vehicles"].append(calc["average_vehicles"])
            metrics["km_traveled"].append(calc["km_traveled"])

    return metrics


# ---------------------------------------------------------------------------
# Delta modes — análogos ao warping path mas no espaço PCA
# ---------------------------------------------------------------------------

def _run_pca_delta_method(ctx, second_dir, delta_mode, interp_method=None,
                           use_eig=False, percentage=100.0):
    """Estimação PCA via delta de pesos: usa pesos do treino + deslocamento do SP.

    Análogo direto ao DTW:
      - Warping path  ↔  w_train_i  (padrão pré-aprendido por série)
      - SP parcial    ↔  Δw = w_SP_new − w_SP_train  (deslocamento do SP no novo cenário)

    delta_mode
    ----------
    'base'  : ŵᵢ = wᵢᵗʳᵃⁱⁿ                   (pesos do treino — sem dados novos)
    'delta' : ŵᵢ = wᵢᵗʳᵃⁱⁿ + Δw               (delta completo)
    'avg'   : ŵᵢ = wᵢᵗʳᵃⁱⁿ + Δw/2             (média igualitária)
    'wavg'  : ŵᵢ = wᵢᵗʳᵃⁱⁿ + (p/100)·Δw      (ponderado pelo percentual)

    interp_method : str ou None
        Como calcular w_SP_new a partir da série parcial do SP.
        None → lstsq com loadings truncados  (como pca_sp)
        str  → interpola SP ao comprimento completo antes de projetar  (como pca_sp_<interp>)
        Ignorado para delta_mode='base' (não usa SP do novo cenário).

    use_eig : bool
        Se True, aplica filtro de Wiener espectral nos pesos finais.
    """
    from pipeline.io.reader import read_single_scenario
    from pipeline.domain.processor import process_dataset, pre_processing_estimation
    from pipeline.domain.metrics import calc_metrics
    from pipeline.stages.pca import (
        _denormalize, _pruning, _weights_from_sp, _weights_from_sp_stretched,
        _reconstruct_delta, _ensure_train_weights,
    )

    cfg = ctx.config
    clustering = ctx.clustering_result
    n_clusters = clustering["number_of_clusters"]
    clusters_array = np.array(clustering["clusters"])
    centroid_size = len(clustering["centroids"]["0"])

    pca_models = getattr(ctx, "pca_models", None)
    if pca_models is None:
        raise RuntimeError("Modelos PCA não encontrados no contexto.")

    # Garante que train_weights_by_label está disponível (computa se necessário)
    _ensure_train_weights(pca_models, ctx)

    ts_full_labels = ctx.time_series_labels
    sp_indexs = ctx.simulation_points_indexs

    metrics = {
        "average_speed": [],
        "percentage_of_track": [],
        "average_vehicles": [],
        "km_traveled": [],
        "missing_time_series": 0,
    }

    # Para 'base': não precisa de dados do novo cenário
    cluster_sp_weights_new = {}
    if delta_mode != "base":
        max_size_2, ss2, _, _, labels_2, _ = read_single_scenario(
            second_dir, sep=cfg.csv_sep, intervals=cfg.intervals.as_dict()
        )
        X_2, mean_2, std_2 = process_dataset(ss2, max_size_2)
        X_2_denorm = _denormalize(X_2, mean_2, std_2)
        _, labels_2_dict = pre_processing_estimation(labels_2)

        for i in range(n_clusters):
            if i not in pca_models:
                continue
            sp_label = ts_full_labels[sp_indexs[i]]
            if sp_label not in labels_2_dict:
                continue
            sp_partial = X_2_denorm[labels_2_dict[sp_label]]
            if interp_method is None:
                weights, _, _ = _weights_from_sp(sp_partial, pca_models[i])
            else:
                weights, _, _ = _weights_from_sp_stretched(
                    sp_partial, pca_models[i], interp_method=interp_method
                )
            cluster_sp_weights_new[i] = weights

    for k, label in enumerate(ts_full_labels):
        link_id = label.split("-")[0]
        if link_id not in (ctx.streets_info or {}):
            metrics["missing_time_series"] += 1
            continue
        cluster_id = int(clusters_array[k])
        if cluster_id not in pca_models:
            metrics["missing_time_series"] += 1
            continue
        sp_w_new = cluster_sp_weights_new.get(cluster_id)  # None for 'base'
        reconstructed = _reconstruct_delta(
            label, pca_models[cluster_id], sp_w_new,
            delta_mode, percentage, use_eig, centroid_size,
        )
        if reconstructed is None:
            metrics["missing_time_series"] += 1
            continue
        reconstructed = _pruning(reconstructed)
        calc = calc_metrics(reconstructed, link_id, ctx.streets_info, printer=False)
        metrics["average_speed"].append(calc["average_speed"])
        metrics["percentage_of_track"].append(calc["percentage_of_track"])
        metrics["average_vehicles"].append(calc["average_vehicles"])
        metrics["km_traveled"].append(calc["km_traveled"])

    return metrics


# ---------------------------------------------------------------------------
# SweepStage
# ---------------------------------------------------------------------------

class SweepStage(Stage):
    """Executa estimação e comparação para múltiplos cenários parciais.

    Para cada cenário definido em ``scenarios``, o estágio:
      1. Carrega o cenário parcial
      2. Executa todos os métodos DTW indicados (+ PCA se ``include_pca=True``)
      3. Compara o resultado contra o cenário real
      4. Salva JSON por cenário e um resumo global

    Ao final, gera gráficos de progressão do erro vs. cenário (se
    ``generate_plots=True``).

    Parâmetros
    ----------
    real_file : str
        Nome do JSON com métricas reais (sem extensão). Default: 'new_simulation'.
    scenarios : list[dict]
        Lista de dicts com ``label``, ``dir`` e ``percentage``.
        Ex: [{"label": "50%", "dir": "input/.../50p/", "percentage": 50.0}]
    methods : list[str]
        Métodos DTW a executar. Default: ['avg', 'base', 'sample', 'wavg'].
    include_pca : bool
        Se True, também executa PCAEstimation para cada cenário. Default: False.
    output_prefix : str
        Prefixo para arquivos de saída. Default: 'sweep'.
    generate_plots : bool
        Se True, gera PDFs de progressão. Default: True.
    """

    name = "sweep"

    def __init__(
        self,
        real_file: str = "new_simulation",
        scenarios: list = None,
        methods: list = None,
        include_pca: bool = False,
        pca_use_sp: bool = True,
        output_prefix: str = "sweep",
        generate_plots: bool = True,
    ):
        self.real_file = real_file
        self.scenarios = scenarios or []
        self.methods = methods or ["avg", "base", "sample", "wavg"]
        self.include_pca = include_pca
        self.pca_use_sp = pca_use_sp
        self.output_prefix = output_prefix
        self.generate_plots = generate_plots

    def execute(self, ctx: PipelineContext) -> None:
        ensure_clustering_loaded(ctx)
        ensure_dir(ctx.output_dir)

        if not self.scenarios:
            print(f"[{self.name}] Nenhum cenário definido — nada a fazer.")
            return

        # Carrega métricas reais
        real_path = os.path.join(ctx.output_dir, self.real_file)
        try:
            real = load_json(real_path)
        except FileNotFoundError:
            raise RuntimeError(
                f"Arquivo de métricas reais não encontrado: '{real_path}.json'\n"
                "Execute ScenarioMetricsStage antes de SweepStage."
            )

        n_real_series = len(real.get("average_speed", []))
        print(f"\n[{self.name}] Iniciando sweep de {len(self.scenarios)} cenário(s)")
        print(f"  Cenário real: {self.real_file} ({n_real_series} séries)")
        print(f"  Métodos DTW:  {self.methods}")
        print(f"  PCA:          {'sim (modo SP)' if self.include_pca and self.pca_use_sp else 'sim (individual)' if self.include_pca else 'não'}")

        wps_folder = ctx.warping_paths_folder or os.path.join(
            ctx.input_dir or ctx.output_dir, "warping_paths"
        )

        # sweep_results: {scenario_label: {label, percentage, methods: {method: compare_result}}}
        sweep_results = {}

        for i, scenario in enumerate(self.scenarios, 1):
            label = scenario.get("label", f"scenario_{i}")
            second_dir = scenario.get("dir", "")
            percentage = float(scenario.get("percentage", 100.0))

            # dir vazio ou keyword 'training'/'100%' → usa o cenário de treinamento
            if not second_dir or second_dir.strip().lower() in ("training", "100%"):
                second_dir = ctx.config.scenario_dir
                print(f"  [{label}] Usando cenário de treinamento: {second_dir}")

            print(f"\n  ── Cenário {i}/{len(self.scenarios)}: {label} ({second_dir})")
            scenario_start = time.time()

            method_compare = {}

            # DTW — sem interpolação (padrão: TimeSeriesResampler linear via tslearn)
            if self.methods:
                print(f"    Executando métodos DTW (padrão): {self.methods}")
                dtw_results = _run_dtw_methods(
                    self.methods, ctx, second_dir, percentage, wps_folder
                )
                for method, metrics in dtw_results.items():
                    compare = _analyse_method(real, metrics, f"{method}@{label}")
                    method_compare[method] = compare
                    _print_method_report(compare)
                    est_key = f"{label.replace('%','p').replace(' ','_')}_{method}"
                    save_json(os.path.join(ctx.output_dir, f"{self.output_prefix}_{est_key}"), metrics)

                # DTW — com cada método de interpolação
                for interp in INTERP_METHODS:
                    for method in self.methods:
                        mkey = f"{method}_{interp}"
                        print(f"    Executando DTW ({mkey})")
                        dtw_results_interp = _run_dtw_methods(
                            [method], ctx, second_dir, percentage, wps_folder,
                            interp_method=interp,
                        )
                        compare = _analyse_method(real, dtw_results_interp[method], f"{mkey}@{label}")
                        method_compare[mkey] = compare
                        _print_method_report(compare)
                        est_key = f"{label.replace('%','p').replace(' ','_')}_{mkey}"
                        save_json(os.path.join(ctx.output_dir, f"{self.output_prefix}_{est_key}"), dtw_results_interp[method])

            # PCA — família completa
            if self.include_pca:
                def _run_and_record(pca_key, metrics_fn):
                    """Helper: roda, analisa, registra e salva."""
                    m = metrics_fn()
                    cmp = _analyse_method(real, m, f"{pca_key}@{label}")
                    method_compare[pca_key] = cmp
                    _print_method_report(cmp)
                    ekey = f"{label.replace('%','p').replace(' ','_')}_{pca_key}"
                    save_json(os.path.join(ctx.output_dir, f"{self.output_prefix}_{ekey}"), m)

                # ── Modos base (sem interp, sem eig) ──────────────────────────
                for pca_mode, pca_key in [("sp", "pca_sp"), ("ind", "pca_ind")]:
                    print(f"    Executando PCA ({pca_key})")
                    _run_and_record(pca_key,
                        lambda m=pca_mode: _run_pca_method(ctx, second_dir, mode=m))

                # ── Modos base + eigenvalue filter ────────────────────────────
                for pca_mode, pca_key in [("sp", "pca_sp_eig"), ("ind", "pca_ind_eig")]:
                    print(f"    Executando PCA ({pca_key})")
                    _run_and_record(pca_key,
                        lambda m=pca_mode: _run_pca_method(ctx, second_dir, mode=m, use_eig=True))

                # ── Modos com interpolação (sem e com eig) ────────────────────
                for interp in INTERP_METHODS:
                    for pca_mode, pca_key_base in [("sp", "pca_sp"), ("stretched", "pca_str")]:
                        # sem eig
                        pca_key = f"{pca_key_base}_{interp}"
                        print(f"    Executando PCA ({pca_key})")
                        _run_and_record(pca_key,
                            lambda m=pca_mode, im=interp: _run_pca_method(
                                ctx, second_dir, mode=m, interp_method=im))
                        # com eig
                        pca_key_e = f"{pca_key_base}_{interp}_eig"
                        print(f"    Executando PCA ({pca_key_e})")
                        _run_and_record(pca_key_e,
                            lambda m=pca_mode, im=interp: _run_pca_method(
                                ctx, second_dir, mode=m, interp_method=im, use_eig=True))

                # ── Delta modes (sem interp) ───────────────────────────────────
                for dm in ["base", "delta", "avg", "wavg"]:
                    for eig, esuf in [(False, ""), (True, "_eig")]:
                        pca_key = f"pca_d_{dm}{esuf}"
                        print(f"    Executando PCA ({pca_key})")
                        _run_and_record(pca_key,
                            lambda d=dm, e=eig: _run_pca_delta_method(
                                ctx, second_dir, delta_mode=d,
                                use_eig=e, percentage=percentage))

                # ── Delta modes com interpolação (sem base, pois não usa SP) ──
                for interp in INTERP_METHODS:
                    for dm in ["delta", "avg", "wavg"]:
                        for eig, esuf in [(False, ""), (True, "_eig")]:
                            pca_key = f"pca_d_{dm}_{interp}{esuf}"
                            print(f"    Executando PCA ({pca_key})")
                            _run_and_record(pca_key,
                                lambda d=dm, im=interp, e=eig: _run_pca_delta_method(
                                    ctx, second_dir, delta_mode=d, interp_method=im,
                                    use_eig=e, percentage=percentage))

            elapsed = time.time() - scenario_start
            sweep_results[label] = {
                "label": label,
                "percentage": percentage,
                "dir": second_dir,
                "methods": method_compare,
                "elapsed_s": round(elapsed, 2),
            }

            # Salva JSON de comparação do cenário
            scenario_key = label.replace('%', 'p').replace(' ', '_')
            compare_path = os.path.join(
                ctx.output_dir, f"{self.output_prefix}_compare_{scenario_key}"
            )
            scenario_compare_json = {
                m: {k: v for k, v in r.items() if k != "label"}
                for m, r in method_compare.items()
            }
            save_json(compare_path, scenario_compare_json)
            print(f"    ✓ {label} concluído em {elapsed:.1f}s → {compare_path}.json")

        # Salva resumo global
        summary_path = os.path.join(ctx.output_dir, f"{self.output_prefix}_summary")
        _save_sweep_summary(sweep_results, summary_path)
        print(f"\n[{self.name}] Resumo salvo em: {summary_path}.json")

        # Imprime tabela de progressão no terminal
        _print_sweep_table(sweep_results)

        # Gera PDFs
        if self.generate_plots and sweep_results:
            from pipeline.stages.plots import save_sweep_pdfs
            plots_dir = os.path.join(ctx.output_dir, "plots")
            scenario_name = ctx.config.scenario_name
            title = f" — {scenario_name}" if scenario_name else ""
            print(f"\n[{self.name}] Gerando gráficos de progressão em: {plots_dir}")
            save_sweep_pdfs(sweep_results, plots_dir, self.output_prefix, title)
            print(f"[{self.name}] Gráficos gerados.")

        print(f"\n[{self.name}] Sweep concluído: {len(sweep_results)} cenário(s) processados.")


# ---------------------------------------------------------------------------
# Serialização e impressão
# ---------------------------------------------------------------------------

def _serializable_compare(compare_result):
    """Remove campos não serializáveis do resultado de comparação."""
    result = {}
    for k, v in compare_result.items():
        if isinstance(v, float) and (v != v):  # NaN
            result[k] = None
        elif isinstance(v, dict):
            result[k] = _serializable_compare(v)
        elif isinstance(v, list):
            result[k] = [x if x == x else None for x in v]
        else:
            result[k] = v
    return result


def _save_sweep_summary(sweep_results, path):
    """Salva JSON resumido da sweep (apenas MAPE por método/métrica por cenário)."""
    summary = {"scenarios": []}
    for label, sdata in sweep_results.items():
        entry = {
            "label": label,
            "percentage": sdata["percentage"],
            "methods": {},
        }
        for method, comp in sdata["methods"].items():
            entry["methods"][method] = {
                "coverage_pct": comp["coverage_pct"],
                "mape": {
                    m: comp["per_series_error"][m]["mape"]
                    for m in _METRICS
                },
                "error_pct": comp["aggregate"]["error_pct"],
            }
        summary["scenarios"].append(entry)
    save_json(path, summary)


def _print_sweep_table(sweep_results):
    """Imprime tabela de progressão: linhas=cenários, colunas=método+métrica (MAPE)."""
    all_methods = []
    for sdata in sweep_results.values():
        for m in sdata["methods"]:
            if m not in all_methods:
                all_methods.append(m)

    print(f"\n  {'='*80}")
    print(f"  PROGRESSÃO DE ERRO (MAPE) — velocidade média")
    print(f"  {'='*80}")
    print(f"  {'Cenário':<12}", end="")
    for m in all_methods:
        print(f"  {m:>10}", end="")
    print()
    print(f"  {'─'*80}")

    for label, sdata in sweep_results.items():
        print(f"  {label:<12}", end="")
        for method in all_methods:
            if method in sdata["methods"]:
                mape = sdata["methods"][method]["per_series_error"]["average_speed"]["mape"]
                print(f"  {mape:>9.2f}%", end="")
            else:
                print(f"  {'—':>10}", end="")
        print()
    print(f"  {'─'*80}")
