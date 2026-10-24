import os
import time
import logging
import numpy as np
from scipy.stats import gmean

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.writer import save_json, load_json, ensure_dir
from pipeline.domain.dtw import estimate, pruning
from pipeline.domain.metrics import calc_metrics
from pipeline.stages.simulation_points import ensure_clustering_loaded


# ---------------------------------------------------------------------------
# Helpers compartilhados entre os métodos de estimação
# ---------------------------------------------------------------------------

def _get_serie_index(label_dict, labels, idx):
    return label_dict[labels[idx]]


def _check_serie_exists(label_dict, labels, idx):
    return labels[idx] in label_dict


def _get_serie_from(dataset, label_dict, labels, idx):
    return dataset[label_dict[labels[idx]]].ravel().astype(float)


def _empty_metrics():
    return {
        "average_speed": [],
        "percentage_of_track": [],
        "average_vehicles": [],
        "km_traveled": [],
        "missing_simulation_points": 0,
        "missing_time_series": 0,
    }


def _append_metrics(metrics, calc):
    metrics["average_speed"].append(calc["average_speed"])
    metrics["percentage_of_track"].append(calc["percentage_of_track"])
    metrics["average_vehicles"].append(calc["average_vehicles"])
    metrics["km_traveled"].append(calc["km_traveled"])


def _print_summary(metrics):
    print(f"  Velocidade Média (gmean): {gmean(metrics['average_speed']):.4f}")
    print(f"  Ocupação (gmean):         {gmean(metrics['percentage_of_track']):.4f}")
    print(f"  Veículos (gmean):         {gmean(metrics['average_vehicles']):.4f}")
    print(f"  Km Trafegados (gmean):    {gmean(metrics['km_traveled']):.4f}")
    print(f"  Séries ausentes:          {metrics['missing_time_series']}")
    print(f"  SP ausentes:              {metrics['missing_simulation_points']}")


# ---------------------------------------------------------------------------
# Implementação dos 4 métodos de estimação
# ---------------------------------------------------------------------------

def run_estimation_avg(
    links_info, clustering, mean, std, mean_2, std_2, X_2,
    ts_labels_first_indexs, ts_labels_first, ts_dict_first, ts_dict_second,
    sp_indexs, wps_folder, printer=False
):
    """Método avg: normalização combinada (média aritmética de ambos os cenários)."""
    metrics = _empty_metrics()
    ts_indexs = np.array(ts_labels_first_indexs)

    for i in range(clustering["number_of_clusters"]):
        print(f"  === Cluster {i} ===")
        try:
            wps = load_json(os.path.join(wps_folder, f"cluster_{i}"))
            cluster_indexs = ts_indexs[np.array(clustering["clusters"]) == i]
            for j, k in enumerate(cluster_indexs):
                if not _check_serie_exists(ts_dict_second, ts_labels_first, sp_indexs[i]):
                    metrics["missing_simulation_points"] += 1
                    continue
                if not _check_serie_exists(ts_dict_second, ts_labels_first, k):
                    metrics["missing_time_series"] += 1
                    continue
                wp = wps[str(j)]
                std_s2 = std[_get_serie_index(ts_dict_first, ts_labels_first, k)][0][0]
                mean_s2 = mean[_get_serie_index(ts_dict_first, ts_labels_first, k)][0][0]
                std_s23 = std_2[_get_serie_index(ts_dict_second, ts_labels_first, k)][0][0]
                mean_s23 = mean_2[_get_serie_index(ts_dict_second, ts_labels_first, k)][0][0]
                std_combined = (std_s2 + std_s23) / 2
                mean_combined = (mean_s2 + mean_s23) / 2
                w1 = _get_serie_from(X_2, ts_dict_second, ts_labels_first, sp_indexs[i])
                estimated = estimate(w1, wp, std_combined, mean_combined)
                link_id = ts_labels_first[k].split("-")[0]
                calc = calc_metrics(pruning(estimated), link_id, links_info, printer=printer)
                _append_metrics(metrics, calc)
        except Exception as e:
            logging.exception(f"Erro no cluster {i}: {e}")

    return metrics


def run_estimation_sample(
    links_info, clustering, mean, std, mean_2, std_2, X_2,
    ts_labels_first_indexs, ts_labels_first, ts_dict_first, ts_dict_second,
    sp_indexs, wps_folder, printer=False
):
    """Método sample: usa apenas a normalização do segundo cenário (parcial)."""
    metrics = _empty_metrics()
    ts_indexs = np.array(ts_labels_first_indexs)

    for i in range(clustering["number_of_clusters"]):
        print(f"  === Cluster {i} ===")
        try:
            wps = load_json(os.path.join(wps_folder, f"cluster_{i}"))
            cluster_indexs = ts_indexs[np.array(clustering["clusters"]) == i]
            for j, k in enumerate(cluster_indexs):
                if not _check_serie_exists(ts_dict_second, ts_labels_first, sp_indexs[i]):
                    metrics["missing_simulation_points"] += 1
                    continue
                if not _check_serie_exists(ts_dict_second, ts_labels_first, k):
                    metrics["missing_time_series"] += 1
                    continue
                wp = wps[str(j)]
                std_s23 = std_2[_get_serie_index(ts_dict_second, ts_labels_first, k)][0][0]
                mean_s23 = mean_2[_get_serie_index(ts_dict_second, ts_labels_first, k)][0][0]
                w1 = _get_serie_from(X_2, ts_dict_second, ts_labels_first, sp_indexs[i])
                estimated = estimate(w1, wp, std_s23, mean_s23)
                link_id = ts_labels_first[k].split("-")[0]
                calc = calc_metrics(pruning(estimated), link_id, links_info, printer=printer)
                _append_metrics(metrics, calc)
        except Exception as e:
            logging.exception(f"Erro no cluster {i}: {e}")

    return metrics


def run_estimation_wavg(
    links_info, clustering, mean, std, mean_2, std_2, X_2,
    ts_labels_first_indexs, ts_labels_first, ts_dict_first, ts_dict_second,
    sp_indexs, wps_folder, percentage_sample=100.0, printer=False
):
    """Método wavg: normalização ponderada pelo percentual de amostra."""
    metrics = _empty_metrics()
    ts_indexs = np.array(ts_labels_first_indexs)
    w = percentage_sample / 100.0

    for i in range(clustering["number_of_clusters"]):
        print(f"  === Cluster {i} ===")
        try:
            wps = load_json(os.path.join(wps_folder, f"cluster_{i}"))
            cluster_indexs = ts_indexs[np.array(clustering["clusters"]) == i]
            for j, k in enumerate(cluster_indexs):
                if not _check_serie_exists(ts_dict_second, ts_labels_first, sp_indexs[i]):
                    metrics["missing_simulation_points"] += 1
                    continue
                if not _check_serie_exists(ts_dict_second, ts_labels_first, k):
                    metrics["missing_time_series"] += 1
                    continue
                wp = wps[str(j)]
                std_s2 = std[_get_serie_index(ts_dict_first, ts_labels_first, k)][0][0]
                mean_s2 = mean[_get_serie_index(ts_dict_first, ts_labels_first, k)][0][0]
                std_s23 = std_2[_get_serie_index(ts_dict_second, ts_labels_first, k)][0][0]
                mean_s23 = mean_2[_get_serie_index(ts_dict_second, ts_labels_first, k)][0][0]
                std_combined = (std_s2 + std_s23 * w) / (w + 1)
                mean_combined = (mean_s2 + mean_s23 * w) / (w + 1)
                w1 = _get_serie_from(X_2, ts_dict_second, ts_labels_first, sp_indexs[i])
                estimated = estimate(w1, wp, std_combined, mean_combined)
                link_id = ts_labels_first[k].split("-")[0]
                calc = calc_metrics(pruning(estimated), link_id, links_info, printer=printer)
                _append_metrics(metrics, calc)
        except Exception as e:
            logging.exception(f"Erro no cluster {i}: {e}")

    return metrics


def run_estimation_base(
    links_info, clustering, mean, std, mean_2, std_2, X_2,
    ts_labels_first_indexs, ts_labels_first, ts_dict_first, ts_dict_second,
    sp_indexs, wps_folder, printer=False
):
    """Método base: usa o centróide diretamente como estimativa (sem warping)."""
    metrics = _empty_metrics()
    ts_indexs = np.array(ts_labels_first_indexs)

    for i in range(clustering["number_of_clusters"]):
        print(f"  === Cluster {i} ===")
        try:
            wps = load_json(os.path.join(wps_folder, f"cluster_{i}"))
            cluster_indexs = ts_indexs[np.array(clustering["clusters"]) == i]
            centroid = np.array(clustering["centroids"][str(i)]).ravel()
            for j, k in enumerate(cluster_indexs):
                if not _check_serie_exists(ts_dict_second, ts_labels_first, sp_indexs[i]):
                    metrics["missing_simulation_points"] += 1
                    continue
                if not _check_serie_exists(ts_dict_second, ts_labels_first, k):
                    metrics["missing_time_series"] += 1
                    continue
                wp = wps[str(j)]
                std_s2 = std[_get_serie_index(ts_dict_first, ts_labels_first, k)][0][0]
                mean_s2 = mean[_get_serie_index(ts_dict_first, ts_labels_first, k)][0][0]
                estimated = estimate(centroid, wp, std_s2, mean_s2)
                link_id = ts_labels_first[k].split("-")[0]
                calc = calc_metrics(pruning(estimated), link_id, links_info, printer=printer)
                _append_metrics(metrics, calc)
        except Exception as e:
            logging.exception(f"Erro no cluster {i}: {e}")

    return metrics


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

class EstimationStage(Stage):
    """Executa um ou mais métodos de estimação e salva os resultados.

    Parâmetros
    ----------
    methods : list[str]
        Métodos a executar. Opções: 'avg', 'base', 'sample', 'wavg'.
    second_scenario_dir : str
        Diretório do cenário parcial (segundo cenário) a ser estimado.
    percentage_sample : float
        Usado apenas pelo método 'wavg'. Percentual da amostra simulada.
    output_prefix : str
        Prefixo dos arquivos de saída. Ex: 'result' → 'result_avg.json'.
    """

    name = "estimation"

    def __init__(
        self,
        methods=None,
        second_scenario_dir: str = "",
        percentage_sample: float = 100.0,
        output_prefix: str = "result",
    ):
        self.methods = methods or ["avg"]
        self.second_scenario_dir = second_scenario_dir
        self.percentage_sample = percentage_sample
        self.output_prefix = output_prefix

    def execute(self, ctx: PipelineContext) -> None:
        ensure_clustering_loaded(ctx)
        cfg = ctx.config

        # Carrega segundo cenário
        from pipeline.io.reader import read_single_scenario
        from pipeline.domain.processor import process_dataset, pre_processing_estimation

        second_dir = self.second_scenario_dir or cfg.scenario_dir
        print(f"[{self.name}] Carregando segundo cenário: {second_dir}")
        _, ss2, _, _, labels_2, _ = read_single_scenario(
            second_dir, sep=cfg.csv_sep, intervals=cfg.intervals.as_dict()
        )
        clustering = ctx.clustering_result
        centroid_size = len(clustering["centroids"]["0"])
        X_2, mean_2, std_2 = process_dataset(ss2, centroid_size)

        ts_labels_first = ctx.time_series_labels
        ts_indexs_first, ts_dict_first = pre_processing_estimation(ts_labels_first)
        _, ts_dict_second = pre_processing_estimation(labels_2)

        sp_indexs = ctx.simulation_points_indexs
        wps_folder = ctx.warping_paths_folder or os.path.join(
            ctx.input_dir or ctx.output_dir, "warping_paths"
        )

        ensure_dir(ctx.output_dir)

        mean = np.array(clustering["mean"])
        std = np.array(clustering["standard_deviation"])

        method_funcs = {
            "avg": lambda: run_estimation_avg(
                ctx.streets_info, clustering, mean, std, mean_2, std_2, X_2,
                ts_indexs_first, ts_labels_first, ts_dict_first, ts_dict_second,
                sp_indexs, wps_folder,
            ),
            "sample": lambda: run_estimation_sample(
                ctx.streets_info, clustering, mean, std, mean_2, std_2, X_2,
                ts_indexs_first, ts_labels_first, ts_dict_first, ts_dict_second,
                sp_indexs, wps_folder,
            ),
            "wavg": lambda: run_estimation_wavg(
                ctx.streets_info, clustering, mean, std, mean_2, std_2, X_2,
                ts_indexs_first, ts_labels_first, ts_dict_first, ts_dict_second,
                sp_indexs, wps_folder,
                percentage_sample=self.percentage_sample,
            ),
            "base": lambda: run_estimation_base(
                ctx.streets_info, clustering, mean, std, mean_2, std_2, X_2,
                ts_indexs_first, ts_labels_first, ts_dict_first, ts_dict_second,
                sp_indexs, wps_folder,
            ),
        }

        for method in self.methods:
            if method not in method_funcs:
                print(f"[{self.name}] Método desconhecido ignorado: {method}")
                continue
            print(f"\n[{self.name}] Executando método: {method}")
            start = time.time()
            metrics = method_funcs[method]()
            elapsed = time.time() - start
            _print_summary(metrics)
            print(f"  Tempo: {elapsed:.1f}s")

            out_path = os.path.join(ctx.output_dir, f"{self.output_prefix}_{method}")
            save_json(out_path, metrics)
            ctx.estimation_results[method] = metrics
            print(f"  Salvo em: {out_path}.json")
