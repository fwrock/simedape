import os
import numpy as np

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.writer import save_json, load_json


def _simulation_points_indexs(clustering_result, X):
    """Para cada cluster, encontra a série mais próxima ao centróide."""
    n_clusters = clustering_result["number_of_clusters"]
    clusters = np.array(clustering_result["clusters"])
    all_indexs = np.arange(len(clusters))
    sp_indexs = []
    for k in range(n_clusters):
        centroid = np.array(clustering_result["centroids"][str(k)]).ravel()
        cluster_indexs = all_indexs[clusters == k]
        best_idx = cluster_indexs[0]
        best_dist = np.inf
        for i in cluster_indexs:
            dist = np.sum(np.abs(centroid - X[i].ravel()))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        sp_indexs.append(int(best_idx))
    return sp_indexs


def _simulation_points_labels(time_series_labels, sp_indexs):
    return [time_series_labels[i] for i in sp_indexs]


def _simulation_points_vehicles(street_vehicles, sp_labels):
    result = {}
    for label in sp_labels:
        street_id = label.split("-")[0]
        vehicles = set()
        for vehicle_list in street_vehicles.get(street_id, []):
            vehicles.update(vehicle_list)
        result[label] = list(vehicles)
    return result


def _clustering_vehicles(street_series_vehicles, clustering_result):
    """Agrega os veículos de cada série temporal por cluster (union)."""
    clusters = np.array(clustering_result["clusters"])
    n_clusters = clustering_result["number_of_clusters"]
    sv = np.array(street_series_vehicles, dtype=object)
    result = {}
    for k in range(n_clusters):
        sv_k = sv[clusters == k]
        union = set()
        for vehicle_list in sv_k:
            union.update(vehicle_list)
        result[str(k)] = list(union)
    return result


class SimulationPointsStage(Stage):
    """Identifica os pontos de simulação (série mais próxima ao centróide de cada cluster)
    e exporta os veículos associados."""

    name = "simulation_points"

    def execute(self, ctx: PipelineContext) -> None:
        ensure_clustering_loaded(ctx)

        print(f"[{self.name}] Calculando pontos de simulação...")
        sp_indexs = _simulation_points_indexs(ctx.clustering_result, ctx.X)
        sp_labels = _simulation_points_labels(ctx.time_series_labels, sp_indexs)
        ctx.simulation_points_indexs = sp_indexs
        ctx.simulation_points_labels = sp_labels

        out = ctx.output_dir
        save_json(os.path.join(out, "simulation_points_indexs"), [int(x) for x in sp_indexs])
        save_json(os.path.join(out, "simulation_points_labels"), sp_labels)
        print(f"[{self.name}] {len(sp_indexs)} pontos de simulação identificados")

        if ctx.street_vehicles:
            sp_vehicles = _simulation_points_vehicles(ctx.street_vehicles, sp_labels)
            save_json(os.path.join(out, "simulation_points_vehicles"), sp_vehicles)

        if ctx.street_series_vehicles:
            clusters_vehicles = _clustering_vehicles(
                ctx.street_series_vehicles, ctx.clustering_result
            )
            save_json(os.path.join(out, "clustering_vehicles"), clusters_vehicles)

        # Salva labels indexados (label -> index)
        ts_label_indexs = list(range(len(ctx.time_series_labels)))
        save_json(os.path.join(out, "time_series_labels_indexs"), ts_label_indexs)


def ensure_clustering_loaded(ctx: PipelineContext):
    """Garante que clustering_result esteja disponível no contexto,
    carregando do disco se necessário."""
    if ctx.clustering_result is None:
        path = os.path.join(ctx.input_dir or ctx.output_dir, ctx.clustering_file)
        print(f"[ensure_clustering] Carregando clustering de: {path}.json")
        ctx.clustering_result = load_json(path)

    if ctx.simulation_points_indexs is None:
        sp_path = os.path.join(ctx.input_dir or ctx.output_dir, "simulation_points_indexs")
        try:
            ctx.simulation_points_indexs = load_json(sp_path)
        except FileNotFoundError:
            pass

    if ctx.time_series_labels is None:
        labels_path = os.path.join(ctx.input_dir or ctx.output_dir, "time_series_labels")
        try:
            ctx.time_series_labels = load_json(labels_path)
        except FileNotFoundError:
            pass
