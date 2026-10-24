import os
import time
import numpy as np
from joblib import Parallel, delayed

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.writer import save_json, load_json, ensure_dir
from pipeline.domain.dtw import get_warping_path_fast
from pipeline.stages.simulation_points import ensure_clustering_loaded


def _compute_cluster_warping_paths(k, clustering_result, dataset, output_folder):
    """Calcula os warping paths entre o centróide do cluster k e todas as suas séries."""
    centroid = np.array(clustering_result["centroids"][str(k)]).ravel()
    clusters = np.array(clustering_result["clusters"])
    cluster_indexs = np.arange(len(clusters))[clusters == k]
    print(f"  Cluster {k}: {len(cluster_indexs)} séries")

    cluster_wps = {}
    for j, i in enumerate(cluster_indexs):
        s = np.array(dataset[i]).ravel()
        wp, _ = get_warping_path_fast(centroid, s)
        cluster_wps[str(j)] = wp

    out_path = os.path.join(output_folder, f"cluster_{k}")
    save_json(out_path, cluster_wps)
    return k


def _compute_cluster_warping_paths_from_file(k, clustering_result, dataset_folder, output_folder):
    """Versão que carrega o dataset do cluster a partir de arquivo (economiza memória)."""
    centroid = np.array(clustering_result["centroids"][str(k)]).ravel()
    dataset = np.array(load_json(os.path.join(dataset_folder, f"dataset_cluster_{k}")))
    print(f"  Cluster {k}: {len(dataset)} séries (carregado do disco)")

    cluster_wps = {}
    for j in range(len(dataset)):
        s = np.array(dataset[j]).ravel()
        wp, _ = get_warping_path_fast(centroid, s)
        cluster_wps[str(j)] = wp
        dataset[j] = None  # libera memória

    out_path = os.path.join(output_folder, f"cluster_{k}")
    save_json(out_path, cluster_wps)
    return k


class WarpingPathsStage(Stage):
    """Calcula os DTW warping paths entre o centróide de cada cluster e todas
    as suas séries temporais.

    Parâmetros
    ----------
    n_jobs : int
        Número de processos paralelos (default: 1).
    load_from_disk : bool
        Se True, carrega cada cluster do disco ao invés de manter tudo em memória.
    """

    name = "warping_paths"

    def __init__(self, n_jobs: int = 1, load_from_disk: bool = False):
        self.n_jobs = n_jobs
        self.load_from_disk = load_from_disk

    def execute(self, ctx: PipelineContext) -> None:
        ensure_clustering_loaded(ctx)
        clustering = ctx.clustering_result
        n_clusters = clustering["number_of_clusters"]
        wps_folder = os.path.join(ctx.output_dir, "warping_paths")
        ensure_dir(wps_folder)
        ctx.warping_paths_folder = wps_folder

        print(f"[{self.name}] Calculando warping paths para {n_clusters} clusters "
              f"(n_jobs={self.n_jobs}, from_disk={self.load_from_disk})...")
        start = time.time()

        if self.load_from_disk:
            dataset_folder = ctx.output_dir
            Parallel(n_jobs=self.n_jobs)(
                delayed(_compute_cluster_warping_paths_from_file)(
                    k, clustering, dataset_folder, wps_folder
                )
                for k in range(n_clusters)
            )
        else:
            if ctx.X is None:
                raise RuntimeError(
                    "Dataset não está no contexto. Execute DataLoadStage e "
                    "ProcessDatasetStage antes de WarpingPathsStage."
                )
            Parallel(n_jobs=self.n_jobs)(
                delayed(_compute_cluster_warping_paths)(
                    k, clustering, ctx.X, wps_folder
                )
                for k in range(n_clusters)
            )

        elapsed = time.time() - start
        print(f"[{self.name}] Warping paths finalizados em {elapsed:.1f}s → {wps_folder}")
