import os
import json
import time
import logging
import numpy as np

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.reader import read_single_scenario, read_street_info
from pipeline.io.clickhouse import read_clickhouse_scenario, ClickHouseClient
from pipeline.io.writer import save_json, ensure_dir
from pipeline.domain.processor import process_dataset
from tslearn.clustering import KShape


class DataLoadStage(Stage):
    """Carrega o cenário (CSV ou ClickHouse) e as informações das vias."""

    name = "data_load"

    def execute(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        if cfg.data_source == "clickhouse":
            self._load_clickhouse(ctx)
        else:
            self._load_csv(ctx)

        print(f"[{self.name}] Séries temporais: {len(ctx.street_series)}, max_size: {ctx.max_size}")

        if cfg.street_info_file:
            print(f"[{self.name}] Carregando informações das vias: {cfg.street_info_file}")
            ctx.streets_info = read_street_info(cfg.street_info_file, sep=cfg.csv_sep)
            print(f"[{self.name}] Vias carregadas: {len(ctx.streets_info)}")

    def _load_csv(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        print(f"[{self.name}] Carregando cenário (CSV): {cfg.scenario_dir}")
        (
            ctx.max_size,
            ctx.street_series,
            ctx.street_series_vehicles,
            ctx.street_vehicles,
            ctx.time_series_labels,
            ctx.street_series_labeled_vehicles,
        ) = read_single_scenario(
            cfg.scenario_dir,
            sep=cfg.csv_sep,
            intervals=cfg.intervals.as_dict(),
        )

    def _load_clickhouse(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        ch = cfg.clickhouse
        scope = f"{ch.database}.{ch.table}"
        if ch.simulation_id:
            scope += f" (simulation_id={ch.simulation_id})"
        print(f"[{self.name}] Carregando cenário do ClickHouse: {scope}")

        client = ClickHouseClient(
            host=ch.host,
            port=ch.port,
            database=ch.database,
            user=ch.user,
            password=ch.password,
            secure=ch.secure,
        )
        (
            ctx.max_size,
            ctx.street_series,
            ctx.street_series_vehicles,
            ctx.street_vehicles,
            ctx.time_series_labels,
            ctx.street_series_labeled_vehicles,
        ) = read_clickhouse_scenario(
            client,
            intervals=cfg.intervals.as_dict(),
            simulation_id=ch.simulation_id,
            table=ch.table,
        )


class ProcessDatasetStage(Stage):
    """Normaliza e reamostrada as séries temporais."""

    name = "process_dataset"

    def execute(self, ctx: PipelineContext) -> None:
        print(f"[{self.name}] Processando dataset...")
        clustering_result = ctx.clustering_result
        # Usa tamanho do centróide se clusterização já foi carregada
        size = (
            len(clustering_result["centroids"]["0"])
            if clustering_result
            else ctx.max_size
        )
        ctx.X, ctx.mean, ctx.std = process_dataset(ctx.street_series, size)
        print(f"[{self.name}] Shape: {ctx.X.shape}")


class ClusteringStage(Stage):
    """Executa K-Shape clustering e salva o resultado."""

    name = "clustering"

    def execute(self, ctx: PipelineContext) -> None:
        cfg = ctx.config
        ensure_dir(ctx.output_dir)

        n_clusters = cfg.clustering.n_clusters
        clustering_size = cfg.clustering.clustering_size

        # Reamostra para clustering_size se configurado (acelera muito o KShape)
        X_clust = ctx.X
        if clustering_size and clustering_size < ctx.X.shape[1]:
            from tslearn.preprocessing import TimeSeriesResampler
            print(f"[{self.name}] Reamostrando para clustering: {ctx.X.shape[1]} → {clustering_size} pontos")
            X_clust = TimeSeriesResampler(sz=clustering_size).fit_transform(ctx.X)
            print(f"  Shape para clustering: {X_clust.shape}")
        else:
            clustering_size = ctx.X.shape[1]

        print(f"[{self.name}] Iniciando K-Shape")
        print(f"  n_clusters={n_clusters}, max_iter={cfg.clustering.max_iter}, "
              f"random_state={cfg.clustering.random_state}")
        print(f"  Dataset shape: {X_clust.shape}")
        start = time.time()
        ks = KShape(
            n_clusters=n_clusters,
            verbose=True,
            random_state=cfg.clustering.random_state,
            max_iter=cfg.clustering.max_iter,
        )
        y_pred = ks.fit_predict(X_clust)
        elapsed = time.time() - start
        print(f"[{self.name}] Clustering finalizado em {elapsed:.1f}s")
        unique, counts = np.unique(y_pred, return_counts=True)
        print(f"  Distribuição por cluster:")
        for c, cnt in zip(unique, counts):
            print(f"    cluster {c:>2}: {cnt:>4} séries")

        # Se reamostrou para clustering, expande centróides de volta para max_size
        full_size = ctx.X.shape[1]
        centroids = ks.cluster_centers_
        if centroids.shape[1] != full_size:
            from tslearn.preprocessing import TimeSeriesResampler
            print(f"[{self.name}] Expandindo centróides: {centroids.shape[1]} → {full_size} pontos")
            centroids = TimeSeriesResampler(sz=full_size).fit_transform(centroids)

        # Serializa resultado
        result = {
            "number_of_clusters": n_clusters,
            "clusters": y_pred.tolist(),
            "centroids": {str(k): centroids[k].ravel().tolist() for k in range(n_clusters)},
            "mean": ctx.mean.tolist(),
            "standard_deviation": ctx.std.tolist(),
            "time_series_by_cluster": {
                str(k): int(np.sum(y_pred == k)) for k in range(n_clusters)
            },
        }
        output_path = os.path.join(ctx.output_dir, cfg.clustering.output_file)
        save_json(output_path, result)
        ctx.clustering_result = result
        print(f"[{self.name}] Resultado salvo em: {output_path}.json")

        # Salva dataset por cluster
        print(f"[{self.name}] Salvando datasets por cluster...")
        for k in range(n_clusters):
            cluster_data = ctx.X[np.array(y_pred) == k]
            save_json(os.path.join(ctx.output_dir, f"dataset_cluster_{k}"), cluster_data.tolist())
            print(f"  dataset_cluster_{k}.json ({len(cluster_data)} séries)")

        # Salva labels
        save_json(os.path.join(ctx.output_dir, "time_series_labels"), ctx.time_series_labels)
        save_json(os.path.join(ctx.output_dir, "street_series_vehicles"), ctx.street_series_vehicles)
        print(f"[{self.name}] Labels salvas em: {ctx.output_dir}/time_series_labels.json")
