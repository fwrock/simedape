from __future__ import annotations
from pipeline.config import SimEDaPEConfig


class PipelineContext:
    """Estado compartilhado entre os estágios do pipeline.

    Armazena todos os artefatos intermediários produzidos durante a execução:
    configurações, dataset carregado, resultado da clusterização, warping paths,
    métricas calculadas, etc.
    """

    def __init__(self, config: SimEDaPEConfig):
        self.config = config

        # --- Dataset ---
        self.streets_info: dict | None = None
        self.max_size: int | None = None
        self.street_series: list | None = None
        self.street_series_vehicles: list | None = None
        self.street_vehicles: dict | None = None
        self.time_series_labels: list | None = None
        self.street_series_labeled_vehicles: dict | None = None

        # --- Dataset processado (normalizado) ---
        self.X = None       # ndarray (n, size, 1)
        self.mean = None    # ndarray (n, 1, 1)
        self.std = None     # ndarray (n, 1, 1)

        # --- Clusterização ---
        self.clustering_result: dict | None = None  # dict carregado do JSON
        self.ts_labels_indexs: list | None = None

        # --- Pontos de simulação ---
        self.simulation_points_indexs: list | None = None
        self.simulation_points_labels: list | None = None

        # --- Warping paths ---
        self.warping_paths_folder: str | None = None

        # --- Resultados de estimação ---
        # dict: method_name -> metrics_dict
        self.estimation_results: dict = {}

        # --- Métricas do cenário original ---
        self.scenario_metrics: dict | None = None

        # --- Modelos PCA ---
        self.pca_models: dict | None = None      # {cluster_id -> {pca, mean, size, ...}}
        self.pca_models_path: str | None = None  # caminho do .joblib salvo

    @property
    def output_dir(self) -> str:
        return self.config.output_dir

    @property
    def input_dir(self) -> str:
        return self.config.input_dir

    @property
    def clustering_file(self) -> str:
        return self.config.clustering.output_file
