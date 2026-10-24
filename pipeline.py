from __future__ import annotations
import os
import time
from typing import List

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.config import SimEDaPEConfig


class Pipeline:
    """Orquestrador do pipeline SimEDaPE.

    Permite encadear estágios e executá-los sequencialmente, compartilhando
    um contexto comum entre eles.

    Uso via Python API
    ------------------
    >>> from pipeline import Pipeline
    >>> from pipeline.stages import DataLoadStage, ProcessDatasetStage, ClusteringStage
    >>> from pipeline.config import SimEDaPEConfig
    >>>
    >>> config = SimEDaPEConfig.from_properties_file("properties/carla_local.properties")
    >>> result = (
    ...     Pipeline(config)
    ...     .add(DataLoadStage())
    ...     .add(ProcessDatasetStage())
    ...     .add(ClusteringStage())
    ...     .run()
    ... )

    Uso via YAML
    ------------
    >>> pipeline = Pipeline.from_yaml("pipeline.yml")
    >>> pipeline.run()
    """

    def __init__(self, config: SimEDaPEConfig):
        self.config = config
        self._stages: List[Stage] = []

    def add(self, stage: Stage) -> "Pipeline":
        """Adiciona um estágio ao final do pipeline."""
        self._stages.append(stage)
        return self

    def run(self) -> PipelineContext:
        """Executa todos os estágios em ordem e retorna o contexto final."""
        ctx = PipelineContext(self.config)
        total_start = time.time()

        print(f"\n{'='*60}")
        print(f"  SimEDaPE Pipeline — {len(self._stages)} estágio(s)")
        print(f"  Cenário: {self.config.scenario_name}")
        print(f"  Saída:   {self.config.output_dir}")
        print(f"{'='*60}\n")

        for i, stage in enumerate(self._stages, 1):
            print(f"[{i}/{len(self._stages)}] Estágio: {stage.name}")
            stage_start = time.time()
            stage.execute(ctx)
            elapsed = time.time() - stage_start
            print(f"  ✓ {stage.name} concluído em {elapsed:.1f}s\n")

        total = time.time() - total_start
        print(f"{'='*60}")
        print(f"  Pipeline concluído em {total:.1f}s")
        print(f"{'='*60}\n")
        return ctx

    @classmethod
    def from_properties(cls, properties_file: str) -> "Pipeline":
        """Cria um pipeline a partir de um arquivo .properties."""
        config = SimEDaPEConfig.from_properties_file(properties_file)
        return cls(config)

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Pipeline":
        """Cria e configura um pipeline completo a partir de um arquivo YAML."""
        import yaml
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        config = SimEDaPEConfig.from_yaml(yaml_file)
        pipeline = cls(config)

        stage_registry = _build_stage_registry()
        for stage_def in data.get("stages", []):
            stage_name = stage_def if isinstance(stage_def, str) else stage_def["name"]
            params = {} if isinstance(stage_def, str) else stage_def.get("params", {})

            if stage_name not in stage_registry:
                raise ValueError(
                    f"Estágio desconhecido: '{stage_name}'. "
                    f"Disponíveis: {list(stage_registry.keys())}"
                )
            stage_cls = stage_registry[stage_name]
            pipeline.add(stage_cls(**params))

        return pipeline


def _build_stage_registry() -> dict:
    from pipeline.stages import (
        DataLoadStage, ProcessDatasetStage, ClusteringStage,
        SimulationPointsStage, WarpingPathsStage,
        ScenarioMetricsStage, EstimationStage, CompareStage,
        PCATrainStage, PCAEstimationStage, SweepStage,
        SelectiveInstantiationStage,
    )
    return {
        "data_load": DataLoadStage,
        "process_dataset": ProcessDatasetStage,
        "clustering": ClusteringStage,
        "simulation_points": SimulationPointsStage,
        "warping_paths": WarpingPathsStage,
        "scenario_metrics": ScenarioMetricsStage,
        "estimation": EstimationStage,
        "compare": CompareStage,
        "pca_train": PCATrainStage,
        "pca_estimation": PCAEstimationStage,
        "sweep": SweepStage,
        "selective_instantiation": SelectiveInstantiationStage,
    }
