import os
import numpy as np

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.writer import save_json, ensure_dir
from pipeline.domain.metrics import calc_metrics
from pipeline.stages.simulation_points import ensure_clustering_loaded

# Recognised transport mode values (matches HTC agent_type column values)
_VALID_MODES = {"car", "bike", "bus", "subway", "person"}


class ScenarioMetricsStage(Stage):
    """Calcula as métricas de tráfego (velocidade, ocupação, veículos, km)
    para todas as séries temporais do cenário (denormalizadas).

    Quando ``SimEDaPEConfig.transport_modes`` contém mais de um modo,
    métricas per-mode são emitidas adicionalmente:
    - ``avg_travel_time_{mode}``
    - ``throughput_{mode}``
    - ``modal_split_{mode}``  (fração de viagens por esse modo)

    Equivale à ação `new_simulation_metric` do main.py original.

    Parâmetros
    ----------
    output_file : str
        Nome do arquivo JSON de saída (sem extensão).
    """

    name = "scenario_metrics"

    def __init__(self, output_file: str = "new_simulation"):
        self.output_file = output_file

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_base_metrics(X_denorm, time_series_labels, streets_info):
        """Computes the four base traffic metrics for all time-series."""
        metrics = {
            "average_speed": [],
            "percentage_of_track": [],
            "average_vehicles": [],
            "km_traveled": [],
        }
        for x in range(len(X_denorm)):
            label = time_series_labels[x]
            link_id = label.split("-")[0]
            if link_id not in streets_info:
                continue
            serie = X_denorm[x].ravel()
            calc = calc_metrics(serie, link_id, streets_info, label)
            metrics["average_speed"].append(calc["average_speed"])
            metrics["percentage_of_track"].append(calc["percentage_of_track"])
            metrics["average_vehicles"].append(calc["average_vehicles"])
            metrics["km_traveled"].append(calc["km_traveled"])
        return metrics

    @staticmethod
    def _compute_per_mode_metrics(
        X_denorm,
        time_series_labels: list,
        streets_info: dict,
        transport_modes: list[str],
    ) -> dict:
        """Computes per-mode aggregate metrics.

        The label convention used by the pipeline is ``<link_id>-<interval>``.
        The mode is inferred from ``streets_info`` if a ``mode`` key is present
        (``streets_info[link_id]`` may be a list ``[length, capacity, freespeed]``
        or a dict with an additional ``"mode"`` key).  When no mode information
        is available in ``streets_info``, all series are attributed to the first
        entry in ``transport_modes`` (graceful degradation).

        Returns a flat dict whose keys follow the patterns:
            ``avg_travel_time_{mode}``, ``throughput_{mode}``,
            ``modal_split_{mode}``
        """
        per_mode: dict[str, dict] = {
            m: {"travel_times": [], "vehicle_counts": []}
            for m in transport_modes
        }

        total_vehicles = 0.0

        for x in range(len(X_denorm)):
            label = time_series_labels[x]
            link_id = label.split("-")[0]
            if link_id not in streets_info:
                continue

            serie = X_denorm[x].ravel()

            # Determine mode for this link
            link_info = streets_info[link_id]
            if isinstance(link_info, dict):
                mode = link_info.get("mode", transport_modes[0])
            else:
                # Default list format [length, capacity, freespeed]; no mode info
                mode = transport_modes[0]

            if mode not in per_mode:
                mode = transport_modes[0]

            avg_count = float(np.mean(serie))
            per_mode[mode]["vehicle_counts"].append(avg_count)
            total_vehicles += avg_count

            # Travel time proxy: mean count / 1 (unit normalised; simulator
            # uses tick-based counts, so "travel time" here is proportional
            # to average occupancy × interval_size)
            per_mode[mode]["travel_times"].append(avg_count)

        result: dict = {}
        for mode in transport_modes:
            counts = per_mode[mode]["vehicle_counts"]
            mode_total = sum(counts)

            avg_travel_time = float(np.mean(counts)) if counts else 0.0
            throughput = float(mode_total)
            modal_split = (mode_total / total_vehicles) if total_vehicles > 0 else 0.0

            result[f"avg_travel_time_{mode}"] = avg_travel_time
            result[f"throughput_{mode}"] = throughput
            result[f"modal_split_{mode}"] = round(modal_split, 6)

        return result

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def execute(self, ctx: PipelineContext) -> None:
        ensure_clustering_loaded(ctx)

        if ctx.streets_info is None:
            raise RuntimeError(
                "streets_info não carregado. Execute DataLoadStage com "
                "street_info_file configurado."
            )
        if ctx.X is None or ctx.mean is None or ctx.std is None:
            raise RuntimeError(
                "Dataset normalizado não encontrado. Execute ProcessDatasetStage primeiro."
            )

        ensure_dir(ctx.output_dir)

        transport_modes: list[str] = getattr(ctx.config, "transport_modes", ["car"])

        print(f"[{self.name}] Calculando métricas para {len(ctx.X)} séries temporais...")
        if len(transport_modes) > 1:
            print(f"[{self.name}] Modos de transporte: {transport_modes}")

        X_denorm = ctx.X * ctx.std + ctx.mean

        # --- Base metrics (always computed) ---
        metrics = self._compute_base_metrics(X_denorm, ctx.time_series_labels, ctx.streets_info)

        # --- Per-mode metrics (only when more than one mode is configured) ---
        if len(transport_modes) > 1:
            per_mode_metrics = self._compute_per_mode_metrics(
                X_denorm, ctx.time_series_labels, ctx.streets_info, transport_modes
            )
            metrics.update(per_mode_metrics)
            print(f"[{self.name}] Per-mode metrics computed for: {transport_modes}")

        out_path = os.path.join(ctx.output_dir, self.output_file)
        save_json(out_path, metrics)
        ctx.scenario_metrics = metrics
        print(f"[{self.name}] Métricas salvas em: {out_path}.json")
        print(f"  Séries processadas: {len(metrics['average_speed'])}")
