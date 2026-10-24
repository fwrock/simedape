"""Selective Instantiation Stage.

After clustering, this stage identifies which simulation entities
(links, routes, vehicles/persons) correspond to the cluster centroids
and must be included in the partial simulation run.

It writes ``selective_instances.json`` (or a configured name) with the
list of selected links, routes, and agent IDs together with the
centroid→series mapping so the HTC simulator knows exactly which
entities to keep instantiated.
"""
from __future__ import annotations

import os
import math

from pipeline.stage import Stage
from pipeline.context import PipelineContext
from pipeline.io.writer import save_json, ensure_dir
from pipeline.stages.simulation_points import ensure_clustering_loaded


def _centroid_to_series_mapping(time_series_labels: list[str], clusters: list[int]) -> dict[str, list[str]]:
    """Maps each cluster label (as string) to the list of series labels it contains."""
    mapping: dict[str, list[str]] = {}
    for idx, cluster_id in enumerate(clusters):
        key = str(cluster_id)
        if key not in mapping:
            mapping[key] = []
        mapping[key].append(time_series_labels[idx])
    return mapping


def _extract_link_ids(series_labels: list[str]) -> list[str]:
    """Extracts unique link IDs from time-series labels.

    Labels follow the convention ``<link_id>-<interval>``, e.g.
    ``12345-0-3600``.  This function returns the unique set of link IDs
    (the part before the first ``-``).
    """
    return sorted({label.split("-")[0] for label in series_labels})


def _extract_routes(simulation_points_labels: list[str]) -> list[str]:
    """Returns one route identifier per simulation point.

    A route is considered to be a direct reference to the centroid link
    series label (``<link_id>-<interval>``).  Downstream systems can use
    these to select the minimal set of routes that cover the centroids.
    """
    return list(simulation_points_labels)


def _collect_agent_ids(
    sp_labels: list[str],
    street_series_labeled_vehicles: dict[str, list[str]] | None,
    street_vehicles: dict[str, list[list[str]]] | None,
    buffer_ratio: float,
) -> list[str]:
    """Collects agent IDs for all simulation-point labels plus a buffer.

    The buffer enlarges coverage by ``buffer_ratio`` of the total unique
    agents, rounded up, sampled from non-centroid series in cluster order.

    Parameters
    ----------
    sp_labels:
        Labels of the centroid series (one per cluster).
    street_series_labeled_vehicles:
        Mapping from series label to vehicle list (populated by DataLoadStage).
    street_vehicles:
        Mapping from link ID to list-of-lists of vehicles per time interval.
    buffer_ratio:
        Fraction of total agents to add as spatial buffer (0.0 = no buffer).

    Returns
    -------
    Sorted list of unique agent ID strings.
    """
    core_agents: set[str] = set()

    if street_series_labeled_vehicles:
        for label in sp_labels:
            vehicles = street_series_labeled_vehicles.get(label, [])
            core_agents.update(vehicles)
    elif street_vehicles:
        for label in sp_labels:
            link_id = label.split("-")[0]
            for vehicle_list in street_vehicles.get(link_id, []):
                core_agents.update(vehicle_list)

    buffer_agents: set[str] = set()
    if buffer_ratio > 0.0 and street_series_labeled_vehicles:
        all_agents: set[str] = set()
        for vehicles in street_series_labeled_vehicles.values():
            all_agents.update(vehicles)

        n_buffer = math.ceil(len(all_agents) * buffer_ratio)
        # Fill buffer from agents not already in core set
        candidates = sorted(all_agents - core_agents)
        buffer_agents.update(candidates[:n_buffer])

    return sorted(core_agents | buffer_agents)


class SelectiveInstantiationStage(Stage):
    """Generates the selective configuration for the partial simulation.

    Reads clustering results from context and outputs a JSON file listing
    which links, routes, and agents correspond to cluster centroids (plus
    an optional spatial buffer).  The output file is consumed by the HTC
    simulator's selective-instantiation mechanism.

    Parameters
    ----------
    output_file : str
        Base name (without ``.json``) for the output file.
        Defaults to ``"selective_instances"``.
    include_buffer_ratio : float
        Fraction of total agents to add as a spatial buffer around the
        centroid agents.  ``0.0`` disables the buffer; ``0.1`` adds 10 %.
        Defaults to ``0.1``.
    """

    name = "selective_instantiation"

    def __init__(
        self,
        output_file: str = "selective_instances",
        include_buffer_ratio: float = 0.1,
    ) -> None:
        self.output_file = output_file
        self.include_buffer_ratio = include_buffer_ratio

    def execute(self, ctx: PipelineContext) -> None:
        ensure_clustering_loaded(ctx)

        if ctx.clustering_result is None:
            raise RuntimeError(
                "clustering_result not found in context. "
                "Run ClusteringStage (or load a prior result) before SelectiveInstantiationStage."
            )

        clustering_result = ctx.clustering_result
        clusters: list[int] = clustering_result["clusters"]
        n_clusters: int = clustering_result["number_of_clusters"]

        # Ensure simulation points (centroid series) are available
        if ctx.simulation_points_labels is None:
            raise RuntimeError(
                "simulation_points_labels not found in context. "
                "Run SimulationPointsStage before SelectiveInstantiationStage."
            )

        sp_labels: list[str] = ctx.simulation_points_labels
        time_series_labels: list[str] = ctx.time_series_labels or []

        print(f"[{self.name}] Generating selective instantiation config...")
        print(f"  Clusters: {n_clusters}, centroid series: {len(sp_labels)}")

        # --- Selected links (unique link IDs from centroid series labels) ---
        selected_links = _extract_link_ids(sp_labels)

        # --- Selected routes (one per centroid label) ---
        selected_routes = _extract_routes(sp_labels)

        # --- Selected agent IDs (centroid agents + buffer) ---
        selected_agent_ids = _collect_agent_ids(
            sp_labels=sp_labels,
            street_series_labeled_vehicles=ctx.street_series_labeled_vehicles,
            street_vehicles=ctx.street_vehicles,
            buffer_ratio=self.include_buffer_ratio,
        )

        # --- Centroid → series mapping ---
        centroid_to_series: dict[str, list[str]] = {}
        if time_series_labels:
            centroid_to_series = _centroid_to_series_mapping(time_series_labels, clusters)

        # --- Compute partial simulation ratio ---
        total_series = len(time_series_labels) if time_series_labels else 0
        partial_sim_ratio = round(n_clusters / total_series, 4) if total_series > 0 else 0.0

        output_payload = {
            "scenario_id": ctx.config.scenario_name,
            "partial_sim_ratio": partial_sim_ratio,
            "selected_links": selected_links,
            "selected_routes": selected_routes,
            "selected_agent_ids": selected_agent_ids,
            "centroid_to_series_mapping": centroid_to_series,
        }

        ensure_dir(ctx.output_dir)
        out_path = os.path.join(ctx.output_dir, self.output_file)
        save_json(out_path, output_payload)

        print(f"[{self.name}] selective_instances saved: {out_path}.json")
        print(f"  Selected links:  {len(selected_links)}")
        print(f"  Selected routes: {len(selected_routes)}")
        print(f"  Selected agents: {len(selected_agent_ids)}")
        print(f"  Partial sim ratio: {partial_sim_ratio:.2%}")
