# Pipeline & Stages

## Structure

`Pipeline` (`pipeline.py`) holds a strictly **sequential list** of `Stage` objects (no DAG). Each
stage's `execute(ctx)` reads/writes attributes directly on one shared, mutable `PipelineContext`
(`context.py`) — there's no formal input/output contract or validation between stages beyond each
stage's own `RuntimeError` checks. `Stage` (`stage.py`) is a trivial ABC (`name` + `execute`).
`Pipeline.run(progress=cb)` calls `cb(event_dict)` for `pipeline_start`, `stage_start`, `stage_end`,
`stage_failed` and `pipeline_end` (used by the HTTP API for progress; callback errors are ignored).
Every stage that loads a scenario goes through `io/source.py:read_scenario`, so `data_source`
(csv/clickhouse) applies to the partial scenario of `estimation`/`pca_estimation`/`sweep` too.

The CLI (`cli.py`) selects stages by name via `--stages a,b,c` (with optional
`--stage-params STAGE.PARAM=VALUE` overrides), or a full pipeline via `--pipeline file.yml`
(`Pipeline.from_yaml`). `simedape --list-stages` prints all registered names + first docstring
line.

## Stage Reference

| Stage | Status | What it does |
|---|---|---|
| `data_load` (`clustering.py`: `DataLoadStage`) | Implemented | Reads per-link time series from CSV (default) or ClickHouse (`data_source=clickhouse`, via `read_clickhouse_scenario`) + street-info CSV. |
| `process_dataset` (`ProcessDatasetStage`) | Implemented | Resamples + Z-normalizes time series. |
| `clustering` (`ClusteringStage`) | Implemented | K-Shape clustering (tslearn); saves centroids/labels/per-cluster datasets. |
| `simulation_points` (`simulation_points.py`) | Implemented | Finds, per cluster, the real series closest (L1) to the centroid — the point to actually simulate. Also exposes `ensure_clustering_loaded`, used by most later stages to lazy-load clustering results from disk. |
| `warping_paths` (`warping_paths.py`) | Implemented | Computes DTW warping paths (via `dtaidistance`) between each centroid and its cluster members, in parallel (`joblib`). |
| `scenario_metrics` (`metrics.py`: `ScenarioMetricsStage`) | Partially implemented | Computes core traffic metrics (speed, occupancy %, avg vehicles, km traveled) for the real/full scenario. Per-transport-mode metrics (`avg_travel_time_*`, `throughput_*`, `modal_split_*`) fall back to a single default mode — no reader currently populates a `"mode"` key in `streets_info`, so the multimodal path is best-effort/unexercised. |
| `estimation` (`estimation.py`) | Implemented | 4 DTW-based estimation methods (`avg`, `sample`, `wavg`, `base`) reconstructing full-network metrics from simulation-point data + warping paths. |
| `compare` (`compare.py`) | Implemented | Compares estimated vs. real metrics (aggregate error via geometric mean, MAPE, per-series error distribution, coverage); prints tables, saves JSON, optionally triggers plots. |
| `pca_train` / `pca_estimation` (`pca.py`) | Implemented | Alternative to DTW estimation: K-Shape clusters + per-cluster PCA trained on full data, reconstructed from partial data via least-squares in PCA weight space. Supports `sp`/`ind`/`stretched` modes plus an eigenvalue/Wiener filter and multiple interpolation stretch methods. Elaborate/experimental. |
| `sweep` (`sweep.py`) | Implemented | Orchestrates running estimation + comparison across many partial-scenario percentages and essentially every DTW/PCA method + interpolation combination, producing progression tables/plots. |
| `selective_instantiation` (`selective_instantiation.py`) | Implemented (unverified handoff) | Determines which links/routes/agent IDs (centroid series + a configurable buffer) must be kept in a partial simulation; writes `selective_instances.json`. The consumer of this file (the HTC engine / `htc-gil`) is not present in this repo — the contract is asserted only in the docstring, not verified against real consumer code here. |
| `plots.py` | Helper, not a `Stage` | Matplotlib utility (bar/box/radar/heatmap PDF reports) invoked by `compare`/`sweep`. |

Names surfaced in the CLI's help text: `data_load, process_dataset, clustering,
simulation_points, warping_paths, scenario_metrics, estimation`. `compare`, `pca_train`,
`pca_estimation`, `sweep`, and `selective_instantiation` exist in the stage registry and are
runnable, but aren't listed in the help epilog — minor doc drift in the CLI itself, not a bug in
the stages.

## Core Numerical Logic (`domain/`)

- `dtw.py` — custom DTW matrix + traceback (`warping_paths`, `best_path`, cached via `joblib`),
  with a fast path through `dtaidistance.dtw.warping_paths_fast`; `time_translation`/`estimate`
  reconstruct a full series from a partial one + a warping path using mean/std; `pruning` clips
  negative values to 0.
- `scaler.py` — `TimeSeriesScalerSimEDaPE`, per-series Z-score normalization (returns normalized
  array + per-series mean/std for later denormalization).
- `metrics.py` — the traffic-metric formulas: speed from a capacity/occupancy model
  `v = freespeed*(1-(n/capacity)^beta)^alpha`, occupancy %, cumulative km traveled, average
  vehicle count.
- `processor.py` — `process_dataset` / `process_dataset_interp` (resample + normalize, two
  interpolation strategies), `pre_processing_estimation` (label↔index lookup).
  `process_dataset_multimodal`/`_dataframe_to_series` look like scaffolding for multimodal
  grouping but aren't called by any stage yet — see [KNOWN_GAPS.md](KNOWN_GAPS.md).
