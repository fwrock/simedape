# SimEDaPE

**Sim**ulation **E**stimation by **Da**ta **P**atterns **E**xploration — the approximate
simulation acceleration technique of the [Hyperbolic Time Chamber (HTC) platform](../README.md),
implemented in Python.

Given a baseline simulation's output, SimEDaPE clusters per-link time series (K-Shape), picks a
representative "simulation point" per cluster, runs a partial simulation to actually instantiate
just those points, and estimates the rest of the network's metrics from the trained
clustering/DTW (or PCA) model — trading a controlled accuracy loss for large reductions in
simulation time/resources.

Originally validated single-mode (car only) in Rocha et al. 2021 against InterSCSimulator; this
module is where multimodal estimation and alternative prediction methods are being extended for
the HTC platform.

> ⚠️ **Before installing**: `domain/dtw.py` imports `dtaidistance`, which is **not declared** in
> `pyproject.toml`'s dependencies. A clean `pip install` will fail at import time unless you
> install it manually (`pip install dtaidistance`). A `tests/` suite now covers the data-source
> readers and configuration — run it with `python3 -m unittest discover -s tests -t .`.
> See [docs/KNOWN_GAPS.md](docs/KNOWN_GAPS.md).

## Install

```bash
pip install -e .
pip install dtaidistance   # not auto-installed — see warning above
pip install -e ".[integration]"   # optional: adds aiohttp for the HTC REST connector
pip install -e ".[api]"           # optional: adds Flask for the management HTTP API
```

Requires Python ≥ 3.10.

## Quick Start

```bash
simedape --list-stages
simedape --config path/to/scenario.properties \
  --stages data_load,process_dataset,clustering,simulation_points,warping_paths,scenario_metrics,estimation,compare
```

Or run a full pipeline defined in YAML:

```bash
simedape --pipeline pipeline.yml
```

Stages run strictly in sequence, passing state through one shared `PipelineContext` — see
[docs/PIPELINE.md](docs/PIPELINE.md) for what each stage does and consumes/produces.

## Input Data

The pipeline expects per-link CSV files (`count;vehicle;time`, configurable separator) plus a
street-info CSV (`link;length;capacity;freespeed`) by default. There is also a Parquet reader
(`read_htc_parquet`, schema `tick, agent_id, agent_type, link_id, speed, travel_time,
co2_emission` — the documented HTC engine output format) but **no pipeline stage currently calls
it**; only the CSV path is wired in.

Since the HTC engine also reports per-link vehicle-flow events to ClickHouse
(`vehicle_link_events`), SimEDaPE can now read the time-series dataset **directly from
ClickHouse** instead of from files: set `data_source=clickhouse` in the configuration and provide
the `clickhouse.*` connection parameters (host/port/database/user/password/table/simulation_id).
The ClickHouse reader returns the exact same per-(link, interval) structure as the CSV reader, so
the rest of the pipeline is agnostic to the source. Communication uses the ClickHouse HTTP
interface (port 8123) with only the Python standard library — no extra driver required.
See [docs/HTC_INTEGRATION.md](docs/HTC_INTEGRATION.md) and [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Documentation

- [Pipeline & Stages](docs/PIPELINE.md) — what each of the 12 stages actually does
- [Configuration](docs/CONFIGURATION.md) — `SimEDaPEConfig` fields, `.properties`/YAML loading
- [HTTP API](docs/API.md) — the Flask server to manage configs and execute runs
- [HTC Integration](docs/HTC_INTEGRATION.md) — expected input formats, `htc_connector.py`, and its assumptions
- [Known Gaps](docs/KNOWN_GAPS.md) — missing dependency, no tests, unintegrated multimodal path
