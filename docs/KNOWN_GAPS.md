# Known Gaps

## Packaging

- `dtaidistance` and `joblib` are now declared in `pyproject.toml` (previously `dtaidistance` was missing
  and `joblib` only arrived transitively via `scikit-learn`). The build backend is
  `setuptools.build_meta`, and the sub-packages are listed explicitly under `[tool.setuptools]`
  (`packages.find` could not discover the `pipeline` mapping, so earlier wheels were empty).
- Packaging uses a deliberate trick: `[tool.setuptools.package-dir] pipeline = "."` maps the
  `simedape/` directory itself to the Python package name `pipeline`, and all internal imports use
  `from pipeline.X import Y` (explained in a comment in `pyproject.toml`, lines 27-34). This is
  intentional, not a bug — don't "fix" the imports to `from simedape.X` without also fixing the
  packaging config, or the CLI entry point (`simedape = "pipeline.cli:main"`) will break.

## Testing

A `tests/` suite now covers the data-source readers, configuration, and the HTTP API:

- `tests/test_clickhouse_client.py` — the ClickHouse HTTP client (formatting, JSON parsing,
  URL/header construction, ping, error handling).
- `tests/test_clickhouse_reader.py` — `build_vehicle_flow_query` and `read_clickhouse_scenario`
  (structure parity with the CSV reader, empty/missing-column edge cases).
- `tests/test_config.py` — `ClickHouseConfig` defaults, YAML/`.properties` loading, and
  `from_dict`/`to_dict` round-trip.
- `tests/test_store.py` / `tests/test_jobs.py` — the API `ConfigStore` and `JobManager`.
- `tests/test_api.py` — the Flask endpoints (config CRUD, stages, run lifecycle) via the test
  client, skipped when Flask isn't installed.

Run with `python3 -m unittest discover -s tests -t .`. The ClickHouse client uses the standard
library (HTTP mocked in tests); the API tests additionally need Flask (`pip install -e ".[test]"`).
The heavier numerical paths (`domain/dtw.py`, estimation, PCA) still have no automated coverage and
should be manually validated against a known scenario (compare stage output) before being
considered safe.

## Unintegrated / Stubbed Paths

- **Multimodal metrics** (`stages/metrics.py`) fall back to a single default transport mode — no
  reader currently populates a `"mode"` key in `streets_info`, so the per-mode metrics
  (`avg_travel_time_*`, `throughput_*`, `modal_split_*`) are best-effort and not exercised
  end-to-end by any current pipeline run.
- **`domain/processor.py`'s multimodal helpers** (`process_dataset_multimodal`,
  `_dataframe_to_series`) exist but aren't called by any stage — scaffolding for a feature not yet
  wired in. This is directly relevant to this work's stated goal of extending SimEDaPE beyond
  single-mode (car-only) estimation — start here rather than from scratch.
- **`read_htc_parquet`** exists and matches the documented HTC engine output schema, but no stage
  calls it — the Parquet ingestion path remains unwired (the CSV and ClickHouse paths are wired).
- **ClickHouse time-series source** is now wired in: set `data_source=clickhouse` plus the
  `clickhouse.*` connection parameters — `DataLoadStage` reads `vehicle_link_events` via
  `io/clickhouse.py` and returns the same structure as the CSV reader.
- **`integration/htc_connector.py`** assumes an HTC engine REST API that isn't verified to exist
  in this repo set — see [HTC_INTEGRATION.md](HTC_INTEGRATION.md).

## Practical Implication

The single-mode (car), CSV/ClickHouse-driven, DTW/PCA-based estimation pipeline is solid and fully
implemented end to end. The platform-level ambitions this module is meant to serve — multimodal
estimation and direct HTC engine (Parquet/REST) integration — have partial scaffolding but are not
yet connected into the runnable pipeline.
