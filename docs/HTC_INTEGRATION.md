# HTC Integration

## Input Formats (`io/reader.py`)

- **CSV path (actually wired into the pipeline)**: globs `*.csv` in a scenario directory
  (`;`-separated by default), columns `count/vehicle/time`, binned into time intervals to build
  labeled time series + vehicle-id lists. `read_street_info` reads the companion
  `link;length;capacity;freespeed` CSV.
- **Parquet path (`read_htc_parquet`, not yet called by any stage)**: reads Parquet files with
  schema `tick, agent_id, agent_type, link_id, speed, travel_time, co2_emission` — documented as
  the actual HTC engine (`hyperbolic-time-chamber`) output format — and normalizes `agent_type`
  into canonical mode labels. If you want simedape to consume real HTC engine output directly
  (rather than a CSV export), you'd currently need to add a stage that calls this function; none
  exists yet.

`io/writer.py` only writes JSON (`save_json`/`load_json`, auto-appends `.json`) — there is no
CSV/Parquet writer in this module.

## ClickHouse Source (`io/clickhouse.py`)

`read_clickhouse_scenario` reads the time-series dataset directly from ClickHouse and returns the
exact same 6-tuple as `read_single_scenario`, so downstream stages don't care where the data came
from:

- Reads the HTC engine's `vehicle_link_events` table (`link_id, tick, vehicle_id,
  vehicle_count_on_link`), filters by `simulation_id` (optional) and the configured
  `[start_time, end_time]`, and bins `tick` into the same intervals as the CSV reader
  (`pd.cut` + `np.arange`).
- Each `(link, interval)` series uses `vehicle_count_on_link` (instantaneous vehicle count
  reported by the engine per event) as the value and `vehicle_id` as the vehicle list — the direct
  analogues of the CSV `count`/`vehicle` columns.
- `ClickHouseClient` is a minimal stdlib (`urllib`) client speaking the **ClickHouse HTTP
  interface** (port 8123, headers `X-ClickHouse-User` / `X-ClickHouse-Key`), mirroring the
  engine's own `ClickHouseClientManager.scala`. Values are bound server-side via `param_*` query
  parameters to avoid SQL injection; table names are validated against a safe identifier pattern.

`DataLoadStage` selects the source via `config.data_source` (`"csv"` or `"clickhouse"`).

## `integration/htc_connector.py`

Two classes, both requiring the optional `aiohttp` dependency (`pip install -e ".[integration]"`):

- **`HtcSimulationConnector`** — calls a presumed HTC engine REST API over HTTP:
  `POST {base_url}/api/simulation/start`, then polls
  `GET {base_url}/api/simulation/status/{run_id}` until `"completed"`/`"failed"`.
- **`HtcResultFetcher`** — reads an assumed output directory layout,
  `<output_dir>/<scenario_id>/<run_id>/*.parquet` + `summary.json`, via `read_htc_parquet`.

**Important caveat**: this assumes an HTC REST API server exists at those exact endpoints.
Nothing elsewhere in this repo confirms that server exists or matches this contract — treat
`htc_connector.py` as a **speculative, forward-looking client integration point**, not a verified
one. Before relying on it, confirm against the actual `hyperbolic-time-chamber` REST API
(`hyperbolic-time-chamber/docs/API.md`) whether the endpoint paths and status semantics match.

## Selective Instantiation Handoff

`stages/selective_instantiation.py` writes `selective_instances.json`, intended to tell an
external scenario-generation service (presumably `htc-gil`'s generator, given the
`dev.htc.simedape.selective-instantiation.v1` Kafka topic naming referenced there) which
links/routes/agents to include in the next partial simulation run. No consumer of this file exists
in this repo — verify the JSON shape against whatever actually reads it (see
[`htc-gil`'s docs](../../htc-gil/docs/ARCHITECTURE.md)) before changing its structure.
