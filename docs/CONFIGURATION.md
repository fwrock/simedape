# Configuration

`SimEDaPEConfig` (`config.py`) is a dataclass with:

| Field | Purpose |
|---|---|
| `scenario_name`, `scenario_dir` | Identify and locate the scenario's input data. |
| `street_info_file` | CSV with `link;length;capacity;freespeed`. |
| `csv_sep` | Separator for per-link CSVs (default `;`). |
| `output_dir`, `input_dir` | Where results are written / read from. |
| `intervals` | `start`, `end`, `interval_size` (seconds) — the time-binning window for series construction. |
| `clustering` | `n_clusters`, `verbose`, `random_state`, `max_iter`, `output_file`, `clustering_size`. |
| `transport_modes` | List, default `["car"]` — see the multimodal caveat in [PIPELINE.md](PIPELINE.md#scenario_metrics) and [KNOWN_GAPS.md](KNOWN_GAPS.md). |
| `data_source` | `"csv"` (default) or `"clickhouse"` — which time-series source is read — by `DataLoadStage` and by the partial scenario of `estimation`, `pca_estimation` and `sweep` (there, `second_scenario_dir` / the sweep `dir` is a directory for csv and a `simulation_id` for clickhouse). |
| `clickhouse` | `ClickHouseConfig` — `host`, `port`, `database`, `user`, `password`, `table`, `simulation_id`, `secure`. Used only when `data_source == "clickhouse"`. |

## ClickHouse source

When `data_source = "clickhouse"`, the pipeline reads the time-series dataset directly from the
HTC engine's ClickHouse `vehicle_link_events` table (default `htc.vehicle_link_events`) instead of
from CSV files. Keys:

- **`.properties`**: `data.source`, `clickhouse.host`, `clickhouse.port`, `clickhouse.database`,
  `clickhouse.user`, `clickhouse.password`, `clickhouse.table`, `clickhouse.simulation_id`,
  `clickhouse.secure`.
- **YAML**:
  ```yaml
  config:
    data_source: clickhouse
    clickhouse:
      host: localhost
      port: 8123
      database: htc
      user: default
      password: ""
      table: vehicle_link_events
      simulation_id: ""   # optional — filter by a single simulation run
      secure: false
  ```

`simulation_id` may be left empty to load **all** simulations in the table; set it to restrict to
one run. The street-info file (`street_info_file`) is still read from CSV — the ClickHouse source
covers the time-series dataset only.

## Loading

- **`.properties`** via `jproperties` — note the config key `scneario.input.dir` preserves a
  historical typo; use it as-is, don't "fix" it without also updating every `.properties` file
  that references it.
- **YAML** — used both for `SimEDaPEConfig` and for full `Pipeline.from_yaml` definitions (stage
  list + per-stage params).

No sample `.properties` or `.yml` files ship in this repo — you must author your own scenario
config before running the CLI.
