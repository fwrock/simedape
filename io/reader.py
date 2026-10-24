import glob
import numpy as np
import pandas as pd

# HTC Parquet output columns
_HTC_PARQUET_COLUMNS = [
    "tick",
    "agent_id",
    "agent_type",
    "link_id",
    "speed",
    "travel_time",
    "co2_emission",
]

# Map HTC agent_type values to SimEDaPE transport_modes labels
_HTC_AGENT_TYPE_TO_MODE: dict[str, str] = {
    "car": "car",
    "Car": "car",
    "CAR": "car",
    "bike": "bike",
    "Bike": "bike",
    "BIKE": "bike",
    "bus": "bus",
    "Bus": "bus",
    "BUS": "bus",
    "subway": "subway",
    "Subway": "subway",
    "SUBWAY": "subway",
    "person": "person",
    "Person": "person",
    "PERSON": "person",
    "pedestrian": "person",
    "Pedestrian": "person",
}


def _read_csv(data_file, sep=";", dtype=None, usecols=None):
    if dtype is None:
        dtype = {"link": int, "length": float, "capacity": float, "freespeed": float}
    if usecols is None:
        usecols = ["link", "length", "capacity", "freespeed"]
    return pd.read_csv(
        data_file,
        sep=sep,
        engine="python",
        quotechar="'",
        dtype=dtype,
        usecols=usecols,
    )


def read_street_info(data_file, sep=";"):
    """Carrega informações de cada via: id, comprimento, capacidade, velocidade livre.

    Retorna
    -------
    dict: {link_id_str -> [length, capacity, freespeed]}
    """
    streets_info = {}
    df = _read_csv(data_file, sep=sep)
    for _, row in df.iterrows():
        streets_info[str(int(row["link"]))] = [
            float(row["length"]),
            float(row["capacity"]),
            float(row["freespeed"]),
        ]
    return streets_info


def _format_interval_label(link_id, interval):
    """Constrói o label ``<link_id>-<start>-<end>`` a partir de um intervalo.

    O mesmo formato é usado pelas fontes CSV e ClickHouse para que o restante
    do pipeline seja agnóstico quanto à origem dos dados.
    """
    return str(link_id) + "-" + str(interval).replace("(", "").replace("]", "").replace(", ", "-")


def _get_time_series_label(name, interval):
    return _format_interval_label(name[0], interval)


def read_single_scenario(dir_files, sep=";", intervals=None):
    """Lê todos os arquivos CSV de um cenário e agrupa as séries temporais
    por intervalos de tempo.

    Parâmetros
    ----------
    dir_files : str
        Diretório com os arquivos .csv (um por via).
    sep : str
        Separador CSV.
    intervals : dict
        {'start_time': int, 'end_time': int, 'interval_size': int}

    Retorna
    -------
    max_size : int
    street_series : list[list[float]]
    street_series_vehicles : list[list[str]]
    street_vehicles : dict[str, list[list[str]]]
    time_series_labels : list[str]
    street_series_labeled_vehicles : dict[str, list[str]]
    """
    if intervals is None:
        intervals = {"start_time": 0, "end_time": 86400, "interval_size": 3600}

    max_size = -1
    street_series = []
    street_series_vehicles = []
    street_series_labeled_vehicles = {}
    street_vehicles = {}
    time_series_labels = []

    bins = np.arange(
        intervals["start_time"],
        intervals["end_time"] + intervals["interval_size"],
        intervals["interval_size"],
    )

    for data_file in glob.glob(dir_files + "*.csv"):
        df = _read_csv(
            data_file,
            sep=sep,
            dtype={"time": int, "vehicle": str, "count": int},
            usecols=["count", "vehicle", "time"],
        )
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        df.dropna(subset=["count"], inplace=True)

        name = str(data_file).replace(dir_files, "").split("-")
        groups = df.groupby(pd.cut(df["time"], bins), observed=False)

        for interval_name, group in groups:
            s = group["count"].tolist()
            v = group["vehicle"].tolist()

            street_id = name[0]
            if street_id not in street_vehicles:
                street_vehicles[street_id] = []
            street_vehicles[street_id].append(v)

            if len(s) > 1:
                street_series.append(s)
                street_series_vehicles.append(v)
                label = _get_time_series_label(name, interval_name)
                street_series_labeled_vehicles[label] = v
                time_series_labels.append(label)
                if max_size < len(s):
                    max_size = len(s)

    return (
        max_size,
        street_series,
        street_series_vehicles,
        street_vehicles,
        time_series_labels,
        street_series_labeled_vehicles,
    )


def read_htc_parquet(path: str) -> "pd.DataFrame":
    """Reads an HTC simulator Parquet output file into a normalised DataFrame.

    The HTC simulator writes Parquet files with the following columns::

        tick          int64   — simulation tick (seconds)
        agent_id      str     — unique agent identifier
        agent_type    str     — vehicle/pedestrian type (car, bike, bus, subway, person)
        link_id       str     — road-network link identifier
        speed         float64 — instantaneous speed (m/s)
        travel_time   float64 — cumulative travel time for this agent (s)
        co2_emission  float64 — CO₂ emitted this tick (g)

    Post-processing applied:
    - ``agent_type`` is mapped to canonical SimEDaPE mode labels via
      ``_HTC_AGENT_TYPE_TO_MODE`` (e.g. ``"Car"`` → ``"car"``).
    - A ``mode`` column is added as an alias for the normalised
      ``agent_type`` values, for compatibility with
      :func:`pipeline.domain.processor.process_dataset_multimodal`.
    - Missing expected columns are silently filled with ``NaN`` / empty
      strings so downstream code can rely on the schema.

    Parameters
    ----------
    path : str
        Path to the ``.parquet`` file (or a directory when partitioned output
        is used — ``pandas.read_parquet`` handles both transparently).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns as described above, plus ``mode``.

    Raises
    ------
    ImportError
        If ``pyarrow`` (or ``fastparquet``) is not installed.
    FileNotFoundError
        If ``path`` does not exist.
    """
    try:
        import pyarrow  # noqa: F401 — validate optional dependency
    except ImportError as exc:
        raise ImportError(
            "read_htc_parquet requires 'pyarrow'. "
            "Install it with: pip install pyarrow"
        ) from exc

    df: pd.DataFrame = pd.read_parquet(path)

    # Ensure all expected columns are present (fill missing with NaN / "")
    for col in _HTC_PARQUET_COLUMNS:
        if col not in df.columns:
            if col in ("agent_id", "agent_type", "link_id"):
                df[col] = ""
            else:
                df[col] = float("nan")

    # Normalise agent_type to canonical mode label
    df["agent_type"] = df["agent_type"].map(
        lambda t: _HTC_AGENT_TYPE_TO_MODE.get(str(t), str(t))
    )

    # Add convenience alias column
    df["mode"] = df["agent_type"]

    # Coerce numeric types
    for numeric_col in ("tick", "speed", "travel_time", "co2_emission"):
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    return df
