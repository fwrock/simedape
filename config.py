from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class IntervalConfig:
    start_time: int = 0
    end_time: int = 86400
    interval_size: int = 3600

    def as_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "interval_size": self.interval_size,
        }


@dataclass
class ClusteringConfig:
    n_clusters: int = 36
    verbose: bool = True
    random_state: int = 10
    max_iter: int = 8
    output_file: str = "clustering"
    clustering_size: int = 0  # 0 = usa max_size dos dados; >0 = reamostrar para esse tamanho


@dataclass
class ClickHouseConfig:
    """Conexão e parâmetros da fonte de dados ClickHouse."""

    host: str = "localhost"
    port: int = 8123
    database: str = "htc"
    user: str = "default"
    password: str = ""
    table: str = "vehicle_link_events"
    simulation_id: str = ""
    secure: bool = False


@dataclass
class SimEDaPEConfig:
    scenario_name: str = "default"
    scenario_dir: str = ""
    street_info_file: str = ""
    csv_sep: str = ";"
    # Diretório de saída (calculado a partir do scenario_name se não fornecido)
    output_dir: str = ""
    # Diretório de entrada com clustering pré-computado
    input_dir: str = ""
    intervals: IntervalConfig = field(default_factory=IntervalConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    # Modos de transporte suportados pelo cenário.
    # Valores válidos: "car", "bike", "bus", "subway", "person".
    # Quando mais de um modo está presente, métricas são computadas por modo.
    transport_modes: list = field(default_factory=lambda: ["car"])

    # Fonte dos dados de séries temporais: "csv" (arquivos) ou "clickhouse".
    data_source: str = "csv"
    # Conexão ClickHouse — usada apenas quando data_source == "clickhouse".
    clickhouse: ClickHouseConfig = field(default_factory=ClickHouseConfig)

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = f"output/{self.scenario_name}/"

    @classmethod
    def from_properties_file(cls, path: str) -> "SimEDaPEConfig":
        from jproperties import Properties

        props = Properties()
        with open(path, "rb") as f:
            props.load(f)

        def get(key, default=""):
            try:
                return props.get(key).data
            except Exception:
                return default

        config = cls.__new__(cls)
        config.scenario_name = get("scenario.name", "default")
        config.scenario_dir = get("scenario.single.dir.files", "")
        config.street_info_file = get("scenario.single.dir.street.info", "")
        config.csv_sep = get("csv.sep", ";")
        config.input_dir = get("scneario.input.dir", "")  # typo original preservado
        config.output_dir = f"output/{config.scenario_name}/"
        config.intervals = IntervalConfig(
            start_time=int(get("intervals.start_time", 0)),
            end_time=int(get("intervals.end_time", 86400)),
            interval_size=int(get("intervals.interval_size", 3600)),
        )
        config.clustering = ClusteringConfig(
            n_clusters=int(get("clustering.n_clusters", 36)),
            verbose=get("clustering.verbose", "True") == "True",
            random_state=int(get("clustering.random_state", 10)),
            max_iter=int(get("clustering.max_iter", 8)),
            output_file=get("clustering.output.save.file", "clustering"),
            clustering_size=int(get("clustering.clustering_size", 0)),
        )
        modes_raw = get("transport.modes", "car")
        config.transport_modes = [m.strip() for m in modes_raw.split(",") if m.strip()]
        config.data_source = get("data.source", "csv")
        config.clickhouse = ClickHouseConfig(
            host=get("clickhouse.host", "localhost"),
            port=int(get("clickhouse.port", 8123)),
            database=get("clickhouse.database", "htc"),
            user=get("clickhouse.user", "default"),
            password=get("clickhouse.password", ""),
            table=get("clickhouse.table", "vehicle_link_events"),
            simulation_id=get("clickhouse.simulation_id", ""),
            secure=get("clickhouse.secure", "False") == "True",
        )
        return config

    @classmethod
    def from_dict(cls, cfg: dict) -> "SimEDaPEConfig":
        """Constrói a config a partir de um dict (mesmas chaves do YAML ``config:``)."""
        config = cls()
        config.scenario_name = cfg.get("scenario_name", "default")
        config.scenario_dir = cfg.get("scenario_dir", "")
        config.street_info_file = cfg.get("street_info_file", "")
        config.csv_sep = cfg.get("csv_sep", ";")
        config.input_dir = cfg.get("input_dir", "")
        config.output_dir = cfg.get("output_dir", f"output/{config.scenario_name}/")
        iv = cfg.get("intervals", {}) or {}
        config.intervals = IntervalConfig(
            start_time=iv.get("start_time", 0),
            end_time=iv.get("end_time", 86400),
            interval_size=iv.get("interval_size", 3600),
        )
        cl = cfg.get("clustering", {}) or {}
        config.clustering = ClusteringConfig(
            n_clusters=cl.get("n_clusters", 36),
            verbose=cl.get("verbose", True),
            random_state=cl.get("random_state", 10),
            max_iter=cl.get("max_iter", 8),
            output_file=cl.get("output_file", "clustering"),
            clustering_size=cl.get("clustering_size", 0),
        )
        config.transport_modes = cfg.get("transport_modes", ["car"])
        config.data_source = cfg.get("data_source", "csv")
        ch = cfg.get("clickhouse", {}) or {}
        config.clickhouse = ClickHouseConfig(
            host=ch.get("host", "localhost"),
            port=ch.get("port", 8123),
            database=ch.get("database", "htc"),
            user=ch.get("user", "default"),
            password=ch.get("password", ""),
            table=ch.get("table", "vehicle_link_events"),
            simulation_id=ch.get("simulation_id", ""),
            secure=ch.get("secure", False),
        )
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "SimEDaPEConfig":
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("config", {}) or {})

    def to_dict(self) -> dict:
        """Serializa a config para um dict (inverso de :meth:`from_dict`)."""
        return {
            "scenario_name": self.scenario_name,
            "scenario_dir": self.scenario_dir,
            "street_info_file": self.street_info_file,
            "csv_sep": self.csv_sep,
            "output_dir": self.output_dir,
            "input_dir": self.input_dir,
            "intervals": self.intervals.as_dict(),
            "clustering": {
                "n_clusters": self.clustering.n_clusters,
                "verbose": self.clustering.verbose,
                "random_state": self.clustering.random_state,
                "max_iter": self.clustering.max_iter,
                "output_file": self.clustering.output_file,
                "clustering_size": self.clustering.clustering_size,
            },
            "transport_modes": self.transport_modes,
            "data_source": self.data_source,
            "clickhouse": {
                "host": self.clickhouse.host,
                "port": self.clickhouse.port,
                "database": self.clickhouse.database,
                "user": self.clickhouse.user,
                "password": self.clickhouse.password,
                "table": self.clickhouse.table,
                "simulation_id": self.clickhouse.simulation_id,
                "secure": self.clickhouse.secure,
            },
        }
