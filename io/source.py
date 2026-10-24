"""Seleção da fonte de dados das séries temporais (CSV ou ClickHouse).

Todos os estágios que carregam um cenário — o de treino em ``data_load`` e o
cenário parcial em ``estimation``/``pca_estimation``/``sweep`` — passam por
``read_scenario``, de modo que ``config.data_source`` vale para todos eles.

O parâmetro ``location`` identifica *qual* cenário ler dentro da fonte:

* ``csv``        — diretório com os arquivos ``*.csv`` (``scenario_dir``);
* ``clickhouse`` — ``simulation_id`` na tabela de eventos.

Vazio significa o cenário de treino definido na config.
"""
from __future__ import annotations

from pipeline.io.clickhouse import ClickHouseClient, read_clickhouse_scenario
from pipeline.io.reader import read_single_scenario

DATA_SOURCES = ("csv", "clickhouse")


def _check_source(cfg) -> str:
    if cfg.data_source not in DATA_SOURCES:
        raise ValueError(
            f"data_source inválido: {cfg.data_source!r}. Válidos: {list(DATA_SOURCES)}"
        )
    return cfg.data_source


def build_clickhouse_client(cfg) -> ClickHouseClient:
    ch = cfg.clickhouse
    return ClickHouseClient(
        host=ch.host,
        port=ch.port,
        database=ch.database,
        user=ch.user,
        password=ch.password,
        secure=ch.secure,
    )


def training_location(cfg) -> str:
    """Identificador do cenário de treino na fonte configurada."""
    if _check_source(cfg) == "clickhouse":
        return cfg.clickhouse.simulation_id
    return cfg.scenario_dir


def read_scenario(cfg, location: str = ""):
    """Lê um cenário da fonte configurada e devolve a mesma 6-upla dos readers.

    ``location`` vazio usa o cenário de treino (``training_location``).
    """
    if _check_source(cfg) == "clickhouse":
        ch = cfg.clickhouse
        return read_clickhouse_scenario(
            build_clickhouse_client(cfg),
            intervals=cfg.intervals.as_dict(),
            simulation_id=location or ch.simulation_id,
            table=ch.table,
        )
    return read_single_scenario(
        location or cfg.scenario_dir,
        sep=cfg.csv_sep,
        intervals=cfg.intervals.as_dict(),
    )


def describe_source(cfg) -> dict:
    """Resumo da fonte para exibição/persistência (nunca inclui a senha)."""
    if _check_source(cfg) == "clickhouse":
        ch = cfg.clickhouse
        return {
            "type": "clickhouse",
            "host": ch.host,
            "port": ch.port,
            "database": ch.database,
            "table": ch.table,
            "simulation_id": ch.simulation_id,
        }
    return {"type": "csv", "scenario_dir": cfg.scenario_dir}
