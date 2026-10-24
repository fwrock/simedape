"""Fonte de dados ClickHouse para o dataset de séries temporais do SimEDaPE.

O motor de simulação HTC (``hyperbolic-time-chamber``) reporta, por via
(link), eventos de fluxo de veículos para o ClickHouse (tabela
``vehicle_link_events`` — ver ``ClickHouseClientManager.scala``). Este módulo
lê esses eventos de volta e reconstrói exatamente a mesma estrutura
``(link, intervalo de tempo)`` produzida por
:func:`pipeline.io.reader.read_single_scenario`, de modo que o restante do
pipeline (clustering, estimação, …) permaneça agnóstico quanto à origem dos
dados.

A comunicação usa a **interface HTTP** do ClickHouse (porta padrão ``8123``),
o mesmo canal usado pelo próprio motor. Apenas a biblioteca padrão do Python é
necessária — não há dependência de driver ClickHouse de terceiros.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from pipeline.io.reader import _format_interval_label

# Colunas do fluxo de veículos lidas da tabela bruta ``vehicle_link_events``.
_VEHICLE_FLOW_COLUMNS = [
    "link_id",
    "tick",
    "vehicle_id",
    "vehicle_count_on_link",
]

# Aceita nomes de tabela simples ou qualificados
# (ex.: ``vehicle_link_events`` ou ``htc.vehicle_link_events``).
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.]+$")


class ClickHouseError(RuntimeError):
    """Erro ao consultar o ClickHouse (HTTP, parse ou validação)."""


class ClickHouseClient:
    """Cliente HTTP mínimo do ClickHouse usado pela fonte de dados.

    Parâmetros
    ----------
    host : str
        Host do servidor ClickHouse (padrão ``"localhost"``).
    port : int
        Porta da interface HTTP (padrão ``8123``).
    database : str
        Banco de dados padrão usado nas consultas (padrão ``"htc"``).
    user : str
        Usuário (enviado no header ``X-ClickHouse-User``).
    password : str
        Senha (enviada no header ``X-ClickHouse-Key``).
    secure : bool
        Se ``True``, usa ``https`` em vez de ``http``.
    timeout : float
        Timeout das requisições em segundos.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "htc",
        user: str = "default",
        password: str = "",
        secure: bool = False,
        timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.database = database
        self.user = user
        self.password = password
        self.secure = secure
        self.timeout = timeout

    @property
    def base_url(self) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.host}:{self.port}/"

    def query(self, sql: str, params: dict | None = None) -> "pd.DataFrame":
        """Executa um ``SELECT`` e retorna o resultado como DataFrame.

        Os valores em ``params`` são enviados como parâmetros ``param_*`` da
        interface HTTP e substituídos pelo próprio ClickHouse (evita injeção de
        SQL em valores).
        """
        params = params or {}
        body = self._fetch(self._ensure_format(sql), params)
        return self._parse(body)

    def ping(self) -> bool:
        """Retorna ``True`` se o servidor respondeu a um ``SELECT 1``."""
        try:
            self._fetch("SELECT 1 FORMAT JSONCompact", {})
            return True
        except (ClickHouseError, OSError):
            return False

    @staticmethod
    def _ensure_format(sql: str) -> str:
        sql = sql.strip().rstrip(";").strip()
        if not re.search(r"\bFORMAT\b", sql, flags=re.IGNORECASE):
            sql = f"{sql} FORMAT JSONCompact"
        return sql

    def _fetch(self, sql: str, params: dict) -> str:
        query = {"query": sql}
        if self.database:
            query["database"] = self.database
        for key, value in params.items():
            query[f"param_{key}"] = str(value)

        url = f"{self.base_url}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, method="GET")
        if self.user:
            request.add_header("X-ClickHouse-User", self.user)
        if self.password:
            request.add_header("X-ClickHouse-Key", self.password)
        request.add_header("Accept", "application/json")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise ClickHouseError(
                f"ClickHouse retornou HTTP {exc.code}: {detail.strip()}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ClickHouseError(
                f"Falha de conexão com o ClickHouse: {exc.reason}"
            ) from exc

    @staticmethod
    def _parse(body: str) -> "pd.DataFrame":
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ClickHouseError(
                f"ClickHouse retornou resposta não-JSON: {body[:200]!r}"
            ) from exc

        if "data" not in payload:
            message = payload.get("exception") or body[:200]
            raise ClickHouseError(f"Erro na consulta ClickHouse: {message}")

        columns = [column["name"] for column in payload.get("meta", [])]
        return pd.DataFrame(payload["data"], columns=columns)


def build_vehicle_flow_query(
    intervals: dict,
    simulation_id: str = "",
    table: str = "vehicle_link_events",
) -> tuple[str, dict]:
    """Constrói o SQL (com placeholders ``{param:Type}``) e os parâmetros.

    Retorna
    -------
    (sql, params)
        SQL para o ClickHouse e o dict de parâmetros a serem enviados como
        ``param_*`` via HTTP.
    """
    if not _IDENTIFIER_RE.match(table):
        raise ValueError(f"Nome de tabela ClickHouse inválido: {table!r}")

    start = intervals["start_time"]
    end = intervals["end_time"]
    columns = ", ".join(_VEHICLE_FLOW_COLUMNS)
    sql = (
        f"SELECT {columns} FROM {table} "
        "WHERE tick >= {start_time:UInt64} AND tick <= {end_time:UInt64}"
    )
    params: dict = {"start_time": int(start), "end_time": int(end)}
    if simulation_id:
        sql += " AND simulation_id = {simulation_id:String}"
        params["simulation_id"] = str(simulation_id)
    sql += " ORDER BY link_id, tick"
    return sql, params

def read_clickhouse_scenario(
    client: ClickHouseClient,
    intervals: dict | None = None,
    simulation_id: str = "",
    table: str = "vehicle_link_events",
) -> tuple:
    """Lê o cenário do ClickHouse e retorna a mesma estrutura do CSV.

    A assinatura e o retorno são idênticos a
    :func:`pipeline.io.reader.read_single_scenario`:

    Retorna
    -------
    max_size : int
    street_series : list[list[float]]
    street_series_vehicles : list[list[str]]
    street_vehicles : dict[str, list[list[str]]]
    time_series_labels : list[str]
    street_series_labeled_vehicles : dict[str, list[str]]

    A série de cada ``(link, intervalo)`` usa ``vehicle_count_on_link``
    (contagem instantânea de veículos reportada pelo motor a cada evento) como
    valor, e ``vehicle_id`` como lista de veículos — os análogos diretos das
    colunas ``count``/``vehicle`` do CSV.
    """
    if intervals is None:
        intervals = {"start_time": 0, "end_time": 86400, "interval_size": 3600}

    sql, params = build_vehicle_flow_query(intervals, simulation_id, table)
    df = client.query(sql, params)

    required = set(_VEHICLE_FLOW_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ClickHouseError(
            f"Resultado do ClickHouse sem as colunas: {sorted(missing)}"
        )

    df = df.copy()
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce")
    df["vehicle_count_on_link"] = pd.to_numeric(
        df["vehicle_count_on_link"], errors="coerce"
    )
    df["vehicle_id"] = df["vehicle_id"].astype(str)
    df = df.dropna(subset=["tick", "vehicle_count_on_link"])

    start = intervals["start_time"]
    end = intervals["end_time"]
    size = intervals["interval_size"]
    bins = np.arange(start, end + size, size)

    max_size = -1
    street_series: list = []
    street_series_vehicles: list = []
    street_series_labeled_vehicles: dict = {}
    street_vehicles: dict = {}
    time_series_labels: list = []

    for link_id, link_group in df.groupby("link_id", sort=True):
        sid = str(link_id)
        for interval_name, group in link_group.groupby(
            pd.cut(link_group["tick"], bins), observed=False
        ):
            if group.empty:
                continue
            s = group["vehicle_count_on_link"].astype(float).tolist()
            v = group["vehicle_id"].tolist()

            if sid not in street_vehicles:
                street_vehicles[sid] = []
            street_vehicles[sid].append(v)

            if len(s) > 1:
                street_series.append(s)
                street_series_vehicles.append(v)
                label = _format_interval_label(sid, interval_name)
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


