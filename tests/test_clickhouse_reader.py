"""Testes do leitor de cenário ClickHouse (build de query + reconstrução)."""
import unittest

import pandas as pd

import tests._bootstrap  # noqa: F401
from pipeline.io.clickhouse import (
    ClickHouseError,
    build_vehicle_flow_query,
    read_clickhouse_scenario,
)
from pipeline.io.reader import _format_interval_label, _get_time_series_label


class FakeClient:
    """Cliente duck-typed que devolve um DataFrame fixo e grava as queries."""

    def __init__(self, df):
        self.df = df
        self.queries = []

    def query(self, sql, params=None):
        self.queries.append((sql, params or {}))
        return self.df


INTERVALS = {"start_time": 0, "end_time": 7200, "interval_size": 3600}


class TestBuildQuery(unittest.TestCase):
    def test_defaults(self):
        sql, params = build_vehicle_flow_query(INTERVALS)
        self.assertIn("FROM vehicle_link_events", sql)
        self.assertIn("{start_time:UInt64}", sql)
        self.assertIn("{end_time:UInt64}", sql)
        self.assertEqual(params, {"start_time": 0, "end_time": 7200})
        self.assertNotIn("simulation_id", params)

    def test_with_simulation_id(self):
        sql, params = build_vehicle_flow_query(INTERVALS, simulation_id="run9")
        self.assertIn("simulation_id = {simulation_id:String}", sql)
        self.assertEqual(params["simulation_id"], "run9")

    def test_invalid_table_raises(self):
        with self.assertRaises(ValueError):
            build_vehicle_flow_query(INTERVALS, table="vehicle_link_events; DROP TABLE x")


class TestReadScenario(unittest.TestCase):
    def test_roundtrip(self):
        df = pd.DataFrame(
            {
                "link_id": ["3", "3", "3", "5"],
                "tick": [100, 3700, 3800, 100],
                "vehicle_id": ["v1", "v2", "v3", "v4"],
                "vehicle_count_on_link": [1, 2, 3, 1],
            }
        )
        client = FakeClient(df)
        (
            max_size,
            street_series,
            street_series_vehicles,
            street_vehicles,
            labels,
            labeled_vehicles,
        ) = read_clickhouse_scenario(client, intervals=INTERVALS)

        self.assertEqual(max_size, 2)
        self.assertEqual(street_series, [[2.0, 3.0]])
        self.assertEqual(street_series_vehicles, [["v2", "v3"]])
        self.assertEqual(labels, ["3-3600-7200"])
        self.assertEqual(labeled_vehicles, {"3-3600-7200": ["v2", "v3"]})
        # street_vehicles inclui todos os bins não-vazios (mesmo de 1 elemento),
        # mas nunca bins vazios.
        self.assertEqual(street_vehicles["3"], [["v1"], ["v2", "v3"]])
        self.assertEqual(street_vehicles["5"], [["v4"]])

        # A query enviada deve conter start/end (sem simulation_id).
        _, params = client.queries[0]
        self.assertEqual(params["start_time"], 0)
        self.assertEqual(params["end_time"], 7200)
        self.assertNotIn("simulation_id", params)

    def test_forwards_simulation_id(self):
        df = pd.DataFrame(
            {
                "link_id": ["3", "3"],
                "tick": [100, 200],
                "vehicle_id": ["v1", "v2"],
                "vehicle_count_on_link": [1, 2],
            }
        )
        client = FakeClient(df)
        read_clickhouse_scenario(
            client, intervals=INTERVALS, simulation_id="run42"
        )
        _, params = client.queries[0]
        self.assertEqual(params["simulation_id"], "run42")

    def test_empty_result(self):
        df = pd.DataFrame(
            columns=["link_id", "tick", "vehicle_id", "vehicle_count_on_link"]
        )
        client = FakeClient(df)
        result = read_clickhouse_scenario(client, intervals=INTERVALS)
        max_size, street_series, _, street_vehicles, labels, _ = result
        self.assertEqual(max_size, -1)
        self.assertEqual(street_series, [])
        self.assertEqual(labels, [])
        self.assertEqual(street_vehicles, {})

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"link_id": ["3"], "tick": [100]})
        client = FakeClient(df)
        with self.assertRaises(ClickHouseError):
            read_clickhouse_scenario(client, intervals=INTERVALS)


class TestLabel(unittest.TestCase):
    def test_format_interval_label(self):
        interval = pd.Interval(3600, 7200, closed="right")
        self.assertEqual(_format_interval_label("3", interval), "3-3600-7200")

    def test_get_time_series_label_keeps_csv_behaviour(self):
        interval = pd.Interval(3600, 7200, closed="right")
        self.assertEqual(_get_time_series_label(["3"], interval), "3-3600-7200")


if __name__ == "__main__":
    unittest.main()
