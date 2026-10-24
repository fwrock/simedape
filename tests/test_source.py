"""Testes do seletor de fonte de dados (CSV x ClickHouse)."""
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import tests._bootstrap  # noqa: F401
from pipeline.config import SimEDaPEConfig
from pipeline.io import source
from pipeline.io.source import (
    describe_source,
    read_scenario,
    training_location,
)


class FakeClient:
    def __init__(self, df):
        self.df = df
        self.queries = []

    def query(self, sql, params=None):
        self.queries.append((sql, params or {}))
        return self.df


CH_DF = pd.DataFrame(
    {
        "link_id": ["3", "3", "3"],
        "tick": [100, 200, 300],
        "vehicle_id": ["v1", "v2", "v3"],
        "vehicle_count_on_link": [1, 2, 3],
    }
)


def _ch_config(**overrides):
    cfg = SimEDaPEConfig.from_dict(
        {
            "scenario_name": "ch",
            "data_source": "clickhouse",
            "intervals": {"start_time": 0, "end_time": 3600, "interval_size": 3600},
            "clickhouse": {"simulation_id": "train", "password": "s3cret"},
            **overrides,
        }
    )
    return cfg


class TestClickHouseSource(unittest.TestCase):
    def _read(self, cfg, location=""):
        client = FakeClient(CH_DF)
        with mock.patch.object(source, "build_clickhouse_client", return_value=client):
            result = read_scenario(cfg, location)
        return result, client

    def test_training_scenario_uses_configured_simulation_id(self):
        cfg = _ch_config()
        self.assertEqual(training_location(cfg), "train")
        result, client = self._read(cfg)
        self.assertEqual(client.queries[0][1]["simulation_id"], "train")
        self.assertEqual(result[1], [[1.0, 2.0, 3.0]])

    def test_partial_scenario_overrides_simulation_id(self):
        _, client = self._read(_ch_config(), location="partial_30")
        self.assertEqual(client.queries[0][1]["simulation_id"], "partial_30")

    def test_same_structure_as_csv_reader(self):
        result, _ = self._read(_ch_config())
        self.assertEqual(len(result), 6)  # mesma 6-upla de read_single_scenario


class TestCsvSource(unittest.TestCase):
    def test_reads_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "7-a.csv"), "w") as fh:
                fh.write("count;vehicle;time\n1;v1;100\n2;v2;200\n3;v3;300\n")
            cfg = SimEDaPEConfig.from_dict(
                {
                    "scenario_dir": tmp + os.sep,
                    "intervals": {"start_time": 0, "end_time": 3600, "interval_size": 3600},
                }
            )
            self.assertEqual(training_location(cfg), tmp + os.sep)
            result = read_scenario(cfg)
            self.assertEqual(result[1], [[1, 2, 3]])
            # location explícito (cenário parcial) tem prioridade sobre scenario_dir
            other = read_scenario(
                SimEDaPEConfig.from_dict({"scenario_dir": "/nao/existe/"}), tmp + os.sep
            )
            self.assertEqual(other[1], [[1, 2, 3]])


class TestSourceValidation(unittest.TestCase):
    def test_unknown_source_rejected(self):
        cfg = SimEDaPEConfig.from_dict({"data_source": "parquet"})
        for fn in (training_location, describe_source, read_scenario):
            with self.assertRaises(ValueError):
                fn(cfg)

    def test_describe_never_leaks_password(self):
        desc = describe_source(_ch_config())
        self.assertEqual(desc["type"], "clickhouse")
        self.assertEqual(desc["simulation_id"], "train")
        self.assertNotIn("password", desc)
        self.assertNotIn("s3cret", str(desc))

    def test_describe_csv(self):
        desc = describe_source(SimEDaPEConfig.from_dict({"scenario_dir": "d/"}))
        self.assertEqual(desc, {"type": "csv", "scenario_dir": "d/"})


if __name__ == "__main__":
    unittest.main()
