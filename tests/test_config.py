"""Testes de configuração da fonte de dados (dataclasses + loaders)."""
import os
import tempfile
import unittest

import tests._bootstrap  # noqa: F401
from pipeline.config import ClickHouseConfig, SimEDaPEConfig

try:
    import jproperties  # noqa: F401

    HAVE_JPROPERTIES = True
except ImportError:
    HAVE_JPROPERTIES = False


class TestClickHouseConfig(unittest.TestCase):
    def test_defaults(self):
        ch = ClickHouseConfig()
        self.assertEqual(ch.host, "localhost")
        self.assertEqual(ch.port, 8123)
        self.assertEqual(ch.database, "htc")
        self.assertEqual(ch.user, "default")
        self.assertEqual(ch.password, "")
        self.assertEqual(ch.table, "vehicle_link_events")
        self.assertEqual(ch.simulation_id, "")
        self.assertFalse(ch.secure)


class TestSimEDaPEConfigDefaults(unittest.TestCase):
    def test_default_source_is_csv(self):
        cfg = SimEDaPEConfig()
        self.assertEqual(cfg.data_source, "csv")
        self.assertEqual(cfg.clickhouse.database, "htc")


class TestFromYaml(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".yml")
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_loads_clickhouse(self):
        path = self._write(
            """
config:
  scenario_name: sp
  data_source: clickhouse
  clickhouse:
    host: ch.example.com
    port: 9000
    database: mydb
    user: admin
    password: secret
    table: link_vehicle_counts
    simulation_id: run42
    secure: true
"""
        )
        cfg = SimEDaPEConfig.from_yaml(path)
        self.assertEqual(cfg.data_source, "clickhouse")
        self.assertEqual(cfg.clickhouse.host, "ch.example.com")
        self.assertEqual(cfg.clickhouse.port, 9000)
        self.assertEqual(cfg.clickhouse.database, "mydb")
        self.assertEqual(cfg.clickhouse.user, "admin")
        self.assertEqual(cfg.clickhouse.password, "secret")
        self.assertEqual(cfg.clickhouse.table, "link_vehicle_counts")
        self.assertEqual(cfg.clickhouse.simulation_id, "run42")
        self.assertTrue(cfg.clickhouse.secure)

    def test_defaults_when_absent(self):
        path = self._write("config:\n  scenario_name: sp\n")
        cfg = SimEDaPEConfig.from_yaml(path)
        self.assertEqual(cfg.data_source, "csv")
        self.assertEqual(cfg.clickhouse.database, "htc")
        self.assertEqual(cfg.clickhouse.table, "vehicle_link_events")


@unittest.skipUnless(HAVE_JPROPERTIES, "jproperties não instalado")
class TestFromProperties(unittest.TestCase):
    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".properties")
        with os.fdopen(fd, "wb") as fh:
            fh.write(text.encode("utf-8"))
        self.addCleanup(os.remove, path)
        return path

    def test_loads_clickhouse(self):
        path = self._write(
            "scenario.name=sp\n"
            "data.source=clickhouse\n"
            "clickhouse.host=ch.example.com\n"
            "clickhouse.port=8123\n"
            "clickhouse.database=htc\n"
            "clickhouse.user=default\n"
            "clickhouse.password=\n"
            "clickhouse.table=vehicle_link_events\n"
            "clickhouse.simulation_id=run9\n"
            "clickhouse.secure=False\n"
        )
        cfg = SimEDaPEConfig.from_properties_file(path)
        self.assertEqual(cfg.data_source, "clickhouse")
        self.assertEqual(cfg.clickhouse.host, "ch.example.com")
        self.assertEqual(cfg.clickhouse.port, 8123)
        self.assertEqual(cfg.clickhouse.simulation_id, "run9")
        self.assertFalse(cfg.clickhouse.secure)


class TestConfigSerialization(unittest.TestCase):
    def test_from_dict_to_dict_roundtrip(self):
        cfg = SimEDaPEConfig.from_dict(
            {
                "scenario_name": "sp",
                "data_source": "clickhouse",
                "intervals": {"start_time": 0, "end_time": 7200, "interval_size": 1800},
                "clustering": {"n_clusters": 12},
                "clickhouse": {"database": "htc", "simulation_id": "run1"},
            }
        )
        d = cfg.to_dict()
        self.assertEqual(d["scenario_name"], "sp")
        self.assertEqual(d["data_source"], "clickhouse")
        self.assertEqual(d["intervals"]["interval_size"], 1800)
        self.assertEqual(d["clustering"]["n_clusters"], 12)
        self.assertEqual(d["clickhouse"]["simulation_id"], "run1")

        cfg2 = SimEDaPEConfig.from_dict(d)
        self.assertEqual(cfg2.to_dict(), d)


if __name__ == "__main__":
    unittest.main()
