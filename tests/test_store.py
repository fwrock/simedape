"""Testes do ConfigStore (CRUD de configurações nomeadas)."""
import os
import tempfile
import unittest

import tests._bootstrap  # noqa: F401
from pipeline.api.store import ConfigStore
from pipeline.config import SimEDaPEConfig


class TestConfigStoreMemory(unittest.TestCase):
    def setUp(self):
        self.store = ConfigStore()

    def test_save_and_get(self):
        cfg = SimEDaPEConfig.from_dict(
            {"scenario_name": "sp", "data_source": "clickhouse"}
        )
        self.store.save("carla", cfg)
        got = self.store.get("carla")
        self.assertEqual(got.scenario_name, "sp")
        self.assertEqual(got.data_source, "clickhouse")

    def test_list_sorted(self):
        self.store.save("b", SimEDaPEConfig())
        self.store.save("a", SimEDaPEConfig())
        self.assertEqual(self.store.list(), ["a", "b"])

    def test_get_missing_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.store.get("nope")

    def test_delete(self):
        self.store.save("x", SimEDaPEConfig())
        self.assertTrue(self.store.delete("x"))
        self.assertFalse(self.store.delete("x"))
        self.assertEqual(self.store.list(), [])

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            self.store.save("../evil", SimEDaPEConfig())


class TestConfigStorePersistence(unittest.TestCase):
    def test_persists_and_reloads(self):
        with tempfile.TemporaryDirectory() as d:
            ConfigStore(storage_dir=d).save(
                "carla", SimEDaPEConfig.from_dict({"scenario_name": "sp"})
            )
            reloaded = ConfigStore(storage_dir=d)
            self.assertEqual(reloaded.list(), ["carla"])
            self.assertEqual(reloaded.get("carla").scenario_name, "sp")

    def test_delete_removes_file(self):
        with tempfile.TemporaryDirectory() as d:
            store = ConfigStore(storage_dir=d)
            store.save("x", SimEDaPEConfig())
            path = os.path.join(d, "x.json")
            self.assertTrue(os.path.exists(path))
            store.delete("x")
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
