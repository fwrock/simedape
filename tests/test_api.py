"""Testes da API HTTP (Flask) — configs, stages e runs."""
import time
import unittest

import tests._bootstrap  # noqa: F401

try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except ImportError:
    HAVE_FLASK = False

if HAVE_FLASK:
    from pipeline.api.app import create_app
    from pipeline.api.jobs import JobManager
    from pipeline.api.store import ConfigStore
    from pipeline.stage import Stage


if HAVE_FLASK:

    class NoopStage(Stage):
        name = "noop"

        def execute(self, ctx):
            ctx.max_size = 10
            ctx.street_series = [[1, 2, 3], [4, 5, 6], [7, 8]]


@unittest.skipUnless(HAVE_FLASK, "flask não instalado")
class TestApi(unittest.TestCase):
    def setUp(self):
        self.store = ConfigStore()
        self.jobs = JobManager(max_workers=2)
        self.app = create_app(
            config_store=self.store,
            job_manager=self.jobs,
            stage_registry={"noop": NoopStage},
        )
        self.client = self.app.test_client()
        self.addCleanup(self.jobs.shutdown, True)

    def _wait(self, run_id, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.jobs.get(run_id)
            if job.status in ("completed", "failed", "cancelled"):
                return job
            time.sleep(0.01)
        raise AssertionError("run não finalizou")

    def test_health(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json(), {"status": "ok"})

    def test_stages(self):
        resp = self.client.get("/api/stages")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual([s["name"] for s in resp.get_json()], ["noop"])

    def test_config_crud(self):
        resp = self.client.post(
            "/api/configs",
            json={
                "name": "carla",
                "config": {"scenario_name": "sp", "data_source": "clickhouse"},
            },
        )
        self.assertEqual(resp.status_code, 201)

        self.assertEqual(self.client.get("/api/configs").get_json()[0]["name"], "carla")

        got = self.client.get("/api/configs/carla").get_json()
        self.assertEqual(got["config"]["data_source"], "clickhouse")

        resp = self.client.put(
            "/api/configs/carla", json={"scenario_name": "sp2"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(self.store.get("carla").scenario_name, "sp2")

        self.assertEqual(
            self.client.delete("/api/configs/carla").status_code, 200
        )
        self.assertEqual(
            self.client.get("/api/configs/carla").status_code, 404
        )

    def test_create_config_validation(self):
        resp = self.client.post("/api/configs", json={"name": "x"})
        self.assertEqual(resp.status_code, 400)

    def test_run_executes_pipeline(self):
        resp = self.client.post(
            "/api/runs",
            json={
                "name": "myrun",
                "stages": ["noop"],
                "config": {"scenario_name": "test-run"},
            },
        )
        self.assertEqual(resp.status_code, 202)
        run_id = resp.get_json()["run_id"]

        job = self._wait(run_id)
        self.assertEqual(job.status, "completed")

        detail = self.client.get(f"/api/runs/{run_id}").get_json()
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(detail["result"]["scenario_name"], "test-run")
        self.assertEqual(detail["result"]["n_series"], 3)
        self.assertEqual(detail["result"]["max_size"], 10)

        self.assertEqual(len(self.client.get("/api/runs").get_json()), 1)

    def test_run_by_config_name(self):
        from pipeline.config import SimEDaPEConfig

        self.store.save("carla", SimEDaPEConfig.from_dict({"scenario_name": "c1"}))
        resp = self.client.post(
            "/api/runs", json={"stages": ["noop"], "config": "carla"}
        )
        self.assertEqual(resp.status_code, 202)
        self.assertEqual(self._wait(resp.get_json()["run_id"]).status, "completed")

    def test_run_unknown_stage(self):
        resp = self.client.post("/api/runs", json={"stages": ["nope"]})
        self.assertEqual(resp.status_code, 400)

    def test_run_missing_config_name(self):
        resp = self.client.post(
            "/api/runs", json={"stages": ["noop"], "config": "missing"}
        )
        self.assertEqual(resp.status_code, 404)

    def test_run_missing_stages(self):
        resp = self.client.post("/api/runs", json={})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
