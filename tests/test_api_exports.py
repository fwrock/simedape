"""Testes dos endpoints de progresso e exportação (/progress, /artifacts, /estimation, /comparison)."""
import json
import os
import tempfile
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

    class NoopStage(Stage):
        name = "noop"

        def execute(self, ctx):
            pass

    class SlowStage(Stage):
        name = "slow"
        gate = None

        def execute(self, ctx):
            SlowStage.gate.wait(5)


def _dump(folder, stem, content):
    with open(os.path.join(folder, f"{stem}.json"), "w") as fh:
        fh.write(json.dumps(content))  # NaN sai como literal, como em save_json


REAL_ERR = float("nan")


@unittest.skipUnless(HAVE_FLASK, "flask não instalado")
class TestExportApi(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = self._tmp.name
        self.jobs = JobManager(max_workers=2)
        self.app = create_app(
            config_store=ConfigStore(),
            job_manager=self.jobs,
            stage_registry={"noop": NoopStage, "slow": SlowStage},
        )
        self.client = self.app.test_client()
        self.addCleanup(self.jobs.shutdown, True)

    def _run(self, config=None, stages=("noop",), wait=True, **extra):
        config = {"scenario_name": "exp", "output_dir": self.out, **(config or {})}
        resp = self.client.post(
            "/api/runs", json={"stages": list(stages), "config": config, **extra}
        )
        self.assertEqual(resp.status_code, 202, resp.get_json())
        run_id = resp.get_json()["run_id"]
        if wait:
            deadline = time.time() + 5
            while self.jobs.get(run_id).status not in ("completed", "failed"):
                self.assertLess(time.time(), deadline)
                time.sleep(0.01)
        return run_id

    def _populate(self):
        _dump(self.out, "new_simulation", {"average_speed": [1, 2]})
        _dump(self.out, "result_avg", {
            "average_speed": [1.0, 2.0, 3.0, 4.0],
            "km_traveled": [10.0, 20.0, 30.0, 40.0],
            "missing_time_series": 2,
        })
        _dump(self.out, "result_wavg", {"average_speed": [1.5, 2.5], "km_traveled": [5.0, 6.0]})
        def cmp(label, mape, err, cov):
            return {
                "label": label, "n_real": 100, "n_estimated": int(cov),
                "coverage_pct": cov,
                "aggregate": {"real": {}, "estimated": {}, "error_pct": {"average_speed": err}},
                "per_series_error": {"average_speed": {"mape": mape}},
            }
        _dump(self.out, "compare_avg", cmp("avg", 4.5, REAL_ERR, 100.0))
        _dump(self.out, "compare_wavg", cmp("wavg", 2.5, -1.2, 80.0))
        _dump(self.out, "compare_summary", {"avg": cmp("avg", 4.5, 0, 100.0)})
        os.makedirs(os.path.join(self.out, "plots"))
        with open(os.path.join(self.out, "plots", "compare_report.pdf"), "wb") as fh:
            fh.write(b"%PDF")

    # -- progresso --------------------------------------------------------

    def test_progress_after_completion(self):
        run_id = self._run(stages=("noop", "noop"))
        body = self.client.get(f"/api/runs/{run_id}/progress").get_json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["total"], 2)
        self.assertEqual(body["completed"], 2)
        self.assertEqual(body["percent"], 100.0)
        self.assertEqual([s["status"] for s in body["stages"]], ["completed", "completed"])

    def test_progress_and_logs_while_running(self):
        import threading
        SlowStage.gate = threading.Event()
        self.addCleanup(SlowStage.gate.set)
        run_id = self._run(stages=("noop", "slow"), wait=False)
        deadline = time.time() + 5
        while True:
            body = self.client.get(f"/api/runs/{run_id}/progress").get_json()
            if body.get("current_stage") == "slow":
                break
            self.assertLess(time.time(), deadline)
            time.sleep(0.01)
        self.assertEqual(body["status"], "running")
        self.assertEqual(body["completed"], 1)
        self.assertEqual(body["percent"], 50.0)
        self.assertIn("slow", self.client.get(f"/api/runs/{run_id}").get_json()["logs"])
        SlowStage.gate.set()

    def test_progress_unknown_run(self):
        self.assertEqual(self.client.get("/api/runs/nope/progress").status_code, 404)

    # -- fonte de dados ---------------------------------------------------

    def test_data_source_recorded_for_csv_and_clickhouse(self):
        csv_id = self._run({"scenario_dir": "d/"})
        ch_id = self._run({
            "data_source": "clickhouse",
            "clickhouse": {"host": "ch", "simulation_id": "s1", "password": "S3cr3t-Pw-9f2"},
        })
        csv_src = self.client.get(f"/api/runs/{csv_id}/progress").get_json()["data_source"]
        ch_src = self.client.get(f"/api/runs/{ch_id}/progress").get_json()["data_source"]
        self.assertEqual(csv_src, {"type": "csv", "scenario_dir": "d/"})
        self.assertEqual(ch_src["type"], "clickhouse")
        self.assertEqual(ch_src["simulation_id"], "s1")
        self.assertNotIn("S3cr3t-Pw-9f2", json.dumps(self.client.get(f"/api/runs/{ch_id}").get_json()))

    def test_invalid_data_source_rejected_at_submit(self):
        resp = self.client.post("/api/runs", json={
            "stages": ["noop"], "config": {"data_source": "parquet"}})
        self.assertEqual(resp.status_code, 400)

    def test_exports_identical_for_both_sources(self):
        self._populate()
        csv_id = self._run()
        ch_id = self._run({"data_source": "clickhouse"})
        for path in ("estimation", "comparison", "artifacts"):
            a = self.client.get(f"/api/runs/{csv_id}/{path}").get_json()
            b = self.client.get(f"/api/runs/{ch_id}/{path}").get_json()
            a.pop("run_id"), b.pop("run_id"), a.pop("data_source", None), b.pop("data_source", None)
            self.assertEqual(a, b, path)

    # -- artefatos --------------------------------------------------------

    def test_artifacts_listing(self):
        self._populate()
        run_id = self._run()
        body = self.client.get(f"/api/runs/{run_id}/artifacts").get_json()
        kinds = {a["name"]: a["kind"] for a in body["artifacts"]}
        self.assertEqual(kinds["new_simulation.json"], "real_metrics")
        self.assertEqual(kinds["result_avg.json"], "estimation")
        self.assertEqual(kinds["compare_wavg.json"], "comparison")
        self.assertEqual(kinds["plots/compare_report.pdf"], "report")

    def test_artifacts_empty_before_any_output(self):
        run_id = self._run({"output_dir": os.path.join(self.out, "vazio")})
        body = self.client.get(f"/api/runs/{run_id}/artifacts").get_json()
        self.assertEqual(body["artifacts"], [])

    # -- estimativas ------------------------------------------------------

    def test_estimation_overview(self):
        self._populate()
        run_id = self._run()
        body = self.client.get(f"/api/runs/{run_id}/estimation").get_json()
        self.assertEqual(sorted(body["methods"]), ["avg", "wavg"])
        speed = body["methods"]["avg"]["metrics"]["average_speed"]
        self.assertEqual((speed["count"], speed["min"], speed["max"], speed["mean"]), (4, 1.0, 4.0, 2.5))
        self.assertEqual(body["methods"]["avg"]["scalars"], {"missing_time_series": 2})

    def test_estimation_detail_with_metric_and_paging(self):
        self._populate()
        run_id = self._run()
        body = self.client.get(
            f"/api/runs/{run_id}/estimation?method=avg&metric=average_speed&offset=1&limit=2"
        ).get_json()
        self.assertEqual(body["data"]["average_speed"], [2.0, 3.0])
        self.assertNotIn("km_traveled", body["data"])
        self.assertEqual(body["total_series"], 4)

    def test_estimation_honours_stage_params_prefix(self):
        _dump(self.out, "meu_avg", {"average_speed": [9.0]})
        run_id = self._run(stage_params={"estimation": {"output_prefix": "meu"}})
        body = self.client.get(f"/api/runs/{run_id}/estimation").get_json()
        self.assertEqual(list(body["methods"]), ["avg"])

    def test_estimation_errors(self):
        self._populate()
        run_id = self._run()
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/estimation?method=zzz").status_code, 404)
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/estimation?method=avg&limit=x").status_code, 400)

    # -- comparação -------------------------------------------------------

    def test_comparison_matrix_and_nan_as_null(self):
        self._populate()
        run_id = self._run()
        resp = self.client.get(f"/api/runs/{run_id}/comparison")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn(b"NaN", resp.data)  # JSON estrito
        body = resp.get_json()
        self.assertEqual(body["methods"], ["avg", "wavg"])  # sem "summary"
        self.assertEqual(body["mape"]["average_speed"], {"avg": 4.5, "wavg": 2.5})
        self.assertEqual(body["aggregate_error_pct"]["average_speed"], {"avg": None, "wavg": -1.2})
        self.assertEqual(body["coverage_pct"], {"avg": 100.0, "wavg": 80.0})
        self.assertEqual(body["metrics"][0]["label"], "Velocidade Média (m/s)")

    def test_comparison_detail(self):
        self._populate()
        run_id = self._run()
        body = self.client.get(f"/api/runs/{run_id}/comparison?method=wavg").get_json()
        self.assertEqual(body["label"], "wavg")
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/comparison?method=zzz").status_code, 404)

    def test_run_detail_has_no_nan_literals(self):
        run_id = self._run()
        self.jobs.get(run_id).result = {"x": float("nan")}
        resp = self.client.get(f"/api/runs/{run_id}")
        self.assertNotIn(b"NaN", resp.data)

    # -- segurança --------------------------------------------------------

    def test_path_traversal_rejected(self):
        self._populate()
        secret = os.path.join(os.path.dirname(self.out), "segredo.json")
        run_id = self._run()
        for bad in ("../segredo", "..%2Fsegredo", "a/b", ".."):
            r = self.client.get(f"/api/runs/{run_id}/estimation?method={bad}")
            self.assertIn(r.status_code, (400, 404), bad)
            r = self.client.get(f"/api/runs/{run_id}/comparison?method={bad}")
            self.assertIn(r.status_code, (400, 404), bad)
        r = self.client.get(f"/api/runs/{run_id}/estimation?prefix=../x&method=avg")
        self.assertEqual(r.status_code, 400)
        self.assertFalse(os.path.exists(secret))

    # -- persistência -----------------------------------------------------

    def test_exports_work_after_restart(self):
        self._populate()
        with tempfile.TemporaryDirectory() as state:
            m1 = JobManager(state_dir=state)
            app1 = create_app(config_store=ConfigStore(), job_manager=m1,
                              stage_registry={"noop": NoopStage})
            resp = app1.test_client().post("/api/runs", json={
                "stages": ["noop"], "config": {"output_dir": self.out}})
            run_id = resp.get_json()["run_id"]
            deadline = time.time() + 5
            while m1.get(run_id).status != "completed":
                self.assertLess(time.time(), deadline)
                time.sleep(0.01)
            m1.shutdown(True)

            m2 = JobManager(state_dir=state)
            self.addCleanup(m2.shutdown, True)
            app2 = create_app(config_store=ConfigStore(), job_manager=m2,
                              stage_registry={"noop": NoopStage})
            body = app2.test_client().get(f"/api/runs/{run_id}/comparison").get_json()
            self.assertEqual(body["methods"], ["avg", "wavg"])


if __name__ == "__main__":
    unittest.main()
