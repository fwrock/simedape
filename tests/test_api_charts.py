"""Testes dos endpoints de dados de gráficos, sweep e relatórios em PDF."""
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

METRICS = ["average_speed", "percentage_of_track", "average_vehicles", "km_traveled"]
NAN = float("nan")


def _dump(folder, stem, content):
    with open(os.path.join(folder, f"{stem}.json"), "w") as fh:
        fh.write(json.dumps(content))


def _compare(label, mape, err, cov=100.0):
    return {
        "label": label, "n_real": 10, "n_estimated": int(cov / 10), "coverage_pct": cov,
        "aggregate": {"real": {}, "estimated": {},
                      "error_pct": {m: err + i for i, m in enumerate(METRICS)}},
        "per_series_error": {
            m: {"min": -10.0, "p25": -2.0, "median": 0.5, "p75": 3.0, "max": 12.0,
                "mean": 1.0, "std": 2.0, "mape": mape + i}
            for i, m in enumerate(METRICS)
        },
    }


def _populate(out):
    _dump(out, "compare_avg", _compare("avg", 4.0, -1.0))
    _dump(out, "compare_wavg", _compare("wavg", 2.0, NAN, cov=80.0))
    _dump(out, "compare_summary", {"avg": _compare("avg", 4.0, -1.0)})
    _dump(out, "new_simulation", {"average_speed": [1.0, 2.0, 3.0, NAN, 5.0, 6.0]})
    _dump(out, "result_avg", {"average_speed": [1.1, 2.2, 2.9, 4.0, 5.5, 6.1]})
    _dump(out, "time_series_labels", ["a-0-1", "a-1-2", "b-0-1", "b-1-2", "c-0-1", "c-1-2"])
    # cenários fora de ordem de propósito; wavg ausente no cenário de 30%
    def entry(mape, cov):
        return {"coverage_pct": cov, "mape": {m: mape + i for i, m in enumerate(METRICS)},
                "error_pct": {m: -mape for m in METRICS}}
    _dump(out, "sweep_summary", {"scenarios": [
        {"label": "p100", "percentage": 100.0,
         "methods": {"avg": entry(5.0, 100.0), "wavg": entry(4.0, 100.0)}},
        {"label": "p30", "percentage": 30.0, "methods": {"avg": entry(NAN, 30.0)}},
    ]})
    _dump(out, "clustering", {
        "number_of_clusters": 2,
        "time_series_by_cluster": {"1": 4, "0": 6},
        "centroids": {"0": [[0.1], [0.2], [0.3]], "1": [[1.0], [2.0], [3.0]]},
    })
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    with open(os.path.join(out, "plots", "compare_avg.pdf"), "wb") as fh:
        fh.write(b"%PDF-1.4 fake")


@unittest.skipUnless(HAVE_FLASK, "flask não instalado")
class TestChartApi(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = self._tmp.name
        self.jobs = JobManager(max_workers=2)
        self.addCleanup(self.jobs.shutdown, True)
        self.app = create_app(config_store=ConfigStore(), job_manager=self.jobs,
                              stage_registry={"noop": NoopStage})
        self.client = self.app.test_client()

    def _run(self, config=None, **extra):
        cfg = {"scenario_name": "c", "output_dir": self.out, **(config or {})}
        resp = self.client.post("/api/runs", json={"stages": ["noop"], "config": cfg, **extra})
        run_id = resp.get_json()["run_id"]
        deadline = time.time() + 5
        while self.jobs.get(run_id).status != "completed":
            self.assertLess(time.time(), deadline)
            time.sleep(0.01)
        return run_id

    def _chart(self, run_id, cid, query=""):
        return self.client.get(f"/api/runs/{run_id}/charts/{cid}{query}")

    # -- catálogo ---------------------------------------------------------

    def test_catalog_flags_follow_available_data(self):
        run_id = self._run()
        empty = self.client.get(f"/api/runs/{run_id}/charts").get_json()
        self.assertEqual(len(empty["charts"]), 11)
        self.assertFalse(any(c["available"] for c in empty["charts"]))
        _populate(self.out)
        full = self.client.get(f"/api/runs/{run_id}/charts").get_json()
        self.assertTrue(all(c["available"] for c in full["charts"]))
        by_id = {c["id"]: c for c in full["charts"]}
        self.assertEqual(by_id["error_distribution"]["params"], ["metric"])
        self.assertEqual(by_id["mape_radar"]["type"], "radar")

    def test_unavailable_data_is_404_and_unknown_chart_is_404(self):
        run_id = self._run()
        self.assertEqual(self._chart(run_id, "mape_by_method").status_code, 404)
        self.assertEqual(self._chart(run_id, "nao_existe").status_code, 404)

    # -- gráficos de compare ----------------------------------------------

    def test_compare_charts(self):
        _populate(self.out)
        run_id = self._run()
        err = self._chart(run_id, "error_aggregate")
        self.assertNotIn(b"NaN", err.data)
        body = err.get_json()
        self.assertEqual(body["type"], "bar")
        self.assertEqual(body["x"]["categories"][0], "Velocidade Média (m/s)")
        series = {s["name"]: s["data"] for s in body["series"]}
        self.assertEqual(series["avg"], [-1.0, 0.0, 1.0, 2.0])
        self.assertEqual(series["wavg"], [None] * 4)  # NaN → null

        mape = self._chart(run_id, "mape_by_method").get_json()
        self.assertEqual({s["name"]: s["data"][0] for s in mape["series"]}, {"avg": 4.0, "wavg": 2.0})
        self.assertEqual(self._chart(run_id, "mape_radar").get_json()["type"], "radar")

        cov = self._chart(run_id, "coverage").get_json()
        self.assertEqual(cov["x"]["categories"], ["avg", "wavg"])
        self.assertEqual(cov["series"][0]["data"], [100.0, 80.0])

    def test_metric_order_matches_the_pdf_reports(self):
        _populate(self.out)
        run_id = self._run()
        cats = self._chart(run_id, "mape_by_method").get_json()["x"]["categories"]
        self.assertEqual(cats, ["Velocidade Média (m/s)", "Ocupação (%)", "Veículos Médios", "Km Trafegados"])
        matrix = self.client.get(f"/api/runs/{run_id}/comparison").get_json()
        self.assertEqual([m["key"] for m in matrix["metrics"]], METRICS)
        table = self.client.get(f"/api/runs/{run_id}/sweep").get_json()
        self.assertEqual([m["key"] for m in table["metrics"]], METRICS)

    def test_error_distribution_boxplot_and_metric_param(self):
        _populate(self.out)
        run_id = self._run()
        box = self._chart(run_id, "error_distribution", "?metric=km_traveled").get_json()
        self.assertEqual(box["type"], "boxplot")
        self.assertEqual(box["series"][0]["data"][0],
                         {"min": -10.0, "q1": -2.0, "median": 0.5, "q3": 3.0, "max": 12.0, "mean": 1.0})
        self.assertEqual(box["meta"]["metric"], "km_traveled")
        self.assertEqual(self._chart(run_id, "error_distribution", "?metric=foo").status_code, 400)

    # -- real vs estimado -------------------------------------------------

    def test_scatter_real_vs_estimated(self):
        _populate(self.out)
        run_id = self._run()
        body = self._chart(run_id, "real_vs_estimated").get_json()
        self.assertEqual(body["type"], "scatter")
        self.assertEqual(body["meta"]["reference"], "identity")
        pts = body["series"][0]["data"]
        self.assertEqual(len(pts), 5)  # a série com NaN no real é descartada
        self.assertEqual(pts[0], {"x": 1.0, "y": 1.1, "label": "a-0-1"})
        small = self._chart(run_id, "real_vs_estimated", "?limit=2").get_json()
        self.assertLessEqual(len(small["series"][0]["data"]), 2)
        self.assertEqual(small["meta"]["total_series"], 6)
        self.assertEqual(self._chart(run_id, "real_vs_estimated", "?limit=0").status_code, 400)
        self.assertEqual(self._chart(run_id, "real_vs_estimated", "?limit=x").status_code, 400)
        self.assertEqual(self._chart(run_id, "real_vs_estimated", "?method=zzz").status_code, 404)

    # -- sweep ------------------------------------------------------------

    def test_sweep_table_sorted_by_percentage(self):
        _populate(self.out)
        run_id = self._run()
        resp = self.client.get(f"/api/runs/{run_id}/sweep")
        self.assertNotIn(b"NaN", resp.data)
        body = resp.get_json()
        self.assertEqual([s["label"] for s in body["scenarios"]], ["p30", "p100"])
        self.assertEqual(body["methods"], ["avg", "wavg"])
        self.assertEqual(body["mape"]["average_speed"]["avg"], [None, 5.0])
        self.assertEqual(body["mape"]["average_speed"]["wavg"], [None, 4.0])  # ausente em p30
        self.assertEqual(body["coverage_pct"]["avg"], [30.0, 100.0])
        only = self.client.get(f"/api/runs/{run_id}/sweep?methods=wavg").get_json()
        self.assertEqual(only["methods"], ["wavg"])
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/sweep?methods=zzz").status_code, 404)
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/sweep?methods=../x").status_code, 400)

    def test_sweep_charts(self):
        _populate(self.out)
        run_id = self._run()
        line = self._chart(run_id, "sweep_progression", "?metric=km_traveled").get_json()
        self.assertEqual(line["x"]["categories"], ["p30", "p100"])
        self.assertEqual(line["meta"]["percentages"], [30.0, 100.0])
        self.assertEqual({s["name"]: s["data"] for s in line["series"]}["avg"], [None, 8.0])
        heat = self._chart(run_id, "sweep_heatmap").get_json()
        self.assertEqual(heat["type"], "heatmap")
        self.assertEqual(heat["y"]["categories"], ["avg", "wavg"])
        self.assertEqual(heat["values"], [[None, 5.0], [None, 4.0]])
        self.assertEqual(self._chart(run_id, "sweep_coverage", "?methods=avg").get_json()["series"][0]["data"],
                         [30.0, 100.0])

    # -- clusters ---------------------------------------------------------

    def test_cluster_charts(self):
        _populate(self.out)
        run_id = self._run()
        sizes = self._chart(run_id, "cluster_sizes").get_json()
        self.assertEqual(sizes["x"]["categories"], ["0", "1"])  # ordenado numericamente
        self.assertEqual(sizes["series"][0]["data"], [6, 4])
        cent = self._chart(run_id, "centroids").get_json()
        self.assertEqual({s["name"]: s["data"] for s in cent["series"]}["cluster 1"], [1.0, 2.0, 3.0])
        self.assertEqual(cent["x"]["categories"], [0, 1, 2])

    # -- prefixos / fontes ------------------------------------------------

    def test_stage_params_prefix_and_query_prefix_ignored(self):
        _dump(self.out, "cmp_avg", _compare("avg", 9.0, 0.0))
        run_id = self._run(stage_params={"compare": {"output_prefix": "cmp"}})
        body = self._chart(run_id, "mape_by_method").get_json()
        self.assertEqual(body["series"][0]["data"][0], 9.0)
        # ?prefix= não redireciona gráficos (há vários prefixos em jogo)
        _populate(self.out)
        other = self._run()
        ok = self._chart(other, "mape_by_method", "?prefix=cmp").get_json()
        self.assertEqual({s["name"] for s in ok["series"]}, {"avg", "wavg"})

    def test_invalid_stage_prefix_rejected(self):
        run_id = self._run(stage_params={"compare": {"output_prefix": "../x"}})
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/charts").status_code, 400)
        self.assertEqual(self._chart(run_id, "mape_by_method").status_code, 400)

    def test_charts_identical_for_csv_and_clickhouse_runs(self):
        _populate(self.out)
        a = self._run()
        b = self._run({"data_source": "clickhouse"})
        for cid in ("mape_by_method", "sweep_heatmap", "real_vs_estimated"):
            ca, cb = self._chart(a, cid).get_json(), self._chart(b, cid).get_json()
            ca.pop("run_id"), cb.pop("run_id")
            self.assertEqual(ca, cb, cid)

    # -- relatórios PDF ---------------------------------------------------

    def test_reports_list_and_download(self):
        _populate(self.out)
        run_id = self._run()
        listing = self.client.get(f"/api/runs/{run_id}/reports").get_json()
        self.assertEqual([r["name"] for r in listing["reports"]], ["compare_avg.pdf"])
        with self.client.get(f"/api/runs/{run_id}/reports/compare_avg.pdf") as resp:
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.mimetype, "application/pdf")
            self.assertTrue(resp.data.startswith(b"%PDF"))

    def test_report_download_works_with_relative_output_dir(self):
        _populate(self.out)
        old = os.getcwd()
        os.chdir(os.path.dirname(self.out))
        self.addCleanup(os.chdir, old)
        run_id = self._run({"output_dir": os.path.basename(self.out)})
        with self.client.get(f"/api/runs/{run_id}/reports/compare_avg.pdf") as resp:
            self.assertEqual(resp.status_code, 200)

    def test_report_download_rejects_unsafe_names(self):
        _populate(self.out)
        with open(os.path.join(self.out, "segredo.json"), "w") as fh:
            fh.write("{}")
        run_id = self._run()
        for bad in ("segredo.json", "..%2Fsegredo.pdf", "nao_existe.pdf", ".pdf", "a%2Fb.pdf"):
            r = self.client.get(f"/api/runs/{run_id}/reports/{bad}")
            self.assertEqual(r.status_code, 404, bad)


if __name__ == "__main__":
    unittest.main()
