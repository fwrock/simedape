"""Testes do JobManager (execução em background com rastreamento)."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest

import tests._bootstrap  # noqa: F401
from pipeline.api.jobs import JobManager


class TestJobManager(unittest.TestCase):
    def setUp(self):
        self.manager = JobManager(max_workers=2)
        self.addCleanup(self.manager.shutdown, True)

    def _wait(self, run_id, status="completed", timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = self.manager.get(run_id)
            if job.status == status:
                return job
            if job.status in ("failed", "cancelled"):
                return job
            time.sleep(0.01)
        raise AssertionError(f"run {run_id} não atingiu '{status}' em {timeout}s")

    def test_completes(self):
        run_id = self.manager.submit(lambda: {"ok": True}, name="t")
        job = self._wait(run_id)
        self.assertEqual(job.status, "completed")
        self.assertEqual(job.result, {"ok": True})
        self.assertIsNotNone(job.started_at)
        self.assertIsNotNone(job.finished_at)

    def test_captures_logs(self):
        run_id = self.manager.submit(lambda: print("hello pipeline"))
        job = self._wait(run_id)
        self.assertIn("hello pipeline", job.logs)

    def test_failure(self):
        def boom():
            raise RuntimeError("boom")

        run_id = self.manager.submit(boom)
        job = self._wait(run_id, status="failed")
        self.assertEqual(job.status, "failed")
        self.assertIn("RuntimeError", job.error)
        self.assertIn("boom", job.error)

    def test_list(self):
        r1 = self.manager.submit(lambda: {})
        r2 = self.manager.submit(lambda: {})
        self._wait(r1)
        self._wait(r2)
        ids = [j.run_id for j in self.manager.list()]
        self.assertIn(r1, ids)
        self.assertIn(r2, ids)

    def test_get_missing(self):
        with self.assertRaises(KeyError):
            self.manager.get("nope")


class TestLiveLogsAndProgress(unittest.TestCase):
    def setUp(self):
        self.manager = JobManager(max_workers=2)
        self.addCleanup(self.manager.shutdown, True)

    def _until(self, predicate, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("condição não atingida")

    def test_logs_visible_while_running(self):
        printed, release = threading.Event(), threading.Event()

        def fn():
            print("linha parcial")
            printed.set()
            release.wait(5)

        run_id = self.manager.submit(fn)
        self.assertTrue(printed.wait(5))
        live = self.manager.get(run_id).to_dict()
        self.assertEqual(live["status"], "running")
        self.assertIn("linha parcial", live["logs"])
        release.set()
        self._until(lambda: self.manager.get(run_id).status == "completed")
        self.assertIn("linha parcial", self.manager.get(run_id).logs)

    def test_concurrent_runs_keep_separate_logs(self):
        both_started = threading.Barrier(2)

        def make(tag):
            def fn():
                print(f"{tag}-antes")
                both_started.wait(5)
                print(f"{tag}-depois")
            return fn

        a = self.manager.submit(make("A"))
        b = self.manager.submit(make("B"))
        self._until(lambda: all(
            self.manager.get(r).status == "completed" for r in (a, b)))
        logs_a, logs_b = self.manager.get(a).logs, self.manager.get(b).logs
        self.assertIn("A-antes", logs_a)
        self.assertIn("A-depois", logs_a)
        self.assertNotIn("B-", logs_a)
        self.assertIn("B-depois", logs_b)
        self.assertNotIn("A-", logs_b)

    def test_streams_restored_after_runs(self):
        before = (sys.stdout, sys.stderr)
        run_id = self.manager.submit(lambda: print("x"))
        self._until(lambda: self.manager.get(run_id).status == "completed")
        self.assertEqual((sys.stdout, sys.stderr), before)

    def test_progress_from_pipeline_events(self):
        def fn(job):
            job.on_progress({"event": "pipeline_start", "stages": ["a", "b"]})
            job.on_progress({"event": "stage_start", "index": 1, "name": "a"})
            job.on_progress({"event": "stage_end", "index": 1, "name": "a", "elapsed": 0.5})
            job.on_progress({"event": "stage_start", "index": 2, "name": "b"})
            return {}

        run_id = self.manager.submit(fn, pass_job=True)
        self._until(lambda: self.manager.get(run_id).status == "completed")
        prog = self.manager.get(run_id).to_dict()["progress"]
        self.assertEqual(prog["total"], 2)
        self.assertEqual(prog["completed"], 1)
        self.assertEqual(prog["current_stage"], "b")
        self.assertEqual(prog["percent"], 100.0)  # run concluído
        self.assertEqual([s["status"] for s in prog["stages"]], ["completed", "running"])

    def test_percent_while_running(self):
        job_box = {}
        paused, release = threading.Event(), threading.Event()

        def fn(job):
            job_box["job"] = job
            job.on_progress({"event": "pipeline_start", "stages": ["a", "b", "c", "d"]})
            job.on_progress({"event": "stage_start", "index": 1, "name": "a"})
            job.on_progress({"event": "stage_end", "index": 1, "name": "a", "elapsed": 0.1})
            paused.set()
            release.wait(5)

        run_id = self.manager.submit(fn, pass_job=True)
        self.assertTrue(paused.wait(5))
        self.assertEqual(self.manager.get(run_id).progress_dict()["percent"], 25.0)
        release.set()

    def test_meta_is_kept(self):
        run_id = self.manager.submit(lambda: {}, meta={"data_source": {"type": "csv"}})
        self.assertEqual(self.manager.get(run_id).meta["data_source"]["type"], "csv")


class TestPersistence(unittest.TestCase):
    def test_runs_survive_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            m1 = JobManager(state_dir=tmp)
            run_id = m1.submit(lambda: {"ok": 1}, name="persist", meta={"output_dir": "o/"})
            deadline = time.time() + 5
            while m1.get(run_id).status != "completed" and time.time() < deadline:
                time.sleep(0.01)
            m1.shutdown(True)

            m2 = JobManager(state_dir=tmp)
            self.addCleanup(m2.shutdown, True)
            job = m2.get(run_id)
            self.assertEqual(job.status, "completed")
            self.assertEqual(job.result, {"ok": 1})
            self.assertEqual(job.meta["output_dir"], "o/")

    def test_interrupted_run_marked_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "abc.json"), "w") as fh:
                json.dump({"run_id": "abc", "name": "x", "status": "running"}, fh)
            with open(os.path.join(tmp, "lixo.json"), "w") as fh:
                fh.write("{não é json")
            manager = JobManager(state_dir=tmp)
            self.addCleanup(manager.shutdown, True)
            job = manager.get("abc")
            self.assertEqual(job.status, "failed")
            self.assertIn("reiniciado", job.error)
            self.assertEqual([j.run_id for j in manager.list()], ["abc"])


if __name__ == "__main__":
    unittest.main()
