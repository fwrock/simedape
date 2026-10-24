"""Testes do JobManager (execução em background com rastreamento)."""
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


if __name__ == "__main__":
    unittest.main()
