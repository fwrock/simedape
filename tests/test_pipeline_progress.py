"""Testes dos eventos de progresso de Pipeline.run."""
import unittest

import tests._bootstrap  # noqa: F401
from pipeline.config import SimEDaPEConfig
from pipeline.pipeline import Pipeline
from pipeline.stage import Stage


class Ok(Stage):
    name = "ok"

    def execute(self, ctx):
        pass


class Boom(Stage):
    name = "boom"

    def execute(self, ctx):
        raise RuntimeError("falhou")


def _pipeline(*stages):
    p = Pipeline(SimEDaPEConfig.from_dict({"scenario_name": "t"}))
    for s in stages:
        p.add(s)
    return p


class TestPipelineProgress(unittest.TestCase):
    def test_event_sequence(self):
        events = []
        _pipeline(Ok(), Ok()).run(progress=events.append)
        self.assertEqual(
            [(e["event"], e.get("index")) for e in events],
            [
                ("pipeline_start", None),
                ("stage_start", 1), ("stage_end", 1),
                ("stage_start", 2), ("stage_end", 2),
                ("pipeline_end", None),
            ],
        )
        self.assertEqual(events[0]["stages"], ["ok", "ok"])

    def test_failure_emits_and_reraises(self):
        events = []
        with self.assertRaises(RuntimeError):
            _pipeline(Ok(), Boom()).run(progress=events.append)
        failed = events[-1]
        self.assertEqual(failed["event"], "stage_failed")
        self.assertEqual(failed["name"], "boom")
        self.assertIn("falhou", failed["error"])

    def test_broken_callback_does_not_break_run(self):
        def bad(_event):
            raise ValueError("observador quebrado")

        ctx = _pipeline(Ok()).run(progress=bad)
        self.assertIsNotNone(ctx)

    def test_progress_is_optional(self):
        self.assertIsNotNone(_pipeline(Ok()).run())


if __name__ == "__main__":
    unittest.main()
