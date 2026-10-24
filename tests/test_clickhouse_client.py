"""Testes do cliente HTTP do ClickHouse (formatação, parse, requisição)."""
import unittest
from unittest import mock

import pandas as pd

import tests._bootstrap  # noqa: F401
from pipeline.io.clickhouse import ClickHouseClient, ClickHouseError


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestEnsureFormat(unittest.TestCase):
    def test_appends_format_when_missing(self):
        self.assertEqual(
            ClickHouseClient._ensure_format("SELECT 1"),
            "SELECT 1 FORMAT JSONCompact",
        )

    def test_does_not_duplicate_format(self):
        self.assertEqual(
            ClickHouseClient._ensure_format("SELECT 1 FORMAT JSONCompact"),
            "SELECT 1 FORMAT JSONCompact",
        )

    def test_strips_trailing_semicolon(self):
        self.assertEqual(
            ClickHouseClient._ensure_format("SELECT 1;"),
            "SELECT 1 FORMAT JSONCompact",
        )


class TestParse(unittest.TestCase):
    def test_parses_json_compact(self):
        body = (
            '{"meta": [{"name": "link_id", "type": "String"},'
            ' {"name": "tick", "type": "UInt64"}],'
            ' "data": [["3", 100], ["5", 200]], "rows": 2}'
        )
        df = ClickHouseClient._parse(body)
        self.assertEqual(list(df.columns), ["link_id", "tick"])
        self.assertEqual(df.shape, (2, 2))
        self.assertEqual(df.iloc[0]["link_id"], "3")

    def test_raises_on_clickhouse_exception(self):
        body = '{"exception": "DB::Exception: Table htc.vehicle_link_events does not exist"}'
        with self.assertRaises(ClickHouseError):
            ClickHouseClient._parse(body)

    def test_raises_on_non_json(self):
        with self.assertRaises(ClickHouseError):
            ClickHouseClient._parse("not json at all")


class TestFetch(unittest.TestCase):
    def test_builds_url_with_database_and_params(self):
        client = ClickHouseClient(
            host="ch.example.com",
            port=9000,
            database="mydb",
            user="alice",
            password="s3cret",
        )
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["request"] = request
            return FakeResponse(
                b'{"meta": [{"name": "x", "type": "UInt8"}], "data": [[1]]}'
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.query("SELECT {v:UInt64} FORMAT JSONCompact", {"v": 7})

        self.assertIn("database=mydb", captured["url"])
        self.assertIn("param_v=7", captured["url"])
        # O urllib capitaliza a chave do header; compara de forma case-insensitive.
        headers = {k.lower(): v for k, v in captured["request"].headers.items()}
        self.assertEqual(headers["x-clickhouse-user"], "alice")
        self.assertEqual(headers["x-clickhouse-key"], "s3cret")

    def test_raises_clickhouse_error_on_http_error(self):
        import urllib.error

        client = ClickHouseClient()

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 500, "Internal Server Error", {}, None
            )

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(ClickHouseError):
                client.query("SELECT 1")


class TestPing(unittest.TestCase):
    def test_ping_true(self):
        client = ClickHouseClient()

        def fake_urlopen(request, timeout=None):
            return FakeResponse(b'{"meta": [], "data": [[1]]}')

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertTrue(client.ping())

    def test_ping_false_on_error(self):
        client = ClickHouseClient()

        def fake_urlopen(request, timeout=None):
            raise OSError("connection refused")

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertFalse(client.ping())


if __name__ == "__main__":
    unittest.main()
