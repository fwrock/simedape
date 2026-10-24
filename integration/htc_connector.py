"""HTC simulator integration connectors.

Provides two classes:

* :class:`HtcSimulationConnector` — triggers an HTC simulation run via HTTP
  and polls until the run completes (or times out).
* :class:`HtcResultFetcher` — collects output files from the HTC simulation
  result directory, including Parquet files readable by
  :func:`pipeline.io.reader.read_htc_parquet`.

Both classes are **optional**: they require ``aiohttp`` at runtime.  If
``aiohttp`` is missing, importing this module raises :exc:`ImportError` with a
helpful message.  The :mod:`pipeline.integration` package catches this and
emits a :exc:`ImportWarning` instead.

Usage example (synchronous wrapper around async methods)::

    import asyncio
    from pipeline.integration.htc_connector import HtcSimulationConnector

    connector = HtcSimulationConnector(
        base_url="http://localhost:8080",
        scenario_id="sp_morning_peak",
    )
    asyncio.run(connector.run_and_wait(timeout=600))
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

try:
    import aiohttp
except ImportError as _exc:
    raise ImportError(
        "HtcSimulationConnector requires 'aiohttp'. "
        "Install it with: pip install aiohttp"
    ) from _exc

logger = logging.getLogger(__name__)


class HtcSimulationConnector:
    """Triggers an HTC simulation run and waits for it to finish.

    The connector uses the HTC REST API (two endpoints):

    * ``POST {base_url}/api/simulation/start`` — starts a simulation run.
      The request body is a JSON object with at least ``scenario_id``.
    * ``GET  {base_url}/api/simulation/status/{run_id}`` — polls run status.
      The response is expected to contain a ``"status"`` field with values
      ``"running"``, ``"completed"``, or ``"failed"``.

    Parameters
    ----------
    base_url : str
        Base URL of the HTC API server, e.g. ``"http://localhost:8080"``.
    scenario_id : str
        HTC scenario identifier to simulate.
    extra_payload : dict, optional
        Additional key/value pairs to include in the start-simulation POST
        body (e.g. selective instantiation file path, random seed).
    poll_interval : float
        Seconds between status-poll requests.  Default: ``5.0``.
    """

    _START_ENDPOINT = "/api/simulation/start"
    _STATUS_ENDPOINT = "/api/simulation/status/{run_id}"

    def __init__(
        self,
        base_url: str,
        scenario_id: str,
        extra_payload: dict[str, Any] | None = None,
        poll_interval: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.scenario_id = scenario_id
        self.extra_payload: dict[str, Any] = extra_payload or {}
        self.poll_interval = poll_interval

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self, session: "aiohttp.ClientSession") -> str:
        """Triggers a simulation run and returns the ``run_id`` string.

        Parameters
        ----------
        session : aiohttp.ClientSession
            An active aiohttp session.

        Returns
        -------
        str
            The ``run_id`` assigned by the HTC server.

        Raises
        ------
        RuntimeError
            If the server returns a non-2xx status or the response body does
            not contain a ``"run_id"`` field.
        """
        url = self.base_url + self._START_ENDPOINT
        payload = {"scenario_id": self.scenario_id, **self.extra_payload}

        logger.info("Starting HTC simulation: POST %s | payload=%s", url, payload)
        async with session.post(url, json=payload) as resp:
            if resp.status not in (200, 201, 202):
                text = await resp.text()
                raise RuntimeError(
                    f"HTC /start returned HTTP {resp.status}: {text}"
                )
            data = await resp.json()

        run_id = data.get("run_id")
        if not run_id:
            raise RuntimeError(
                f"HTC /start response missing 'run_id' field: {data}"
            )

        logger.info("Simulation started — run_id=%s", run_id)
        return str(run_id)

    async def poll_until_done(
        self,
        session: "aiohttp.ClientSession",
        run_id: str,
        timeout: float = 3600.0,
    ) -> dict[str, Any]:
        """Polls the status endpoint until the run is completed or fails.

        Parameters
        ----------
        session : aiohttp.ClientSession
            An active aiohttp session.
        run_id : str
            The run identifier returned by :meth:`start`.
        timeout : float
            Maximum seconds to wait before raising :exc:`TimeoutError`.

        Returns
        -------
        dict
            The final status response body from the HTC API.

        Raises
        ------
        TimeoutError
            If ``timeout`` seconds elapse before the run completes.
        RuntimeError
            If the run status is ``"failed"``.
        """
        url = self.base_url + self._STATUS_ENDPOINT.format(run_id=run_id)
        deadline = time.monotonic() + timeout

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"HTC simulation run_id={run_id} did not finish within {timeout}s"
                )

            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                else:
                    logger.warning(
                        "Status poll returned HTTP %s for run_id=%s — retrying",
                        resp.status,
                        run_id,
                    )
                    data = {}

            status = data.get("status", "unknown")
            logger.debug("run_id=%s status=%s", run_id, status)

            if status == "completed":
                logger.info("Simulation run_id=%s completed.", run_id)
                return data
            elif status == "failed":
                raise RuntimeError(
                    f"HTC simulation run_id={run_id} failed: {data}"
                )

            await asyncio.sleep(self.poll_interval)

    async def run_and_wait(self, timeout: float = 3600.0) -> dict[str, Any]:
        """Convenience method: starts the simulation and waits for completion.

        Creates its own ``aiohttp.ClientSession``.  Use :meth:`start` /
        :meth:`poll_until_done` directly if you manage a session externally.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for the run to finish.

        Returns
        -------
        dict
            Final status payload from the HTC API.
        """
        async with aiohttp.ClientSession() as session:
            run_id = await self.start(session)
            return await self.poll_until_done(session, run_id, timeout=timeout)


class HtcResultFetcher:
    """Fetches HTC simulation output files from a result directory.

    HTC writes output into a directory whose structure is::

        <output_dir>/
            <scenario_id>/
                <run_id>/
                    *.parquet        — per-tick agent telemetry
                    summary.json     — aggregate run metadata

    Parameters
    ----------
    output_dir : str
        Root directory where HTC stores simulation results.
    scenario_id : str
        Scenario identifier (sub-directory name).
    """

    def __init__(self, output_dir: str, scenario_id: str) -> None:
        self.output_dir = output_dir
        self.scenario_id = scenario_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def result_path(self, run_id: str) -> Path:
        """Returns the Path to the result directory for a given run."""
        return Path(self.output_dir) / self.scenario_id / str(run_id)

    def list_parquet_files(self, run_id: str) -> list[Path]:
        """Returns a sorted list of Parquet files for the given run.

        Parameters
        ----------
        run_id : str
            The HTC run identifier.

        Returns
        -------
        list[Path]
            Sorted list of ``.parquet`` file paths found in the run directory.
            Empty list if the directory does not exist or has no Parquet files.
        """
        run_dir = self.result_path(run_id)
        if not run_dir.is_dir():
            logger.warning("HTC result directory not found: %s", run_dir)
            return []
        return sorted(run_dir.glob("*.parquet"))

    def read_results(self, run_id: str) -> "pd.DataFrame":
        """Reads all Parquet output files for a run into a single DataFrame.

        Uses :func:`pipeline.io.reader.read_htc_parquet` for each file so
        that the returned DataFrame has the canonical SimEDaPE schema
        (including the ``mode`` column).

        Parameters
        ----------
        run_id : str
            The HTC run identifier.

        Returns
        -------
        pandas.DataFrame
            Concatenated DataFrame of all Parquet output files.

        Raises
        ------
        FileNotFoundError
            If no Parquet files are found in the run directory.
        """
        import pandas as pd
        from pipeline.io.reader import read_htc_parquet

        parquet_files = self.list_parquet_files(run_id)
        if not parquet_files:
            raise FileNotFoundError(
                f"No Parquet files found for run_id={run_id} in "
                f"{self.result_path(run_id)}"
            )

        frames = []
        for pf in parquet_files:
            logger.debug("Reading HTC Parquet: %s", pf)
            frames.append(read_htc_parquet(str(pf)))

        df = pd.concat(frames, ignore_index=True)
        logger.info(
            "HTC results loaded: run_id=%s, %d rows, %d files",
            run_id,
            len(df),
            len(parquet_files),
        )
        return df

    def read_summary(self, run_id: str) -> dict[str, Any]:
        """Reads the ``summary.json`` metadata file for a run.

        Parameters
        ----------
        run_id : str
            The HTC run identifier.

        Returns
        -------
        dict
            Parsed JSON summary.  Empty dict if the file does not exist.
        """
        import json
        summary_path = self.result_path(run_id) / "summary.json"
        if not summary_path.is_file():
            logger.warning("HTC summary.json not found: %s", summary_path)
            return {}
        with summary_path.open("r") as fh:
            return json.load(fh)
