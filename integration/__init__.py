"""simedape.integration — optional connectors to the HTC simulator.

This sub-package provides helpers for triggering HTC simulation runs and
fetching their output.  All classes are *optional*: if the required
third-party library (``aiohttp``) is not installed, importing this package
will emit a warning rather than raising an error.
"""
from __future__ import annotations

import warnings

try:
    from pipeline.integration.htc_connector import HtcSimulationConnector, HtcResultFetcher
    __all__ = ["HtcSimulationConnector", "HtcResultFetcher"]
except ImportError as _exc:
    warnings.warn(
        f"simedape.integration could not import HTC connectors: {_exc}. "
        "Install 'aiohttp' to enable HTTP-based simulation triggering.",
        ImportWarning,
        stacklevel=2,
    )
    __all__ = []
