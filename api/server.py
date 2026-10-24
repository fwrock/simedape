"""Ponto de entrada para rodar o servidor da API (Flask).

Uso::

    python -m pipeline.api.server --host 0.0.0.0 --port 5000 \\
        --configs-dir configs --max-workers 2
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Servidor de gerenciamento do pipeline SimEDaPE."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--configs-dir",
        default=None,
        help="Diretório p/ persistir as configurações (JSON). Vazio = só memória.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Máximo de runs do pipeline executando em paralelo.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Import local: só exige Flask quando o servidor é de fato iniciado.
    from pipeline.api.app import create_app
    from pipeline.api.jobs import JobManager
    from pipeline.api.store import ConfigStore

    app = create_app(
        config_store=ConfigStore(storage_dir=args.configs_dir),
        job_manager=JobManager(max_workers=args.max_workers),
    )
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
