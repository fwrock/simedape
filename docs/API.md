# HTTP API

O SimEDaPE expõe uma API REST (Flask) para **gerenciar configurações**, **listar estágios** e
**executar o pipeline** em background, com rastreamento de status/logs/resultado.

## Rodando o servidor

```bash
pip install -e ".[api]"                 # instala o Flask
python -m pipeline.api.server \
  --host 0.0.0.0 --port 5000 \
  --configs-dir configs \               # opcional: persiste configs em configs/<name>.json
  --max-workers 2
```

Sem `--configs-dir`, as configurações ficam apenas em memória. O servidor responde JSON em todos
os endpoints (inclusive erros).

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET`  | `/api/health` | Health check. |
| `GET`  | `/api/stages` | Lista os estágios disponíveis (nome + 1ª linha da docstring). |
| `GET`  | `/api/configs` | Lista as configurações nomeadas. |
| `POST` | `/api/configs` | Cria/sobrescreve uma config (`{"name": ..., "config": {...}}`). |
| `GET`  | `/api/configs/<name>` | Retorna uma config. |
| `PUT`  | `/api/configs/<name>` | Atualiza uma config (corpo = dict da config). |
| `DELETE` | `/api/configs/<name>` | Remove uma config. |
| `POST` | `/api/runs` | Submete um run (ver abaixo). |
| `GET`  | `/api/runs` | Lista runs (status, tempos, logs, resultado). |
| `GET`  | `/api/runs/<run_id>` | Detalha um run. |
| `DELETE` | `/api/runs/<run_id>` | Cancela um run ainda na fila. |

## Criando uma configuração

```bash
curl -X POST http://localhost:5000/api/configs \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "carla",
        "config": {
          "scenario_name": "carla_local",
          "scenario_dir": "path/to/csv",
          "data_source": "clickhouse",
          "clickhouse": {"database": "htc", "table": "vehicle_link_events"}
        }
      }'
```

O campo `config` usa as mesmas chaves do bloco `config:` do YAML (ver
[docs/CONFIGURATION.md](CONFIGURATION.md)).

## Executando um pipeline

```bash
curl -X POST http://localhost:5000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "run1",
        "stages": ["data_load", "process_dataset", "clustering"],
        "config": "carla",                       # nome de uma config já salva, OU
        # "config": {"scenario_name": "..."},    # um dict inline
        "stage_params": {"clustering": {"n_clusters": 48}}
      }'
# → 202 {"run_id": "..."}
```

O run é executado em uma thread de background. Acompanhe com:

```bash
curl http://localhost:5000/api/runs/<run_id>
# → {"run_id": ..., "status": "completed", "result": {...}, "logs": "..."}
```

`status` assume os valores `queued`, `running`, `completed`, `failed` ou `cancelled`. `logs`
contém o `stdout`/`stderr` capturado do pipeline (que usa `print` extensivamente), e `result`
traz um resumo (`scenario_name`, `output_dir`, `n_series`, `max_size` e, quando presentes,
`scenario_metrics`/`estimation_results`).

## Notas

- Executar o pipeline real requer as dependências numéricas (`tslearn`, `scikit-learn`,
  `dtaidistance`, etc.); os endpoints de `/stages` e `/runs` retornam `503` com mensagem clara se
  essas dependências não estiverem disponíveis.
- Cancelamento (`DELETE /api/runs/<id>`) só é possível enquanto o run está **na fila**; runs já
  iniciados retornam `409`.
