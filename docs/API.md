# HTTP API

O SimEDaPE expõe uma API REST (Flask) para **gerenciar configurações**, **listar estágios** e
**executar o pipeline** em background, com rastreamento de status/logs/resultado.

## Rodando o servidor

```bash
pip install -e ".[api]"                 # instala o Flask
python -m pipeline.api.server \
  --host 0.0.0.0 --port 5000 \
  --configs-dir configs \               # opcional: persiste configs em configs/<name>.json
  --runs-dir runs \                     # opcional: persiste runs em runs/<run_id>.json
  --max-workers 2
```

Sem `--configs-dir` / `--runs-dir`, configurações e runs ficam apenas em memória (e se perdem ao reiniciar). O servidor responde JSON em todos
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
| `GET`  | `/api/runs/<run_id>/progress` | Progresso por estágio (ver abaixo). |
| `GET`  | `/api/runs/<run_id>/artifacts` | Arquivos JSON/PDF gerados em `output_dir`, com o `kind` de cada um. |
| `GET`  | `/api/runs/<run_id>/estimation` | Resultados de estimação por método. |
| `GET`  | `/api/runs/<run_id>/comparison` | Matriz método × métrica, ou o detalhe de um método. |
| `GET`  | `/api/runs/<run_id>/charts` | Catálogo dos gráficos disponíveis para o run. |
| `GET`  | `/api/runs/<run_id>/charts/<chart_id>` | Dados de um gráfico, prontos para um frontend (ver abaixo). |
| `GET`  | `/api/runs/<run_id>/sweep` | Tabela de progressão do sweep: cenário parcial × método. |
| `GET`  | `/api/runs/<run_id>/reports` | Lista os relatórios PDF gerados. |
| `GET`  | `/api/runs/<run_id>/reports/<name>` | Baixa um PDF (`application/pdf`). |

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

## Progresso e logs

`GET /api/runs/<id>/progress` devolve `status`, `data_source`, `total`/`completed`/`percent` (por
estágio, não por série), `current_stage` e, para cada estágio, `status`
(`pending|running|completed|failed`), `started_at` e `elapsed`. `GET /api/runs/<id>` também traz
`progress`, além de `meta` (fonte de dados, `output_dir`, estágios pedidos). Os `logs` ficam
legíveis **durante** a execução (últimos ~2M caracteres) e cada run captura apenas a saída da
própria thread, então runs paralelos não misturam logs.

## Exportação de resultados

Os endpoints abaixo só **leem** arquivos já gravados pelos estágios em `output_dir`; nada é
recalculado. Funcionam igual para runs vindos de CSV ou de ClickHouse. Valores `NaN` (ex.: erro %
com baseline zero) são devolvidos como `null`. Nomes de método/prefixo aceitam só
`[A-Za-z0-9._-]`; outros valores retornam `400`/`404`.

- `GET .../estimation` — resumo por método (`result_<método>.json`): nº de séries e min/média/máx
  por métrica. Com `?method=avg` devolve o resultado completo; `&metric=average_speed` restringe a
  uma métrica e `&offset=`/`&limit=` paginam as séries.
- `GET .../comparison` — matriz com `mape` e `aggregate_error_pct` por métrica × método, mais
  `coverage_pct`. Com `?method=avg` devolve o `compare_avg.json` completo.
- `GET .../artifacts` — lista os arquivos e seu `kind` (`estimation`, `comparison`,
  `real_metrics`, `clustering`, `simulation_points`, `report`, ...).
- O prefixo dos arquivos vem de `stage_params` do run (`estimation.output_prefix`,
  `compare.output_prefix`) ou de `?prefix=`; padrões `result` e `compare`.
- Runs que compartilham o mesmo `output_dir` (mesmo `scenario_name`) enxergam os mesmos arquivos.

## Fonte de dados e cenário parcial

`data_source` (`csv` ou `clickhouse`) vale para **todos** os estágios que leem um cenário:
`data_load` e também o cenário parcial de `estimation`, `pca_estimation` e `sweep`. Nesses
estágios, `second_scenario_dir` (e o `dir` de cada cenário do `sweep`) é um **diretório** em modo
`csv` e um **`simulation_id`** em modo `clickhouse`; vazio (ou `training`/`100%` no sweep) usa o
cenário de treino. A senha do ClickHouse nunca aparece em `meta`/`progress`.

## Dados para gráficos

Os PDFs de `compare`/`sweep` são desenhados a partir dos JSONs que os estágios já gravam. A API
devolve esses mesmos dados em JSON, sem depender de biblioteca de plotagem: o frontend escolhe como
desenhar (Chart.js, ECharts, Recharts...). `GET .../charts` lista o catálogo; cada item traz `id`,
`title`, `type`, `source`, os `params` aceitos e `available` (se os dados existem em `output_dir`).
Os valores vêm sempre na mesma ordem dos relatórios (velocidade, ocupação, veículos, km).

| `chart_id` | `type` | Conteúdo | `params` |
|---|---|---|---|
| `error_aggregate` | `bar` | Erro % agregado (gmean), por métrica e método | — |
| `mape_by_method` | `bar` | MAPE por métrica e método | — |
| `mape_radar` | `radar` | MAPE por métrica (eixos = métricas) | — |
| `coverage` | `bar` | Cobertura (% de séries estimadas) por método | — |
| `error_distribution` | `boxplot` | Distribuição do erro por série (min/q1/mediana/q3/máx/média) | `metric` |
| `real_vs_estimated` | `scatter` | Real × estimado por série (diagonal y = x) | `method`, `metric`, `limit` (padrão 500) |
| `sweep_progression` | `line` | MAPE × cenário parcial (ordenado por %) | `metric`, `methods` |
| `sweep_heatmap` | `heatmap` | MAPE método × cenário | `metric`, `methods` |
| `sweep_coverage` | `line` | Cobertura × cenário parcial | `methods` |
| `cluster_sizes` | `bar` | Séries por cluster | — |
| `centroids` | `line` | Centróides normalizados de cada cluster | — |

`metric` é `average_speed` (padrão), `percentage_of_track`, `average_vehicles` ou `km_traveled`;
`methods` é uma lista separada por vírgula (o sweep gera dezenas de variantes de interpolação/PCA,
então filtre no gráfico de linhas). `real_vs_estimated` amostra séries uniformemente até `limit` e
informa `meta.total_series`/`returned`/`step`.

Formato (`series[].data` alinha com `x.categories`; `null` = sem valor):

```json
{
  "run_id": "...", "id": "mape_by_method", "title": "MAPE por método", "type": "bar", "source": "compare",
  "x": {"label": "Métrica", "categories": ["Velocidade Média (m/s)", "Ocupação (%)", "..."]},
  "y": {"label": "MAPE (%)", "unit": "%"},
  "series": [{"name": "avg", "data": [169.2, 135.0, 135.0, 33.2]}],
  "meta": {}
}
```

Variações por `type`: `boxplot` usa `data` = lista de `{min, q1, median, q3, max, mean}` (uma por
categoria/método); `heatmap` traz `x.categories` (colunas), `y.categories` (linhas) e `values`
(matriz linhas × colunas) no lugar de `series`; `scatter` usa `data` = lista de
`{x, y, label}` e `meta.reference = "identity"`.

`GET .../sweep` devolve a tabela completa: `scenarios` (`label`, `percentage`, ordenados por %),
`methods`, `metrics` e, por método, listas alinhadas aos cenários em `mape[metric][method]`,
`error_pct[metric][method]` e `coverage_pct[method]` (`null` onde o método não rodou).

Os PDFs ficam em `GET .../reports` (lista) e `GET .../reports/<name>` (download); só nomes simples
de `.pdf` em `plots/` são aceitos. Os prefixos dos arquivos seguem `stage_params` do run
(`compare.output_prefix`, `sweep.output_prefix`, `estimation.output_prefix`).

## Notas

- Executar o pipeline real requer as dependências numéricas (`tslearn`, `scikit-learn`,
  `dtaidistance`, etc.); os endpoints de `/stages` e `/runs` retornam `503` com mensagem clara se
  essas dependências não estiverem disponíveis.
- Cancelamento (`DELETE /api/runs/<id>`) só é possível enquanto o run está **na fila**; runs já
  iniciados retornam `409`.
