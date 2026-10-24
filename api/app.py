"""Aplicação Flask e rotas da API de gerenciamento do pipeline SimEDaPE."""
from __future__ import annotations

import json

from flask import Blueprint, Flask, current_app, jsonify, request

from pipeline.api import exports
from pipeline.api.jobs import JobManager
from pipeline.api.store import ConfigStore
from pipeline.config import SimEDaPEConfig
from pipeline.io.source import describe_source
from pipeline.pipeline import Pipeline, _build_stage_registry

api = Blueprint("api", __name__, url_prefix="/api")


def create_app(
    config_store: ConfigStore | None = None,
    job_manager: JobManager | None = None,
    stage_registry: dict | None = None,
) -> Flask:
    """Cria a aplicação Flask, permitindo injetar serviços (útil em testes)."""
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    app.extensions["simedape"] = {
        "config_store": config_store or ConfigStore(),
        "job_manager": job_manager or JobManager(),
        "stage_registry": stage_registry,  # None → carrega sob demanda
    }
    app.register_blueprint(api)
    _register_error_handlers(app)
    return app


def _services() -> dict:
    return current_app.extensions["simedape"]


def _registry() -> dict:
    services = _services()
    registry = services["stage_registry"]
    if registry is None:
        registry = _build_stage_registry()
        services["stage_registry"] = registry
    return registry


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "Não encontrado."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return jsonify({"error": "Método não permitido."}), 405


# --------------------------------------------------------------------------
# Health / Stages
# --------------------------------------------------------------------------

@api.get("/health")
def health():
    return jsonify({"status": "ok"})


@api.get("/stages")
def list_stages():
    try:
        registry = _registry()
    except Exception as exc:  # dependências do pipeline indisponíveis
        return jsonify({"error": f"Não foi possível carregar os estágios: {exc}"}), 503
    return jsonify([
        {"name": name, "description": (cls.__doc__ or "").strip().split("\n")[0]}
        for name, cls in registry.items()
    ])


# --------------------------------------------------------------------------
# Configurations
# --------------------------------------------------------------------------

@api.get("/configs")
def list_configs():
    store = _services()["config_store"]
    return jsonify([
        {"name": name, "config": store.get(name).to_dict()} for name in store.list()
    ])


@api.post("/configs")
def create_config():
    body = request.get_json(silent=True) or {}
    name = body.get("name")
    cfg = body.get("config")
    if not isinstance(name, str) or not name:
        return jsonify({"error": "Campo 'name' (string) é obrigatório."}), 400
    if not isinstance(cfg, dict):
        return jsonify({"error": "Campo 'config' (dict) é obrigatório."}), 400
    try:
        config = SimEDaPEConfig.from_dict(cfg)
    except Exception as exc:
        return jsonify({"error": f"Config inválida: {exc}"}), 400
    store = _services()["config_store"]
    try:
        store.save(name, config)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"name": name, "config": config.to_dict()}), 201


@api.get("/configs/<name>")
def get_config(name):
    store = _services()["config_store"]
    try:
        config = store.get(name)
    except KeyError:
        return jsonify({"error": f"Configuração não encontrada: {name!r}"}), 404
    return jsonify({"name": name, "config": config.to_dict()})


@api.put("/configs/<name>")
def update_config(name):
    store = _services()["config_store"]
    if name not in store.list():
        return jsonify({"error": f"Configuração não encontrada: {name!r}"}), 404
    body = request.get_json(silent=True) or {}
    cfg = body.get("config", body)
    if not isinstance(cfg, dict):
        return jsonify({"error": "Corpo deve ser um dict de configuração."}), 400
    try:
        config = SimEDaPEConfig.from_dict(cfg)
    except Exception as exc:
        return jsonify({"error": f"Config inválida: {exc}"}), 400
    store.save(name, config)
    return jsonify({"name": name, "config": config.to_dict()})


@api.delete("/configs/<name>")
def delete_config(name):
    store = _services()["config_store"]
    if not store.delete(name):
        return jsonify({"error": f"Configuração não encontrada: {name!r}"}), 404
    return jsonify({"deleted": name})

# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

@api.post("/runs")
def create_run():
    body = request.get_json(silent=True) or {}
    stages = body.get("stages")
    if not isinstance(stages, list) or not stages:
        return jsonify({"error": "Campo 'stages' (lista não vazia) é obrigatório."}), 400

    try:
        registry = _registry()
    except Exception as exc:
        return jsonify({"error": f"Não foi possível carregar os estágios: {exc}"}), 503

    unknown = [s for s in stages if s not in registry]
    if unknown:
        return jsonify({"error": f"Estágios desconhecidos: {unknown}"}), 400

    store = _services()["config_store"]
    cfg_ref = body.get("config")
    try:
        if isinstance(cfg_ref, str):
            config = store.get(cfg_ref)
        elif isinstance(cfg_ref, dict):
            config = SimEDaPEConfig.from_dict(cfg_ref)
        else:
            config = SimEDaPEConfig.from_dict({})
    except KeyError:
        return jsonify({"error": f"Configuração não encontrada: {cfg_ref!r}"}), 404
    except Exception as exc:
        return jsonify({"error": f"Config inválida: {exc}"}), 400

    stage_params = body.get("stage_params", {}) or {}

    try:
        source = describe_source(config)
    except ValueError as exc:
        return jsonify({"error": f"Config inválida: {exc}"}), 400

    meta = {
        "scenario_name": config.scenario_name,
        "output_dir": config.output_dir,
        "data_source": source,
        "config": cfg_ref if isinstance(cfg_ref, str) else None,
        "stages": stages,
        "stage_params": stage_params,
        "clustering_file": config.clustering.output_file,
    }

    def run(job) -> dict:
        pipeline = Pipeline(config)
        for stage_name in stages:
            params = stage_params.get(stage_name, {}) or {}
            pipeline.add(registry[stage_name](**params))
        ctx = pipeline.run(progress=job.on_progress)
        return _summarize_context(ctx)

    run_id = _services()["job_manager"].submit(
        run, name=body.get("name", ""), meta=meta, pass_job=True
    )
    return jsonify({"run_id": run_id}), 202


@api.get("/runs")
def list_runs():
    return jsonify(exports.sanitize([j.to_dict() for j in _services()["job_manager"].list()]))


@api.get("/runs/<run_id>")
def get_run(run_id):
    try:
        job = _services()["job_manager"].get(run_id)
    except KeyError:
        return jsonify({"error": f"Run não encontrado: {run_id!r}"}), 404
    return jsonify(exports.sanitize(job.to_dict()))


@api.delete("/runs/<run_id>")
def cancel_run(run_id):
    manager = _services()["job_manager"]
    try:
        cancelled = manager.cancel(run_id)
    except KeyError:
        return jsonify({"error": f"Run não encontrado: {run_id!r}"}), 404
    if not cancelled:
        return jsonify({"error": "Run já iniciado/finalizado; cancelamento indisponível."}), 409
    return jsonify({"run_id": run_id, "status": "cancelled"})



# --------------------------------------------------------------------------
# Progresso e exportação de resultados de um run
# --------------------------------------------------------------------------

def _job_or_404(run_id):
    try:
        return _services()["job_manager"].get(run_id), None
    except KeyError:
        return None, (jsonify({"error": f"Run não encontrado: {run_id!r}"}), 404)


def _stage_prefix(job, stage: str, key: str, default: str) -> str:
    """Prefixo dos arquivos de um estágio: ``?prefix=`` > stage_params do run > default."""
    override = request.args.get("prefix")
    if override:
        return override
    params = (job.meta.get("stage_params") or {}).get(stage) or {}
    return params.get(key, default)


@api.get("/runs/<run_id>/progress")
def run_progress(run_id):
    job, err = _job_or_404(run_id)
    if err:
        return err
    return jsonify(exports.sanitize({
        "run_id": job.run_id,
        "name": job.name,
        "status": job.status,
        "data_source": job.meta.get("data_source"),
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        **job.progress_dict(),
    }))


@api.get("/runs/<run_id>/artifacts")
def run_artifacts(run_id):
    job, err = _job_or_404(run_id)
    if err:
        return err
    out_dir = job.meta.get("output_dir")
    return jsonify({
        "run_id": job.run_id,
        "output_dir": out_dir,
        "data_source": job.meta.get("data_source"),
        "artifacts": exports.list_artifacts(out_dir, job.meta.get("clustering_file", "clustering")),
    })


@api.get("/runs/<run_id>/estimation")
def run_estimation(run_id):
    job, err = _job_or_404(run_id)
    if err:
        return err
    out_dir = job.meta.get("output_dir")
    prefix = _stage_prefix(job, "estimation", "output_prefix", "result")
    method = request.args.get("method")
    if not exports.valid_name(prefix):
        return jsonify({"error": "Prefixo inválido."}), 400
    if not method:
        return jsonify({"run_id": run_id, **exports.estimation_overview(out_dir, prefix)})
    if not exports.valid_name(method):
        return jsonify({"error": "Método inválido."}), 400
    try:
        offset = max(int(request.args.get("offset", 0)), 0)
        limit = request.args.get("limit")
        limit = max(int(limit), 0) if limit is not None else None
    except ValueError:
        return jsonify({"error": "'offset' e 'limit' devem ser inteiros."}), 400
    try:
        detail = exports.estimation_detail(
            out_dir, method, prefix, request.args.get("metric"), offset, limit
        )
    except exports.ArtifactNotFound:
        return jsonify({"error": f"Estimativa não encontrada: {method!r}"}), 404
    return jsonify({"run_id": run_id, **detail})


@api.get("/runs/<run_id>/comparison")
def run_comparison(run_id):
    job, err = _job_or_404(run_id)
    if err:
        return err
    out_dir = job.meta.get("output_dir")
    prefix = _stage_prefix(job, "compare", "output_prefix", "compare")
    method = request.args.get("method")
    if not exports.valid_name(prefix):
        return jsonify({"error": "Prefixo inválido."}), 400
    if not method:
        return jsonify({"run_id": run_id, **exports.comparison_matrix(out_dir, prefix)})
    if not exports.valid_name(method):
        return jsonify({"error": "Método inválido."}), 400
    try:
        detail = exports.comparison_detail(out_dir, method, prefix)
    except exports.ArtifactNotFound:
        return jsonify({"error": f"Comparação não encontrada: {method!r}"}), 404
    return jsonify({"run_id": run_id, "method": method, **detail})


def _summarize_context(ctx) -> dict:
    summary = {
        "scenario_name": ctx.config.scenario_name,
        "output_dir": ctx.output_dir,
    }
    if getattr(ctx, "street_series", None) is not None:
        summary["n_series"] = len(ctx.street_series)
    if getattr(ctx, "max_size", None) is not None:
        summary["max_size"] = ctx.max_size
    for key in ("scenario_metrics", "estimation_results"):
        value = getattr(ctx, key, None)
        if value:
            try:
                json.dumps(value)
                summary[key] = value
            except (TypeError, ValueError):
                summary[key] = f"<não serializável: {type(value).__name__}>"
    return summary


