"""
SimEDaPE CLI

Uso:
    python run.py --config properties/carla_local.properties \\
                  --stages data_load,process_dataset,clustering,simulation_points

    python run.py --pipeline pipeline.yml

    python run.py --config properties/carla_local.properties \\
                  --stages warping_paths \\
                  --stage-params warping_paths.n_jobs=4,warping_paths.load_from_disk=true

    python run.py --config properties/carla_local.properties \\
                  --stages estimation \\
                  --stage-params estimation.methods=avg,sample \\
                              estimation.second_scenario_dir=input/carla-experiments/experiments/1#trips-100000-20-50p/
"""
import argparse
import sys
import os

# Garante que o diretório raiz do projeto esteja no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pipeline import Pipeline, _build_stage_registry
from pipeline.config import SimEDaPEConfig


def parse_stage_params(raw_params: list[str]) -> dict:
    """Transforma ['stage.param=value', ...] em {'stage': {'param': value}}."""
    result = {}
    for item in raw_params:
        key, _, value = item.partition("=")
        stage, _, param = key.partition(".")
        if stage not in result:
            result[stage] = {}
        # Conversão de tipos básicos
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        elif "," in value and not value.startswith("["):
            value = [v.strip() for v in value.split(",")]
        result[stage][param] = value
    return result


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="SimEDaPE — Simulation Estimation using Dynamic Partial Execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", "-c",
        metavar="FILE",
        help="Arquivo .properties de configuração (ex: properties/carla_local.properties)",
    )
    group.add_argument(
        "--pipeline", "-p",
        metavar="FILE",
        help="Arquivo YAML de definição do pipeline (ex: pipeline.yml)",
    )

    parser.add_argument(
        "--stages", "-s",
        metavar="STAGES",
        help="Estágios a executar, separados por vírgula. "
             "Disponíveis: data_load, process_dataset, clustering, "
             "simulation_points, warping_paths, scenario_metrics, estimation",
    )
    parser.add_argument(
        "--stage-params",
        nargs="*",
        metavar="STAGE.PARAM=VALUE",
        default=[],
        help="Parâmetros para estágios específicos. Ex: warping_paths.n_jobs=4",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Lista os estágios disponíveis e sai.",
    )

    args = parser.parse_args()

    if args.list_stages:
        registry = _build_stage_registry()
        print("Estágios disponíveis:")
        for name, cls in registry.items():
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            print(f"  {name:<20} {doc}")
        return

    if args.pipeline:
        pipeline = Pipeline.from_yaml(args.pipeline)
        pipeline.run()
        return

    # Modo --config + --stages
    if not args.stages:
        parser.error("--stages é obrigatório quando usando --config")

    stage_names = [s.strip() for s in args.stages.split(",")]
    stage_params = parse_stage_params(args.stage_params or [])
    registry = _build_stage_registry()

    config_path = args.config
    if not os.path.exists(config_path):
        candidate = os.path.join("properties", config_path)
        if os.path.exists(candidate):
            config_path = candidate
        else:
            print(f"Erro: arquivo de configuração não encontrado: '{config_path}'")
            print(f"  Tentativa também em: '{candidate}'")
            sys.exit(1)

    config = SimEDaPEConfig.from_properties_file(config_path)
    p = Pipeline(config)

    for stage_name in stage_names:
        if stage_name not in registry:
            print(f"Erro: estágio '{stage_name}' desconhecido.")
            print(f"Disponíveis: {', '.join(registry.keys())}")
            sys.exit(1)
        params = stage_params.get(stage_name, {})
        p.add(registry[stage_name](**params))

    p.run()


if __name__ == "__main__":
    main()
