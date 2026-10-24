import json
import os


def load_json(path):
    """Carrega um arquivo JSON. A extensão .json é adicionada se não estiver presente."""
    if not path.endswith(".json"):
        path = path + ".json"
    with open(path, "r") as f:
        return json.loads(f.read())


def save_json(path, content):
    """Salva conteúdo como JSON. Sobrescreve se existir."""
    if not path.endswith(".json"):
        path = path + ".json"
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        f.write(json.dumps(content))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
