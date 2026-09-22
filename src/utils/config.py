from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config