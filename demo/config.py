import yaml
from pathlib import Path

def get_config_path() -> Path:
    """Return the absolute path to the demo configuration file.
    The file lives in the same directory as this module (demo/config.py).
    """
    return Path(__file__).resolve().parent / "demo_config.yaml"

def load_config() -> dict:
    """Load and return the demo configuration as a dictionary.
    Raises a clear FileNotFoundError if the file does not exist.
    """
    cfg_path = get_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Demo configuration file not found: {cfg_path}\n"
            "Expected location: demo/demo_config.yaml"
        )
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)
