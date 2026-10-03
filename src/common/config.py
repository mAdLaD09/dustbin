"""Validated YAML configuration shared by training and Pi applications."""
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[2]

class ConfigError(ValueError):
    pass

def load_config(path: str | Path = ROOT / "config.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    validate_config(cfg)
    return cfg

def validate_config(cfg: dict[str, Any]) -> None:
    required = {"classes", "dataset", "camera", "model", "prediction", "servo", "system"}
    missing = required - set(cfg)
    if missing:
        raise ConfigError(f"Missing configuration section(s): {', '.join(sorted(missing))}")
    if set(cfg["classes"]) < {"wet", "dry"}:
        raise ConfigError("classes must define at least wet and dry")
    if cfg["dataset"]["train_fraction"] + cfg["dataset"]["validation_fraction"] >= 1:
        raise ConfigError("dataset fractions must leave a positive test fraction")
    if int(cfg["camera"]["index"]) < 0:
        raise ConfigError("camera.index must be zero or greater")
    if not 0 <= float(cfg["model"]["confidence_threshold"]) <= 1:
        raise ConfigError("model.confidence_threshold must be between 0 and 1")
    for key in ("neutral_angle", "wet_angle", "dry_angle"):
        value = float(cfg["servo"][key])
        if not 0 <= value <= 180:
            raise ConfigError(f"servo.{key} must be between 0 and 180")
    if int(cfg["prediction"]["minimum_votes"]) > int(cfg["prediction"]["window_size"]):
        raise ConfigError("prediction.minimum_votes cannot exceed window_size")
