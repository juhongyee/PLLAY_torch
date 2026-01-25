# src/config.py
from __future__ import annotations

from dataclasses import dataclass, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints
import re

import yaml


# =========================
# Dataclass schema
# =========================

@dataclass(frozen=True)
class ImageSpec:
    channels: int
    height: int
    width: int


@dataclass(frozen=True)
class AugmentSpec:
    gaussian_noise_std: float = 0.0
    normalize: bool = True
    corrupt_prob: float = 0.0       
    noise_prob: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    name: str
    root: str
    batch_size: int
    num_workers: int = 4
    pin_memory: bool = True
    image: ImageSpec = ImageSpec(channels=1, height=28, width=28)
    augmentation: AugmentSpec = AugmentSpec()
    train_topo_path: Optional[str] = None
    test_topo_path: Optional[str] = None


@dataclass(frozen=True)
class ImageBackboneConfig:
    type: str
    in_dim: int
    hidden_dims: List[int]
    out_dim: int
    dropout: float = 0.0


@dataclass(frozen=True)
class TopoConfig:
    # NOTE: v0에서는 topo 계산을 "feature-only"로 써도 되게 설계하되,
    # extractor/embedder 이름과 파라미터는 config에 남겨 둡니다.
    extractor: str = "none"
    embedder: str = "none"
    diag_max_points: int = 64
    landscape_num_layers: int = 4
    landscape_bins: int = 100
    out_dim: int = 64


@dataclass(frozen=True)
class FusionConfig:
    type: str = "concat"
    out_dim: int = 0  # optional: can be derived later


@dataclass(frozen=True)
class HeadConfig:
    hidden_dims: List[int]
    dropout: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int
    image_backbone: ImageBackboneConfig
    topo: TopoConfig = TopoConfig()
    fusion: FusionConfig = FusionConfig()
    head: HeadConfig = HeadConfig(hidden_dims=[128], dropout=0.0)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float = 0.0
    amp: bool = True
    grad_clip_norm: float = 1.0 # (Optional)
    log_every: int = 50
    eval_every: int = 1
    save_every: int = 1


@dataclass(frozen=True)
class AppConfig:
    seed: int = 42
    device: str = "cuda"
    run_name: str = "exp"
    output_dir: str = "runs/${run_name}"
    data: DataConfig = None  # type: ignore[assignment]
    model: ModelConfig = None  # type: ignore[assignment]
    train: TrainConfig = None  # type: ignore[assignment]


# =========================
# Exceptions
# =========================

class ConfigError(ValueError):
    pass


# =========================
# YAML / dict helpers
# =========================

def _read_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ConfigError(f"YAML root must be a mapping(dict): {p}")
    return obj


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dicts.
    - dict + dict => deep merge
    - else => override wins
    """
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


_TEMPLATE_RE = re.compile(r"\$\{([A-Za-z0-9_.-]+)\}")


def _resolve_templates(value: Any, context: Dict[str, Any]) -> Any:
    """
    Simple ${key} template resolver.
    Supports:
      - ${run_name} (top-level)
      - ${data.root} (dot path)
    """
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            key = m.group(1)
            resolved = _get_by_dotpath(context, key)
            if resolved is None:
                raise ConfigError(f"Unresolved template key: {key}")
            return str(resolved)
        return _TEMPLATE_RE.sub(repl, value)

    if isinstance(value, dict):
        return {k: _resolve_templates(v, context) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_templates(v, context) for v in value]

    return value


def _get_by_dotpath(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# =========================
# Dataclass construction
# =========================

def _require(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required key '{key}' in {where}")
    return d[key]


def _build_dataclass(cls, d: Dict[str, Any], where: str):
    # convert dict from YAML to dataclass instance
    if not is_dataclass(cls):
        raise ConfigError(f"Internal error: {cls} is not a dataclass")
    if not isinstance(d, dict):
        raise ConfigError(f"Expected dict for {cls.__name__} at {where}, got {type(d)}")

    type_hints = get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for f in fields(cls):
        name = f.name
        field_type = type_hints.get(name, f.type)
        if name in d:
            val = d[name]
            # Nested dataclass?
            if is_dataclass(field_type):
                kwargs[name] = _build_dataclass(field_type, val, where=f"{where}.{name}")
            else:
                try:
                    if field_type == float and not isinstance(val, float):
                        val = float(val)
                    elif field_type == int and not isinstance(val, int):
                        val = int(val)
                    elif field_type == bool and not isinstance(val, bool):
                        # "true", "True" 문자열 처리 (필요시)
                        if isinstance(val, str):
                            val = val.lower() == "true"
                except (ValueError, TypeError) as e:
                    raise ConfigError(
                        f"Type mismatch for '{name}' at {where}. Expected {field_type}, got {val} ({type(val)})"
                    ) from e
                    
                kwargs[name] = val
        else:
            # allow default / default_factory
            pass

    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"Failed to construct {cls.__name__} at {where}: {e}") from e


def _build_app_config(cfg: Dict[str, Any]) -> AppConfig:
    # Required top-level sections
    data_d = _require(cfg, "data", "root")
    model_d = _require(cfg, "model", "root")
    train_d = _require(cfg, "train", "root")

    data = _build_dataclass(DataConfig, data_d, "data")
    model = _build_dataclass(ModelConfig, model_d, "model")
    train = _build_dataclass(TrainConfig, train_d, "train")

    # App-level keys (optional w/ defaults)
    app_kwargs = {k: v for k, v in cfg.items() if k not in ("data", "model", "train", "defaults")}
    app = AppConfig(data=data, model=model, train=train, **app_kwargs)
    return app


# =========================
# Defaults composition
# =========================

def _load_with_defaults(config_path: Path, config_root: Path) -> Dict[str, Any]:
    """
    Supports:
      default.yaml:
        defaults:
          - data: mnist
          - model: topo_mlp
          - train: basic
        seed: 42
        ...

    It will load:
      configs/data/mnist.yaml, configs/model/topo_mlp.yaml, configs/train/basic.yaml
    """
    base = _read_yaml(config_path)

    defaults = base.get("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, list):
        raise ConfigError(f"'defaults' must be a list in {config_path}")

    merged: Dict[str, Any] = {}
    for item in defaults:
        if not isinstance(item, dict) or len(item) != 1:
            raise ConfigError(
                f"Each defaults entry must be a single-key mapping like {{data: mnist}}. Got: {item}"
            )
        section, name = next(iter(item.items()))
        if not isinstance(section, str) or not isinstance(name, str):
            raise ConfigError(f"Invalid defaults entry: {item}")

        subpath = config_root / section / f"{name}.yaml"
        subcfg = _read_yaml(subpath)
        # Inject into merged at section key
        merged = _deep_merge(merged, {section: subcfg})

    # Then merge base (base overrides defaults)
    merged = _deep_merge(merged, base)

    return merged


# =========================
# Validation
# =========================

def validate_config(cfg: AppConfig) -> None:
    # data
    if cfg.data.batch_size <= 0:
        raise ConfigError("data.batch_size must be > 0")
    if cfg.data.num_workers < 0:
        raise ConfigError("data.num_workers must be >= 0")
    if cfg.data.image.channels not in (1, 3):
        raise ConfigError("data.image.channels must be 1 or 3")
    if cfg.data.image.height <= 0 or cfg.data.image.width <= 0:
        raise ConfigError("data.image.height/width must be > 0")
    if cfg.data.augmentation.gaussian_noise_std < 0:
        raise ConfigError("data.augmentation.gaussian_noise_std must be >= 0")

    # model
    if cfg.model.num_classes < 2:
        raise ConfigError("model.num_classes must be >= 2")
    if cfg.model.image_backbone.in_dim <= 0:
        raise ConfigError("model.image_backbone.in_dim must be > 0")
    if cfg.model.image_backbone.out_dim <= 0:
        raise ConfigError("model.image_backbone.out_dim must be > 0")
    if cfg.model.topo.out_dim < 0:
        raise ConfigError("model.topo.out_dim must be >= 0")
    if cfg.model.topo.diag_max_points <= 0:
        raise ConfigError("model.topo.diag_max_points must be > 0")
    if cfg.model.topo.landscape_bins <= 0:
        raise ConfigError("model.topo.landscape_bins must be > 0")
    if cfg.model.topo.landscape_num_layers <= 0:
        raise ConfigError("model.topo.landscape_num_layers must be > 0")

    # train
    if cfg.train.epochs <= 0:
        raise ConfigError("train.epochs must be > 0")
    if cfg.train.lr <= 0:
        raise ConfigError("train.lr must be > 0")
    if cfg.train.weight_decay < 0:
        raise ConfigError("train.weight_decay must be >= 0")
    if cfg.train.grad_clip_norm < 0:
        raise ConfigError("train.grad_clip_norm must be >= 0")
    if cfg.train.log_every <= 0:
        raise ConfigError("train.log_every must be > 0")
    if cfg.train.eval_every <= 0:
        raise ConfigError("train.eval_every must be > 0")
    if cfg.train.save_every <= 0:
        raise ConfigError("train.save_every must be > 0")

    # app
    if cfg.seed < 0:
        raise ConfigError("seed must be >= 0")
    if not isinstance(cfg.device, str) or not cfg.device:
        raise ConfigError("device must be a non-empty string")
    if not isinstance(cfg.run_name, str) or not cfg.run_name:
        raise ConfigError("run_name must be a non-empty string")
    if not isinstance(cfg.output_dir, str) or not cfg.output_dir:
        raise ConfigError("output_dir must be a non-empty string")


# =========================
# Public API
# =========================

def load_config(
    config_path: Union[str, Path],
    *,
    config_root: Optional[Union[str, Path]] = None,
    resolve_templates: bool = True,
    validate: bool = True,
) -> AppConfig:
    """
    Load YAML configs into AppConfig.

    Typical usage:
      cfg = load_config("configs/default.yaml")

    Args:
      config_path: path to a YAML file (usually configs/default.yaml)
      config_root: root directory containing {data,model,train}/ subfolders.
                   If None, inferred as parent of config_path.
      resolve_templates: resolve ${...} placeholders using merged dict context.
      validate: run validation checks.

    Returns:
      AppConfig (dataclass)
    """
    config_path = Path(config_path)
    if config_root is None:
        config_root = config_path.parent
    config_root = Path(config_root)

    merged_dict = _load_with_defaults(config_path, config_root)

    if resolve_templates:
        # Use dict context for template resolution
        merged_dict = _resolve_templates(merged_dict, merged_dict)

    cfg = _build_app_config(merged_dict)

    if validate:
        validate_config(cfg)

    return cfg


def dump_resolved_config(cfg: AppConfig, output_path: Union[str, Path]) -> None:
    """
    Save resolved dataclass config to YAML (for reproducibility).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)


# =========================
# Small self-check (optional)
# =========================
if __name__ == "__main__":
    # Example:
    #   python -m src.config configs/default.yaml
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m src.config <path/to/config.yaml>")
    cfg = load_config(sys.argv[1])
    print(cfg)
