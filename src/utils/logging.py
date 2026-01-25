from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import time


def setup_logger(
    name: str = "pllay_torch",
    *,
    log_dir: Optional[Union[str, Path]] = None,
    filename: str = "run.log",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Create a logger with optional console + file handlers.

    Args:
      name: logger name
      log_dir: if provided and file=True, logs to log_dir/filename
      filename: log file name
      level: logging level
      console: enable console handler
      file: enable file handler (requires log_dir)

    Returns:
      configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs from root

    # Prevent adding handlers multiple times (common in notebooks / repeated calls)
    if getattr(logger, "_configured", False):
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if file:
        if log_dir is None:
            if console:
                # 콘솔 핸들러가 있어야 warning이 보임
                logger.warning("file=True but log_dir=None. File logging is disabled.")
        else:
            p = Path(log_dir)
            p.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(p / filename, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    return logger


def format_metrics(metrics: Dict[str, Any], *, prefix: str = "", precision: int = 4) -> str:
    """
    Convert a metrics dict to a compact string.
    Example: {"loss": 0.12345, "acc": 0.9876} -> "loss=0.1235 acc=0.9876"
    """
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.{precision}f}")
        else:
            parts.append(f"{k}={v}")
    msg = " ".join(parts)
    return f"{prefix}{msg}".strip()


class MetricLogger:
    """
    Minimal helper to print/log metrics at steps/epochs.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = time.time()

    def log_step(self, epoch: int, step: int, metrics: Dict[str, Any]) -> None:
        msg = format_metrics(metrics, prefix=f"[epoch={epoch} step={step}] ")
        self.logger.info(msg)

    def log_epoch(self, epoch: int, split: str, metrics: Dict[str, Any]) -> None:
        elapsed = time.time() - self.start_time
        msg = format_metrics(metrics, prefix=f"[epoch={epoch} {split}] ")
        self.logger.info(f"{msg} (elapsed={elapsed:.1f}s)")
