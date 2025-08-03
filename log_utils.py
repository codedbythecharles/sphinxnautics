"""
logger_utils.py – simple, consistent logger across scripts.
"""
import logging
from pathlib import Path
from datetime import datetime

def init_logger(log_level: str = "INFO",
                log_dir: str | None = None,
                rank: int | None = None) -> logging.Logger:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=getattr(logging, log_level.upper(), "INFO"),
                        format=fmt, datefmt=datefmt,
                        handlers=[logging.StreamHandler()])

    logger = logging.getLogger("sphinxnautics")

    if log_dir and (rank is None or rank == 0):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"run_{stamp}.log")
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)
    return logger
