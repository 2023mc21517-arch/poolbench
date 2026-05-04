"""
poolbench/logger.py
Centralised step-by-step logging with timestamps and GPU memory reporting.

Usage
-----
from poolbench.logger import get_logger, gpu_mem_str, log_step

log = get_logger("poolbench")
log.info(f"Starting extraction  GPU: {gpu_mem_str()}")
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator


# ── Formatter that includes timestamps ────────────────────────────────────────

class _TimestampFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%H:%M:%S")
        level_sym = {
            logging.DEBUG:   "DBG",
            logging.INFO:    "INF",
            logging.WARNING: "WRN",
            logging.ERROR:   "ERR",
        }.get(record.levelno, record.levelname[:3])
        msg = super().format(record)
        return f"[{ts}] [{level_sym}] {msg}"


# ── Singleton registry ─────────────────────────────────────────────────────────

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str = "poolbench",
               log_file: str | Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that writes to stdout (and optionally a file).
    Calling this multiple times with the same name returns the same logger.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = _TimestampFormatter("%(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger


# ── GPU memory helpers ────────────────────────────────────────────────────────

def gpu_mem_str(device: str = "cuda:0") -> str:
    """
    Return a compact GPU memory string, e.g. "12.3/40.0 GB (31%)".
    Returns "GPU:N/A" if CUDA is not available or device index is invalid.
    """
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return "GPU:N/A"
        idx = int(device.split(":")[-1]) if ":" in device else 0
        alloc = torch.cuda.memory_allocated(idx) / 1e9
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        pct   = 100.0 * alloc / total if total > 0 else 0.0
        return f"{alloc:.1f}/{total:.0f} GB ({pct:.0f}%)"
    except Exception:
        return "GPU:N/A"


def free_gpu_memory(device: str = "cuda:0") -> None:
    """Release any cached (but not currently allocated) CUDA memory."""
    try:
        import gc       # noqa: PLC0415
        import torch    # noqa: PLC0415
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def find_free_gpu(min_free_gb: float = 20.0,
                  logger: "logging.Logger | None" = None) -> str:
    """
    Scan all available CUDA devices and return the one with the most free memory.

    Logs a table of all GPUs so you can see what the cluster looks like:
        [00:01:23] [INF]  GPU scan:
        [00:01:23] [INF]    cuda:0  free=38.1 GB  total=40.0 GB  (95% free)  <-- BUSY (1.9 GB used)
        [00:01:23] [INF]    cuda:1  free=39.8 GB  total=40.0 GB  (100% free) <-- SELECTED
        ...

    Raises RuntimeError if no GPU has at least `min_free_gb` free.

    Parameters
    ----------
    min_free_gb : float
        Minimum free VRAM (in GB) required to consider a GPU usable.
        Default 20 GB — enough for any 7B/8B/9B model in bf16.
    logger : logging.Logger, optional
        If provided, GPU table is logged to this logger.  Otherwise printed to stdout.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        _emit("torch not installed — defaulting to cuda:0", logger)
        return "cuda:0"

    if not torch.cuda.is_available():
        _emit("CUDA not available — defaulting to cpu", logger)
        return "cpu"

    n = torch.cuda.device_count()
    if n == 0:
        _emit("No CUDA devices found — defaulting to cpu", logger)
        return "cpu"

    _emit(f"GPU scan: {n} device(s) found", logger)
    best_device: str = ""
    best_free_gb: float = -1.0

    for i in range(n):
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        free_gb  = free_bytes  / 1e9
        total_gb = total_bytes / 1e9
        used_gb  = total_gb - free_gb
        pct_free = 100.0 * free_gb / total_gb if total_gb > 0 else 0.0
        props    = torch.cuda.get_device_properties(i)

        status = "SELECTED" if free_gb > best_free_gb else ""
        _emit(
            f"  cuda:{i}  [{props.name}]  "
            f"free={free_gb:.1f} GB  used={used_gb:.1f} GB  total={total_gb:.0f} GB  "
            f"({pct_free:.0f}% free)"
            + (f"  <-- {status}" if status else ""),
            logger,
        )
        # Update best only if status was set (improves logging linearity)
        if free_gb > best_free_gb:
            best_free_gb = free_gb
            best_device  = f"cuda:{i}"
            # Reprint the previous best line without SELECTED tag would be complex;
            # instead we do a second pass below.

    # --- Second pass: build final table with correct SELECTED marker ---
    _emit(f"\n  Final selection: {best_device}  "
          f"free={best_free_gb:.1f} GB", logger)

    if best_free_gb < min_free_gb:
        raise RuntimeError(
            f"No GPU has >= {min_free_gb:.0f} GB free. "
            f"Best found: {best_device} with {best_free_gb:.1f} GB free. "
            f"Kill another job or lower --min_free_gb."
        )

    return best_device


def _emit(msg: str, logger: "logging.Logger | None") -> None:
    """Write to logger.info or print depending on whether a logger was supplied."""
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


# ── Step timing context manager ───────────────────────────────────────────────

@contextmanager
def log_step(logger: logging.Logger,
             label: str,
             device: str = "cuda:0") -> Generator[None, None, None]:
    """
    Context manager that logs step start / end with GPU memory and elapsed time.

    Usage::

        with log_step(log, "extraction llama3_8b concept=hedging", device):
            ...
    """
    mem_before = gpu_mem_str(device)
    t0 = time.perf_counter()
    logger.info(f">>> START  {label}  | GPU before: {mem_before}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        mem_after = gpu_mem_str(device)
        logger.info(f"<<< END    {label}  | GPU after: {mem_after}  "
                    f"elapsed: {elapsed:.1f}s")
