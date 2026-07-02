from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from numbers import Integral, Real
from time import perf_counter
from typing import Iterable, Mapping


VERBOSE = False
PROGRESS_WIDTH = 28


def set_verbose(enabled: bool) -> None:
    global VERBOSE
    VERBOSE = enabled


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {sec:.1f}s"
    if minutes:
        return f"{minutes}m {sec:.1f}s"
    return f"{sec:.1f}s"


def log_line(message: str, *, tag: str = "INFO") -> None:
    print(f"[{now_text()}] [{tag}] {message}", flush=True)


def progress_text(current: int, total: int, *, width: int = PROGRESS_WIDTH) -> str:
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({current / total:.0%})"


@contextmanager
def log_stage(name: str):
    start = perf_counter()
    log_progress(name, 0, 1, extra="start")
    try:
        yield
    except Exception as exc:
        log_line(f"{name} failed | elapsed {format_duration(perf_counter() - start)} | {exc}", tag="ERROR")
        raise
    else:
        log_progress(name, 1, 1, extra=f"done in {format_duration(perf_counter() - start)}")


def log_progress(
    name: str,
    current: int,
    total: int,
    *,
    start_time: float | None = None,
    extra: str = "",
    always: bool = True,
) -> None:
    if not always and not VERBOSE:
        return
    elapsed = f" | elapsed {format_duration(perf_counter() - start_time)}" if start_time is not None else ""
    suffix = f" | {extra}" if extra else ""
    log_line(f"{name} {progress_text(current, total)}{elapsed}{suffix}", tag="PROG")


def print_table(title: str, rows: Iterable[Mapping[str, object]], *, always: bool = False) -> None:
    if not always and not VERBOSE:
        return
    rows = list(rows)
    if not rows:
        return
    headers = list(rows[0].keys())
    table = [[_format_cell(row.get(h, "")) for h in headers] for row in rows]
    widths = [
        max(len(str(h)), *(len(row[i]) for row in table))
        for i, h in enumerate(headers)
    ]
    log_line(title, tag="DATA")
    print("  " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)), flush=True)
    print("  " + "-+-".join("-" * w for w in widths), flush=True)
    for row in table:
        print("  " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))), flush=True)


def print_summary(title: str, values: Mapping[str, object]) -> None:
    log_line(title, tag="RESULT")
    for key, value in values.items():
        print(f"  {key}: {_format_cell(value)}", flush=True)


def _format_cell(value: object) -> str:
    if isinstance(value, Real) and not isinstance(value, (Integral, bool)):
        return f"{float(value):.6f}"
    return str(value)
