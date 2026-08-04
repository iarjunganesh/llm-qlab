"""
results_schema.py — CSV schema and migration for benchmark results.

Deliberately free of any llama_cpp import so that analysis tooling
(compare_quants.py) runs on machines without a CUDA build of
llama-cpp-python — you should be able to plot results anywhere.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

UNKNOWN = -1.0

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"

CSV_FIELDS = [
    "model_name",
    "model_family",
    "quant_type",
    "n_gpu_layers",
    "n_runs",
    "prompt_tokens",
    "generated_tokens",
    "prefill_tps",
    "prefill_tps_std",
    "decode_tps",
    "decode_tps_std",
    "ttft_ms",
    "ttft_ms_std",
    "vram_delta_mb",
    "vram_total_mb",
    "vram_residency",
    "offload_state",
    "pstate",
    "mem_clock_mhz",
    "throttle_reasons",
    "load_time_s",
    "model_size_mb",
    "timing_source",
]

# Schemas from earlier revisions, newest first. Rows in these formats carry
# throughput produced by a wall-clock fallback bug: "prompt_tps" was really
# prompt_tokens/total_time, and "gen_tps" divided generation by prefill and
# decode combined. Neither is recoverable after the fact, so migration keeps
# each row's identity and drops the invalid metrics rather than carrying
# wrong numbers forward under new column names.
LEGACY_SCHEMAS = [
    [
        "model_name", "model_family", "quant_type", "prompt_tokens",
        "generated_tokens", "prompt_tps", "gen_tps", "vram_mb",
        "load_time_s", "ttft_ms", "model_size_mb",
    ],
    [
        "model_name", "model_family", "quant_type", "prompt_tokens",
        "generated_tokens", "prompt_tps", "gen_tps", "vram_mb",
        "load_time_s", "model_size_mb",
    ],
    [
        "model_name", "quant_type", "prompt_tokens", "generated_tokens",
        "prompt_tps", "gen_tps", "vram_mb", "load_time_s", "model_size_mb",
    ],
]

# Older layouts whose *data* is still valid — they simply predate columns added
# later. Rows are widened with defaults and keep their measured values.
#
# This distinction matters: an earlier revision treated "older schema" and
# "untrustworthy data" as the same thing, so adding two columns silently
# invalidated every existing row and the results file read as empty.
COMPATIBLE_SCHEMAS = [
    # pre-clock-telemetry (before pstate / mem_clock_mhz / throttle_reasons)
    [
        "model_name", "model_family", "quant_type", "n_gpu_layers", "n_runs",
        "prompt_tokens", "generated_tokens", "prefill_tps", "prefill_tps_std",
        "decode_tps", "decode_tps_std", "ttft_ms", "ttft_ms_std",
        "vram_delta_mb", "vram_total_mb", "vram_residency", "offload_state",
        "load_time_s", "model_size_mb", "timing_source",
    ],
    # pre-residency (before vram_residency / offload_state)
    [
        "model_name", "model_family", "quant_type", "n_gpu_layers", "n_runs",
        "prompt_tokens", "generated_tokens", "prefill_tps", "prefill_tps_std",
        "decode_tps", "decode_tps_std", "ttft_ms", "ttft_ms_std",
        "vram_delta_mb", "vram_total_mb", "load_time_s", "model_size_mb",
        "timing_source",
    ],
]

# Rows whose throughput was measured without clock-state verification. The
# numbers are not wrong the way LEGACY_SCHEMAS rows are wrong — they were timed
# correctly — but they carry no evidence the card was in a consistent power
# state, and a 2026-08-04 audit found up to 27% drift between runs that the
# harness had reported as converged. Excluded from charts and tables, kept in
# the file so the history is auditable.
UNSTABLE_MARKER = "unstable_clocks"

LEGACY_MARKER = "legacy_invalid"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_row(values: list[str]) -> dict | None:
    """Coerce a CSV row from any known schema into the current one."""
    row: dict[str, Any]
    if len(values) == len(CSV_FIELDS):
        row = dict(zip(CSV_FIELDS, values))
        legacy = False
    elif any(len(values) == len(s) for s in COMPATIBLE_SCHEMAS):
        # Widened, not invalidated: the measurements are still good.
        schema = next(s for s in COMPATIBLE_SCHEMAS if len(values) == len(s))
        row = dict(zip(schema, values))
        legacy = False
    else:
        for schema in LEGACY_SCHEMAS:
            if len(values) == len(schema):
                row = dict(zip(schema, values))
                legacy = True
                break
        else:
            return None

    if legacy:
        row["prefill_tps"] = UNKNOWN
        row["decode_tps"] = UNKNOWN
        row["timing_source"] = LEGACY_MARKER
        row["vram_total_mb"] = row.get("vram_mb", UNKNOWN)
        # Old rows recorded whole-board usage, not usage attributable to the
        # model, so there is no delta to recover.
        row["vram_delta_mb"] = UNKNOWN
        row["n_runs"] = 1
        row["n_gpu_layers"] = UNKNOWN

    return {
        "model_name": row.get("model_name", "unknown"),
        "model_family": row.get("model_family") or "unknown",
        "quant_type": row.get("quant_type") or "unknown",
        "n_gpu_layers": safe_int(row.get("n_gpu_layers", UNKNOWN), -1),
        "n_runs": safe_int(row.get("n_runs", 1), 1),
        "prompt_tokens": safe_int(row.get("prompt_tokens", 0)),
        "generated_tokens": safe_int(row.get("generated_tokens", 0)),
        "prefill_tps": round(safe_float(row.get("prefill_tps", UNKNOWN), UNKNOWN), 2),
        "prefill_tps_std": round(safe_float(row.get("prefill_tps_std", 0.0)), 2),
        "decode_tps": round(safe_float(row.get("decode_tps", UNKNOWN), UNKNOWN), 2),
        "decode_tps_std": round(safe_float(row.get("decode_tps_std", 0.0)), 2),
        "ttft_ms": round(safe_float(row.get("ttft_ms", UNKNOWN), UNKNOWN), 2),
        "ttft_ms_std": round(safe_float(row.get("ttft_ms_std", 0.0)), 2),
        "vram_delta_mb": round(safe_float(row.get("vram_delta_mb", UNKNOWN), UNKNOWN), 1),
        "vram_total_mb": round(safe_float(row.get("vram_total_mb", UNKNOWN), UNKNOWN), 1),
        "vram_residency": round(safe_float(row.get("vram_residency", UNKNOWN), UNKNOWN), 3),
        "offload_state": row.get("offload_state") or "unknown",
        # Rows predating clock telemetry get "unknown" rather than a plausible
        # default: not recording the P-state is exactly what made the earlier
        # sweep uninvestigable, and inventing one would repeat that.
        "pstate": row.get("pstate") or "unknown",
        "mem_clock_mhz": round(safe_float(row.get("mem_clock_mhz", UNKNOWN), UNKNOWN), 0),
        "throttle_reasons": row.get("throttle_reasons") or "unknown",
        "load_time_s": round(safe_float(row.get("load_time_s", 0.0)), 2),
        "model_size_mb": round(safe_float(row.get("model_size_mb", 0.0)), 1),
        "timing_source": row.get("timing_source") or "unknown",
    }


def ensure_csv_schema() -> None:
    """Migrate the results CSV to the current schema, backing up the original."""
    if not CSV_PATH.exists():
        return

    with open(CSV_PATH, "r", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    needs_migration = rows[0] != CSV_FIELDS
    normalized, skipped, invalidated = [], 0, 0
    for raw in rows[1:]:
        row = normalize_row(raw)
        if row is None:
            skipped += 1
            continue
        if row["timing_source"] == LEGACY_MARKER:
            invalidated += 1
        normalized.append(row)

    if not needs_migration and skipped == 0:
        return

    backup = CSV_PATH.with_name(f"{CSV_PATH.stem}.legacy.{int(time.time())}{CSV_PATH.suffix}")
    CSV_PATH.replace(backup)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)

    print(f"[info] Migrated benchmark CSV to latest schema: {CSV_PATH}")
    print(f"[info] Original preserved at: {backup}")
    if invalidated:
        print(
            f"[warn] {invalidated} row(s) predate the prefill-timing fix; their "
            "throughput columns are marked unmeasured. Re-run to repopulate."
        )
    if skipped:
        print(f"[warn] Skipped {skipped} malformed row(s) during migration.")


def save_result(result: dict, *, attempts: int = 6) -> None:
    """Append a result row, retrying while the CSV is locked.

    On Windows the results file is routinely held open by Excel, an editor, or
    anything tailing it — and a transient lock here throws away a benchmark run
    that already cost minutes of GPU time. Retry with backoff, and if the file
    is still unavailable, write the row to a sidecar so the measurement
    survives for manual recovery rather than being lost to a formatting detail.
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            ensure_csv_schema()
            write_header = not CSV_PATH.exists()
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                writer.writerow(result)
            print(f"Result saved to {CSV_PATH}")
            return
        except PermissionError as exc:
            last_error = exc
            delay = 0.5 * (2 ** attempt)
            print(f"[warn] {CSV_PATH} is locked (attempt {attempt + 1}/{attempts}); retrying in {delay:.1f}s")
            time.sleep(delay)

    rescue = CSV_PATH.with_name(f"{CSV_PATH.stem}.rescued.{int(time.time())}{CSV_PATH.suffix}")
    with open(rescue, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    print(f"[error] Could not append to {CSV_PATH} after {attempts} attempts: {last_error}")
    print(f"[error] Row preserved at {rescue} — merge it manually once the lock clears.")
