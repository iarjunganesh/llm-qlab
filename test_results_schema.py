"""Regression tests for CSV schema migration.

The bug these exist to prevent: adding two columns to CSV_FIELDS made every
existing 18-column row match no known schema, so normalize_row returned None
for all of them and the results file read as empty. The cause was treating
"older schema" and "untrustworthy data" as one category, which is only true
for the layouts produced by the wall-clock timing bug.
"""

import csv

import pytest

import results_schema as rs
from results_schema import (
    COMPATIBLE_SCHEMAS,
    CSV_FIELDS,
    LEGACY_MARKER,
    LEGACY_SCHEMAS,
    UNKNOWN,
    normalize_row,
)

CURRENT_ROW = [
    "llama-2-7b-chat.Q4_K_M", "llama2", "Q4_K_M", "99", "5", "16", "127",
    "168.35", "48.65", "19.79", "3.37", "95.46", "21.44",
    "4302.0", "4937.0", "0.982", "resident",
    "P0", "12001", "none", "3.06", "3891.9", "perf_counters",
]

# 20 columns: everything current except the clock-telemetry trio.
PRE_CLOCK_ROW = [
    "llama-2-7b-chat.Q4_K_M", "llama2", "Q4_K_M", "99", "5", "16", "127",
    "168.35", "48.65", "19.79", "3.37", "95.46", "21.44",
    "4302.0", "4937.0", "0.982", "resident", "3.06", "3891.9", "perf_counters",
]

# 18 columns: everything current except vram_residency and offload_state.
PRE_RESIDENCY_ROW = [
    "llama-2-7b-chat.Q4_K_M", "llama2", "Q4_K_M", "99", "5", "16", "127",
    "168.35", "48.65", "19.79", "3.37", "95.46", "21.44",
    "4302.0", "4937.0", "3.06", "3891.9", "perf_counters",
]

LEGACY_11_ROW = [
    "llama-2-7b-chat.Q4_K_M", "llama2", "Q4_K_M", "16", "127",
    "3.99", "31.70", "4937", "3.06", "86.94", "3891.9",
]

LEGACY_10_ROW = [
    "llama-2-7b-chat.Q4_K_M", "llama2", "Q4_K_M", "16", "127",
    "3.99", "31.70", "4937", "3.06", "3891.9",
]

LEGACY_9_ROW = [
    "llama-2-7b-chat.Q4_K_M", "Q4_K_M", "16", "127",
    "3.99", "31.70", "4937", "3.06", "3891.9",
]

ALL_GENERATIONS = [
    ("current", CURRENT_ROW),
    ("pre_clock", PRE_CLOCK_ROW),
    ("pre_residency", PRE_RESIDENCY_ROW),
    ("legacy_11", LEGACY_11_ROW),
    ("legacy_10", LEGACY_10_ROW),
    ("legacy_9", LEGACY_9_ROW),
]


@pytest.mark.parametrize("name,row", ALL_GENERATIONS)
def test_every_generation_is_readable(name, row):
    """No known layout may be dropped. This is the regression."""
    assert normalize_row(row) is not None, f"{name} row was dropped"


@pytest.mark.parametrize("name,row", ALL_GENERATIONS)
def test_normalized_row_has_exact_current_fields(name, row):
    assert set(normalize_row(row)) == set(CSV_FIELDS)


def test_schema_lengths_are_unambiguous():
    """Migration dispatches on column count, so counts must be distinct."""
    lengths = [len(CSV_FIELDS)]
    lengths += [len(s) for s in COMPATIBLE_SCHEMAS]
    lengths += [len(s) for s in LEGACY_SCHEMAS]
    assert len(lengths) == len(set(lengths)), f"ambiguous column counts: {lengths}"


def test_widened_schema_preserves_measurements():
    """A compatible row predates columns; its numbers are still good."""
    row = normalize_row(PRE_RESIDENCY_ROW)
    assert row["decode_tps"] == 19.79
    assert row["prefill_tps"] == 168.35
    assert row["timing_source"] == "perf_counters"
    # Absent columns get sentinels, not silent zeros.
    assert row["vram_residency"] == UNKNOWN
    assert row["offload_state"] == "unknown"


def test_pre_clock_schema_preserves_measurements():
    """Adding clock telemetry must not invalidate the sweep before it."""
    row = normalize_row(PRE_CLOCK_ROW)
    assert row["decode_tps"] == 19.79
    assert row["vram_residency"] == 0.982
    assert row["offload_state"] == "resident"
    assert row["timing_source"] == "perf_counters"
    # Unrecorded clock state must read as unknown, never as a plausible P0:
    # inventing it would erase exactly the evidence these columns exist for.
    assert row["pstate"] == "unknown"
    assert row["mem_clock_mhz"] == UNKNOWN
    assert row["throttle_reasons"] == "unknown"


def test_current_schema_roundtrips_new_columns():
    row = normalize_row(CURRENT_ROW)
    assert row["vram_residency"] == 0.982
    assert row["offload_state"] == "resident"
    assert row["pstate"] == "P0"
    assert row["mem_clock_mhz"] == 12001
    assert row["throttle_reasons"] == "none"


@pytest.mark.parametrize("name,row", [
    ("legacy_11", LEGACY_11_ROW),
    ("legacy_10", LEGACY_10_ROW),
    ("legacy_9", LEGACY_9_ROW),
])
def test_legacy_throughput_is_invalidated_not_carried(name, row):
    """Wall-clock-bug rows keep their identity but lose their bad numbers."""
    out = normalize_row(row)
    assert out["timing_source"] == LEGACY_MARKER
    assert out["decode_tps"] == UNKNOWN
    assert out["prefill_tps"] == UNKNOWN
    assert out["model_name"] == "llama-2-7b-chat.Q4_K_M"


def test_unknown_width_is_rejected():
    assert normalize_row(["only", "three", "columns"]) is None


def test_migration_of_pre_residency_file_keeps_rows(tmp_path, monkeypatch):
    """End-to-end: the failure was an emptied file, so assert on the file."""
    csv_path = tmp_path / "benchmark_results.csv"
    pre_residency_header = next(
        s for s in COMPATIBLE_SCHEMAS if len(s) == len(PRE_RESIDENCY_ROW)
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(pre_residency_header)
        writer.writerow(PRE_RESIDENCY_ROW)
    monkeypatch.setattr(rs, "CSV_PATH", csv_path)

    rs.ensure_csv_schema()

    with open(csv_path, newline="") as f:
        migrated = list(csv.DictReader(f))
    assert len(migrated) == 1, "migration emptied the results file"
    assert migrated[0]["decode_tps"] == "19.79"
    assert migrated[0]["offload_state"] == "unknown"
