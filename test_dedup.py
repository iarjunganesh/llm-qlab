"""Tests for collapsing repeated measurements of one configuration.

The bug these exist to prevent: benchmark.py appends rather than replaces, so
re-running a configuration — the harness's own advice when a row comes back
flagged unstable_clocks — left both rows in the file. Which one reached a chart
depended on row order, so a superseded number could be published after it had
already been re-measured and corrected.
"""

import pandas as pd

from compare_quants import _keep_latest_per_config


def row(name="llama-2-7b-chat.Q5_K_M", quant="Q5_K_M", layers=99,
        decode=51.17, source="perf_counters"):
    return {
        "model_name": name, "model_family": "llama2", "quant_type": quant,
        "n_gpu_layers": layers, "decode_tps": decode, "timing_source": source,
    }


def test_later_row_supersedes_earlier():
    df = pd.DataFrame([
        row(decode=54.45, source="unstable_clocks"),
        row(decode=51.17, source="perf_counters"),
    ])
    out = _keep_latest_per_config(df)
    assert len(out) == 1
    assert out.iloc[0]["decode_tps"] == 51.17
    assert out.iloc[0]["timing_source"] == "perf_counters"


def test_the_real_mistral_case_keeps_only_the_clean_run():
    """Three appended attempts; only the last is publishable."""
    df = pd.DataFrame([
        row(name="mistral-7b-instruct-v0.1.Q5_K_M", decode=53.11,
            source="unstable_clocks"),
        row(name="mistral-7b-instruct-v0.1.Q5_K_M", decode=49.26,
            source="unstable_clocks"),
        row(name="mistral-7b-instruct-v0.1.Q5_K_M", decode=49.07,
            source="perf_counters"),
    ])
    out = _keep_latest_per_config(df)
    assert len(out) == 1
    assert out.iloc[0]["decode_tps"] == 49.07


def test_distinct_configurations_are_all_kept():
    df = pd.DataFrame([
        row(name="a", quant="Q4_K_M"),
        row(name="b", quant="Q5_K_M"),
        row(name="c", quant="Q8_0"),
    ])
    assert len(_keep_latest_per_config(df)) == 3


def test_same_model_at_different_offload_is_not_a_duplicate():
    """The offload ladder measures one model at many layer counts."""
    df = pd.DataFrame([row(layers=0), row(layers=16), row(layers=99)])
    assert len(_keep_latest_per_config(df)) == 3


def test_missing_key_columns_pass_through_untouched():
    df = pd.DataFrame([{"model_name": "a", "decode_tps": 1.0}])
    assert len(_keep_latest_per_config(df)) == 1


def test_index_is_reset_so_positional_lookups_stay_valid():
    df = pd.DataFrame([row(decode=1.0), row(decode=2.0), row(name="other")])
    out = _keep_latest_per_config(df)
    assert list(out.index) == list(range(len(out)))
