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


class TestFlaggedRunsCannotDeleteVerifiedOnes:
    """The real llama-2 Q5_K_M case.

    Measured clean at 65.31 t/s, re-measured hours later with a 2.4% clock
    spread, and vanished from every chart: the good row was discarded as
    superseded and the flagged row filtered out behind it, leaving the series
    empty. An unstable measurement is an absence of evidence, not evidence the
    earlier verified one was wrong.
    """

    def test_earlier_verified_row_wins_over_a_later_flagged_one(self):
        df = pd.DataFrame([
            row(decode=65.31, source="perf_counters"),
            row(decode=65.40, source="unstable_clocks"),
        ])
        out = _keep_latest_per_config(df)
        assert len(out) == 1
        assert out.iloc[0]["decode_tps"] == 65.31
        assert out.iloc[0]["timing_source"] == "perf_counters"

    def test_the_series_does_not_disappear(self):
        """The symptom that surfaced the bug: a hole in the chart."""
        df = pd.DataFrame([
            row(quant="Q4_K_M", decode=74.5, source="perf_counters"),
            row(quant="Q5_K_M", decode=65.31, source="perf_counters"),
            row(quant="Q5_K_M", decode=65.40, source="unstable_clocks"),
        ])
        out = _keep_latest_per_config(df)
        publishable = out[out["timing_source"] == "perf_counters"]
        assert set(publishable["quant_type"]) == {"Q4_K_M", "Q5_K_M"}

    def test_a_newer_verified_row_still_wins(self):
        """Recency must still beat staleness among publishable rows."""
        df = pd.DataFrame([
            row(decode=51.17, source="perf_counters"),
            row(decode=65.31, source="perf_counters"),
        ])
        assert _keep_latest_per_config(df).iloc[0]["decode_tps"] == 65.31

    def test_all_flagged_keeps_the_newest_so_the_flag_stays_visible(self):
        df = pd.DataFrame([
            row(decode=54.45, source="unstable_clocks"),
            row(decode=65.40, source="unstable_clocks"),
        ])
        out = _keep_latest_per_config(df)
        assert len(out) == 1
        assert out.iloc[0]["decode_tps"] == 65.40
        assert out.iloc[0]["timing_source"] == "unstable_clocks"

    def test_a_refused_row_does_not_shadow_a_measured_one(self):
        df = pd.DataFrame([
            row(decode=45.71, source="perf_counters"),
            row(decode=-1.0, source="skipped_insufficient_vram"),
        ])
        assert _keep_latest_per_config(df).iloc[0]["decode_tps"] == 45.71


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
