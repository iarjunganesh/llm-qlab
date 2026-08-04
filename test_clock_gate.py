"""Tests for the GPU clock-state gate.

The bug these exist to prevent: `wait_for_stable_clocks` watched `clocks.sm`
alone and returned True while the *memory* clock was still a P-state low.
Decode is memory-bandwidth-bound, so measurements taken in that window read
~20% slow. On the 2026-08-04 sweep this produced a bimodal decode distribution
with a constant ~1.23x ratio across every quantization format — the signature
of a clock artifact rather than a property of any model — and all 18 runs
reported convergence.
"""

import pytest

import bench_core
from bench_core import (
    MIN_MEM_CLOCK_FRACTION,
    UNKNOWN,
    ClockSampler,
    _modal_pstate,
    _summarize_throttles,
    clocks_are_boosted,
    mem_clock_fraction,
    note_mem_clock,
    reference_max_mem_mhz,
)


@pytest.fixture(autouse=True)
def reset_observed_max(monkeypatch):
    """The observed peak is process-global; isolate it between tests."""
    monkeypatch.setattr(bench_core, "_observed_max_mem_mhz", UNKNOWN)


def state(pstate="P0", mem=12001, mem_max=12001, sm=3090, sm_max=3090,
          throttle="none"):
    return {"pstate": pstate, "sm_mhz": sm, "mem_mhz": mem,
            "sm_max_mhz": sm_max, "mem_max_mhz": mem_max, "throttle": throttle}


class TestBoostDetection:
    def test_full_memory_clock_at_p0_is_boosted(self):
        assert clocks_are_boosted(state()) is True

    def test_low_memory_clock_is_not_boosted_even_at_full_sm_clock(self):
        """The exact blind spot: SM pinned at max, memory a P-state low.

        The old SM-only check returned True here.
        """
        assert clocks_are_boosted(state(mem=9800, sm=3090)) is False

    def test_idle_memory_clock_is_not_boosted(self):
        assert clocks_are_boosted(state(pstate="P5", mem=810)) is False

    def test_p0_label_alone_does_not_qualify(self):
        """A card can report P0 while the memory domain is still ramping."""
        assert clocks_are_boosted(state(pstate="P0", mem=6000)) is False

    def test_missing_telemetry_never_claims_boosted(self):
        """Absent evidence must not be read as evidence of a good state."""
        blank = state(pstate="unknown", mem=UNKNOWN, mem_max=UNKNOWN)
        assert clocks_are_boosted(blank) is False

    def test_threshold_is_inclusive_at_the_boundary(self):
        mem_max = 12001
        assert clocks_are_boosted(
            state(mem=mem_max * MIN_MEM_CLOCK_FRACTION, mem_max=mem_max)) is True

    def test_just_below_threshold_is_rejected(self):
        mem_max = 12001
        assert clocks_are_boosted(
            state(mem=mem_max * (MIN_MEM_CLOCK_FRACTION - 0.01),
                  mem_max=mem_max)) is False


class TestMemClockFraction:
    def test_fraction_is_ratio_of_current_to_max(self):
        assert mem_clock_fraction(state(mem=6000, mem_max=12000)) == 0.5

    def test_unknown_without_telemetry(self):
        assert mem_clock_fraction(state(mem=UNKNOWN, mem_max=UNKNOWN)) == UNKNOWN

    def test_zero_max_does_not_divide_by_zero(self):
        assert mem_clock_fraction(state(mem=100, mem_max=0)) == UNKNOWN


class TestThrottleClassification:
    """SwPowerCap is how boost works, not a fault.

    Treating it as disqualifying rejected every run in 12 attempts while the
    card drew 54 W of a 108 W limit at 62 C.
    """

    def test_sw_power_cap_alone_does_not_disqualify(self):
        assert bench_core._disqualifying_throttles([state(throttle="0x4")]) == set()

    def test_hw_slowdown_disqualifies(self):
        assert bench_core._disqualifying_throttles([state(throttle="0x8")])

    def test_thermal_slowdown_disqualifies(self):
        assert bench_core._disqualifying_throttles([state(throttle="0x20")])
        assert bench_core._disqualifying_throttles([state(throttle="0x40")])

    def test_power_brake_disqualifies(self):
        assert bench_core._disqualifying_throttles([state(throttle="0x80")])

    def test_power_cap_combined_with_thermal_still_disqualifies(self):
        """A benign bit must not mask a disqualifying one set alongside it."""
        assert bench_core._disqualifying_throttles([state(throttle="0x24")]) == {"0x20"}

    def test_clean_and_unknown_samples_disqualify_nothing(self):
        assert bench_core._disqualifying_throttles(
            [state(throttle="none"), state(throttle="unknown")]) == set()


class TestClockCoherence:
    """Accepted runs must have run at comparable clocks for stdev to mean anything."""

    def test_identical_clocks_are_coherent(self):
        coherent, spread = bench_core._clocks_are_coherent([12101, 12101, 12101])
        assert coherent is True and spread == 0.0

    def test_runs_split_across_two_pstates_are_incoherent(self):
        coherent, spread = bench_core._clocks_are_coherent([12101, 11101])
        assert coherent is False
        assert spread == pytest.approx(0.0826, abs=1e-3)

    def test_tight_throughput_overrides_a_wide_clock_spread(self):
        """The real mistral Q5_K_M case: 3.5% clock spread, 0.5% throughput."""
        coherent, spread = bench_core._clocks_are_coherent(
            [11401, 11101, 11201, 11501, 11101],
            [49.18, 49.26, 49.40, 49.25, 49.44],
        )
        assert coherent is True
        assert spread == pytest.approx(0.035, abs=1e-3)

    def test_wide_clock_spread_with_loose_throughput_still_fails(self):
        """The override must not readmit the artifact it exists to exclude."""
        coherent, _ = bench_core._clocks_are_coherent(
            [11101, 9001], [49.2, 40.0])
        assert coherent is False

    def test_throughput_override_needs_throughput_data(self):
        assert bench_core._clocks_are_coherent([12101, 11101], None)[0] is False

    def test_small_jitter_is_tolerated(self):
        coherent, _ = bench_core._clocks_are_coherent([12101, 12000])
        assert coherent is True

    def test_a_single_run_is_trivially_coherent(self):
        assert bench_core._clocks_are_coherent([12101]) == (True, 0.0)


class TestClockSampleSummaries:
    def test_modal_pstate_picks_the_dominant_state(self):
        samples = [state(pstate="P0"), state(pstate="P0"), state(pstate="P2")]
        assert _modal_pstate(samples) == "P0"

    def test_modal_pstate_of_nothing_is_unknown(self):
        assert _modal_pstate([]) == "unknown"

    def test_throttles_are_deduplicated_and_sorted(self):
        samples = [state(throttle="0x4"), state(throttle="0x2"),
                   state(throttle="0x4")]
        assert _summarize_throttles(samples) == "0x2|0x4"

    def test_clean_samples_summarize_as_none(self):
        assert _summarize_throttles([state(), state()]) == "none"

    def test_unknown_is_not_reported_as_a_throttle(self):
        assert _summarize_throttles([state(throttle="unknown")]) == "none"


@pytest.mark.parametrize("pstate,mem,expected", [
    ("P0", 12001, True),
    ("P0", 11500, True),
    ("P2", 12001, False),
    ("P5", 810, False),
    ("P8", 405, False),
])
def test_gate_matrix(pstate, mem, expected):
    assert clocks_are_boosted(state(pstate=pstate, mem=mem)) is expected


class TestObservedMaxReference:
    """nvidia-smi's clocks.max.mem under-reports: 12001 spec, 12101 achieved.

    A threshold measured against the spec figure is measured from the wrong
    zero, so the observed peak has to win.
    """

    def test_observed_peak_overrides_reported_max(self):
        note_mem_clock(12101)
        assert reference_max_mem_mhz(state(mem_max=12001)) == 12101

    def test_reported_max_used_before_anything_observed(self):
        assert reference_max_mem_mhz(state(mem_max=12001)) == 12001

    def test_observed_peak_only_ratchets_upward(self):
        note_mem_clock(12101)
        note_mem_clock(9001)
        assert reference_max_mem_mhz(state(mem_max=12001)) == 12101

    def test_real_pstates_are_graded_against_observed_peak(self):
        """The three P-states this card actually visits under load."""
        note_mem_clock(12101)
        assert clocks_are_boosted(state(pstate="P0", mem=12101)) is True
        assert clocks_are_boosted(state(pstate="P2", mem=11101)) is False
        assert clocks_are_boosted(state(pstate="P4", mem=9001)) is False

    def test_unknown_everywhere_yields_unknown(self):
        assert reference_max_mem_mhz(state(mem_max=UNKNOWN)) == UNKNOWN


class TestClockSamplerAggregation:
    """The minimum under load governs admissibility, not the mean.

    A run that spent most of its time at P0 but dipped to P4 partway through is
    not a measurement at P0 — averaging would hide exactly the transition that
    produced the 1.23x artifact.
    """

    def _sampler_with(self, samples):
        sampler = ClockSampler()
        sampler.samples = samples
        return sampler

    def test_idle_samples_are_excluded_from_under_load(self):
        sampler = self._sampler_with([
            state(pstate="P5", mem=810), state(pstate="P0", mem=12101),
            state(pstate="P8", mem=405),
        ])
        assert [s["mem_mhz"] for s in sampler.under_load()] == [12101]

    def test_min_is_taken_over_loaded_samples_only(self):
        """A trailing idle sample must not be read as a mid-run dip."""
        sampler = self._sampler_with([
            state(pstate="P0", mem=12101), state(pstate="P0", mem=12101),
            state(pstate="P5", mem=810),
        ])
        assert sampler.min_mem_mhz() == 12101

    def test_a_genuine_mid_run_dip_is_reported(self):
        sampler = self._sampler_with([
            state(pstate="P0", mem=12101), state(pstate="P4", mem=9001),
            state(pstate="P0", mem=12101),
        ])
        assert sampler.min_mem_mhz() == 9001

    def test_no_samples_yields_unknown(self):
        assert self._sampler_with([]).min_mem_mhz() == UNKNOWN

    def test_sampler_thread_starts_and_stops_cleanly(self, monkeypatch):
        monkeypatch.setattr(bench_core, "read_clock_state",
                            lambda: state(pstate="P0", mem=12101))
        with ClockSampler(interval_s=0.01) as sampler:
            import time
            time.sleep(0.08)
        assert sampler.samples, "sampler collected nothing while running"
        assert sampler._thread is not None and not sampler._thread.is_alive()
