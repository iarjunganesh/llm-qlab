"""Chart theming — one selected palette per mode, shared by every figure.

Dark mode is *selected*, not derived. It is the same hue families stepped for a
dark surface, because inverting a light palette moves every colour out of the
lightness band it was validated in and quietly breaks the colourblind
separation the order exists to guarantee.

Both modes are validated against their own surface:

* categorical, adjacent pairs — CVD ΔE 9.2 light / 9.4 dark (≥8 target),
  normal-vision ΔE 27.6 light / 26.5 dark (≥15 floor)
* ordinal ramp — monotone lightness, adjacent ΔL ≥ 0.06, and the step nearest
  the surface clearing 2:1 (2.06:1 light, 2.15:1 dark)

The ordinal band differs per mode and that is the point: on light the ramp must
not start lighter than blue step 250, on dark it must not end darker than step
600. A mark that recedes into the surface is legal for a continuous scale where
the lightest value means "near zero", and not for a discrete one that has to
stay visible.
"""

from typing import Any

# Blue steps 250..650 — the band legal for an ordinal encoding on a light
# surface. Step 200 exists but measures 1.74:1 there, under the floor.
_ORDINAL_LIGHT = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281",
]

# Blue steps 250..600. Stops short of 650/700, which sink into the dark surface.
_ORDINAL_DARK = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95",
]

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        # Both modes carry an explicit suffix. An unsuffixed default would make
        # the light file look like "the chart" and the dark one like a variant,
        # when neither is primary — each is selected for its own surface.
        "suffix": "-light",
        "surface": "#fcfcfb",
        "ink": "#1a1a19",
        "muted": "#5c5c5a",
        "grid": "#e4e4e1",
        "axis": "#c9c9c5",
        "error": "#5c5c5a",
        # Categorical slots in the fixed order — the order is the CVD-safety
        # mechanism and must not be re-sorted per chart.
        "categorical": ["#2a78d6", "#eb6834", "#1baf7a",
                        "#eda100", "#e87ba4", "#4a3aa7"],
        "ordinal": _ORDINAL_LIGHT,
    },
    "dark": {
        "suffix": "-dark",
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "muted": "#c3c2b7",
        "grid": "#333331",
        "axis": "#4a4a47",
        "error": "#8a8a87",
        "categorical": ["#3987e5", "#d95926", "#199e70",
                        "#c98500", "#d55181", "#9085e9"],
        "ordinal": _ORDINAL_DARK,
    },
}

# Above this an ordinal ramp cannot keep adjacent steps 0.06 apart in lightness,
# so the formats stop being separable. A property of the ramp, not a threshold
# to tune: past it the answer is small multiples, not finer steps.
MAX_ORDINAL_STEPS = 5


def ordinal_ramp(theme: dict[str, Any], n: int) -> list[str]:
    """Evenly spaced steps across the mode's ordinal band, light to dark."""
    steps = theme["ordinal"]
    if n <= 1:
        return [steps[len(steps) // 2]]
    if n > MAX_ORDINAL_STEPS:
        print(f"[warn] {n} ordinal categories exceeds the {MAX_ORDINAL_STEPS} "
              "this ramp can keep visually distinct — adjacent categories will "
              "be hard to tell apart. Consider faceting.")
    last = len(steps) - 1
    return [steps[round(i * last / (n - 1))] for i in range(n)]


def categorical_map(theme: dict[str, Any], names: list[str]) -> dict[str, str]:
    """Map each name to a fixed categorical slot.

    Keyed on the sorted name rather than row order, so filtering the data cannot
    repaint the surviving series — colour follows the entity, never its rank.
    """
    slots = theme["categorical"]
    return {name: slots[i % len(slots)] for i, name in enumerate(sorted(names))}


def ink_legend(legend, theme: dict[str, Any]) -> None:
    """Recolour legend text to the mode's ink.

    Matplotlib defaults legend text to black regardless of figure facecolor, so
    on a dark surface it renders black-on-near-black and the series names become
    unreadable. Lives here rather than in one chart module because it was fixed
    in the comparison charts and missed in the ladder — exactly the divergence a
    shared theme is supposed to prevent.

    Text wears text tokens, never a series colour: the swatch beside it already
    carries identity.
    """
    if legend is None:
        return
    title = legend.get_title()
    if title is not None:
        title.set_color(theme["muted"])
    for text in legend.get_texts():
        text.set_color(theme["ink"])


def themed_path(path, theme: dict[str, Any]):
    """Insert the theme suffix before the extension: foo.png -> foo-dark.png."""
    return path.with_name(f"{path.stem}{theme['suffix']}{path.suffix}")
