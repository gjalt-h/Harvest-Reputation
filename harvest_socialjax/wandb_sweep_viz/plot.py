from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, StrMethodFormatter

from .config import SweepDef
from .series import AggregatedSeries, EPISODE_XLABEL, metric_ylabel


# Orchard-grid palette (matches plot_thesis_env_layouts FOV apples / sand / mist).
# Soft, low-chroma greens and neutrals — not saturated Tailwind primaries.
SWEEP_COLORS = [
    "#7aab72",  # soft orchard mid-green
    "#a9dc99",  # FOV apple green (plus-patch layouts)
    "#5f8f6a",  # deeper meadow green
    "#8fb8c9",  # cool mist blue-grey
    "#c4b59a",  # warm sand / spawn-pad
    "#9aab8c",  # sage
    "#b8a67a",  # muted sand
    "#6d8f9c",  # slate-teal
]

# Fixed arm colours for thesis comparison plots (consistent across scripts).
ARM_COLORS = {
    "peer": "#5f8f6a",  # meadow green
    "gimm": "#c4b59a",  # warm sand
    "police": "#1f4e79",  # dark blue
    "enf": "#a67c52",  # muted terracotta
    "independent": "#7aab72",
    "shared": "#b8956a",
}

# Short legend names (same keys as ARM_COLORS).
ARM_LABELS = {
    "peer": "Peer",
    "peer_masked": "Peer mask",
    "gimm": "Good imm",
    "police": "Police",
    "police_masked": "Police mask",
    "enf": "Direct enf",
    "independent": "Individual",
    "shared": "Shared",
}

# Half-page A4 readability (axis / ticks / legend).
THESIS_FONT = {
    "label": 20,
    "tick": 16,
    "legend": 14,
}

# Overlay reference in comparison plots (solid=indep, dashed=shared).
BASELINE_COLOR = "#b0b0b0"
# Baseline-only figures (no treatment arms): clear meadow green, same solid/dashed.
BASELINE_STANDALONE_COLOR = "#5f8f6a"
OPT_MODE_COLORS = {
    "independent": ARM_COLORS["independent"],
    "shared": ARM_COLORS["shared"],
    "baseline": BASELINE_STANDALONE_COLOR,
}
# Tag timeout: cool mist blues (same family as layout cool_mist floor).
TAG_TIMEOUT_COLORS = [
    "#5a7a88",
    "#8fb8c9",
    "#6d8f9c",
    "#a8c5d0",
    "#4a6670",
]
# Beam geometry: warm sand / muted terracotta (spawn-pad / sand floors).
BEAM_GEOMETRY_COLORS = [
    "#b8956a",
    "#c4b59a",
    "#a67c52",
    "#d4bc8c",
    "#8f7355",
]


def _series_base_label(label: str) -> str:
    """Strip trailing ' (independent|shared)' mode suffix."""
    for mode in (" (independent)", " (shared)"):
        if label.endswith(mode):
            return label[: -len(mode)]
    return label.split(" (")[0]


def _legend_label(label: str) -> str:
    """Compact legend text: sweep param only (no mode / baseline / extras)."""
    base = _series_base_label(label)
    # Composite baseline arms: "tag=25 · apa=24.5 (baseline)" → "tag=25"
    for sep in (" · ", " ·", "· "):
        if sep in base:
            base = base.split(sep, 1)[0]
            break
    base = base.replace(" (baseline)", "").replace("(baseline)", "").strip()
    return base


def _is_baseline_label(base: str) -> bool:
    return "baseline" in base.lower()


def _is_tag_timeout_label(base: str) -> bool:
    text = base.lower()
    if _is_baseline_label(text):
        return False
    return text.startswith("tag") or "tag=" in text or "tag_timeout" in text


def _is_beam_geometry_label(base: str) -> bool:
    text = base.lower()
    return text.startswith("beam") or "beam" in text


def _parse_n_apa_from_label(base: str) -> tuple[int | None, int | None]:
    import re

    m_n = re.search(r"N=(\d+)", base)
    m_apa = re.search(r"apa=(\d+)", base)
    n = int(m_n.group(1)) if m_n else None
    apa = int(m_apa.group(1)) if m_apa else None
    return n, apa


def _parse_apa_from_label(base: str) -> int | None:
    _, apa = _parse_n_apa_from_label(base)
    return apa


def _is_plus_patch_label(base: str) -> bool:
    return "plus_patch" in base.lower()


def _assign_colors(series_list: list[AggregatedSeries], *, sweep_id: str) -> dict[str, str]:
    """Map base sweep labels → colors.

    For ``n4_tag_beam``: tag-timeout curves share blues, beam curves share
    oranges, and the reused N=4 default is black. Other sweeps keep the
    sequential palette.
    """
    bases: list[str] = []
    for series in series_list:
        base = _series_base_label(series.label)
        if base not in bases:
            bases.append(base)

    color_map: dict[str, str] = {}
    if sweep_id == "n_apa18_vs_24":
        n_colors: dict[int, str] = {}
        for base in bases:
            n, _ = _parse_n_apa_from_label(base)
            if n is None:
                color_map[base] = SWEEP_COLORS[len(color_map) % len(SWEEP_COLORS)]
                continue
            if n not in n_colors:
                n_colors[n] = SWEEP_COLORS[len(n_colors) % len(SWEEP_COLORS)]
            color_map[base] = n_colors[n]
        return color_map

    if sweep_id == "n_sweep_layout_compare":
        n_colors: dict[int, str] = {}
        for base in bases:
            n, _ = _parse_n_apa_from_label(base)
            if n is None:
                color_map[base] = SWEEP_COLORS[len(color_map) % len(SWEEP_COLORS)]
                continue
            if n not in n_colors:
                n_colors[n] = SWEEP_COLORS[len(n_colors) % len(SWEEP_COLORS)]
            color_map[base] = n_colors[n]
        return color_map

    if sweep_id == "n4_density_layout_compare":
        apa_colors: dict[int, str] = {}
        for base in bases:
            apa = _parse_apa_from_label(base)
            if apa is None:
                color_map[base] = SWEEP_COLORS[len(color_map) % len(SWEEP_COLORS)]
                continue
            if apa not in apa_colors:
                apa_colors[apa] = SWEEP_COLORS[len(apa_colors) % len(SWEEP_COLORS)]
            color_map[base] = apa_colors[apa]
        return color_map

    if sweep_id == "n4_tag_beam":
        tag_i = 0
        beam_i = 0
        for base in bases:
            if _is_baseline_label(base):
                color_map[base] = BASELINE_COLOR
            elif _is_tag_timeout_label(base):
                color_map[base] = TAG_TIMEOUT_COLORS[tag_i % len(TAG_TIMEOUT_COLORS)]
                tag_i += 1
            elif _is_beam_geometry_label(base):
                color_map[base] = BEAM_GEOMETRY_COLORS[beam_i % len(BEAM_GEOMETRY_COLORS)]
                beam_i += 1
            else:
                color_map[base] = SWEEP_COLORS[len(color_map) % len(SWEEP_COLORS)]
        return color_map

    if sweep_id in ("n4_tag_timeout", "n4_tag_timeout_plus98"):
        tag_i = 0
        for base in bases:
            if _is_baseline_label(base):
                color_map[base] = BASELINE_COLOR
            else:
                color_map[base] = TAG_TIMEOUT_COLORS[tag_i % len(TAG_TIMEOUT_COLORS)]
                tag_i += 1
        return color_map

    if sweep_id in ("n4_beam_geometry", "n4_beam_plus98"):
        beam_i = 0
        for base in bases:
            if _is_baseline_label(base):
                color_map[base] = BASELINE_COLOR
            else:
                color_map[base] = BEAM_GEOMETRY_COLORS[beam_i % len(BEAM_GEOMETRY_COLORS)]
                beam_i += 1
        return color_map

    for i, base in enumerate(bases):
        if _is_baseline_label(base):
            color_map[base] = BASELINE_COLOR
        else:
            color_map[base] = SWEEP_COLORS[i % len(SWEEP_COLORS)]
    return color_map


def _style_for_series(
    sweep_id: str,
    base_label: str,
    mode: str,
    *,
    is_baseline: bool = False,
) -> dict:
    if sweep_id == "n_apa18_vs_24":
        _, apa = _parse_n_apa_from_label(base_label)
        width = 2.4
        if apa == 24:
            return {"linestyle": ":", "linewidth": width, "alpha": 0.95}
        return {"linestyle": "-", "linewidth": width, "alpha": 1.0}
    if sweep_id in ("n_sweep_layout_compare", "n4_density_layout_compare"):
        width = 2.4
        if _is_plus_patch_label(base_label):
            return {"linestyle": ":", "linewidth": width, "alpha": 0.95}
        return {"linestyle": "-", "linewidth": width, "alpha": 1.0}
    return _style_for_mode(mode, is_baseline=is_baseline)


def _style_for_mode(mode: str, *, is_baseline: bool = False) -> dict:
    if is_baseline:
        # Keep baseline readable against colored families.
        if mode == "shared":
            return {"linestyle": "--", "linewidth": 2.4, "alpha": 0.95}
        return {"linestyle": "-", "linewidth": 2.8, "alpha": 1.0}
    if mode == "shared":
        return {"linestyle": "--", "linewidth": 2.0, "alpha": 0.95}
    return {"linestyle": "-", "linewidth": 2.4, "alpha": 1.0}


@dataclass
class SweepPlotBundle:
    sweep_id: str
    metric: str
    figure_path: str | None
    series: list[AggregatedSeries]
    title: str
    subtitle: str
    description: str
    sweep_label: str
    n_independent: int
    n_shared: int


def _is_whole_number(value: float) -> bool:
    return abs(value - round(value)) < 1e-9


def _format_episode_k(value: float, _pos: int) -> str:
    """Compact episode axis labels: 1000 -> 1k, 2500 -> 2.5k."""
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:.0f}M" if _is_whole_number(scaled) else f"{scaled:.1f}M"
    if abs_value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:.0f}k" if _is_whole_number(scaled) else f"{scaled:.1f}k"
    return f"{int(value)}" if _is_whole_number(value) else f"{value:.0f}"


def _format_with_commas(value: float, _pos: int) -> str:
    """Thousands separators for large y-axis values."""
    if abs(value) >= 1_000:
        return f"{value:,.0f}" if _is_whole_number(value) else f"{value:,.1f}"
    if _is_whole_number(value):
        return f"{int(value)}"
    return f"{value:.3g}"


def _apply_axis_formatters(ax) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(_format_episode_k))
    ymax = ax.get_ylim()[1]
    if ymax >= 1_000:
        ax.yaxis.set_major_formatter(FuncFormatter(_format_with_commas))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.3g}"))


def _style_for_optim_mode(mode: str) -> dict:
    width = 2.4
    if mode == "shared":
        return {"linestyle": "--", "linewidth": width, "alpha": 0.95}
    if mode == "baseline":
        return {"linestyle": "-.", "linewidth": width, "alpha": 1.0}
    return {"linestyle": "-", "linewidth": width, "alpha": 1.0}


def plot_baseline_eq_metric(
    sweep: SweepDef,
    metric: str,
    panels: dict[float, list[AggregatedSeries]],
    *,
    out_path,
    x_label: str = EPISODE_XLABEL,
    show_std: bool = True,
    figsize=(12.5, 4.8),
) -> SweepPlotBundle:
    eq_values = sorted(panels.keys())
    fig, axes = plt.subplots(1, len(eq_values), figsize=figsize, sharey=True)
    if len(eq_values) == 1:
        axes = [axes]

    mode_order = ("independent", "shared", "baseline")
    for ax, eq in zip(axes, eq_values):
        panel_series = sorted(
            panels[eq],
            key=lambda s: mode_order.index(s.mode) if s.mode in mode_order else 99,
        )
        for series in panel_series:
            color = OPT_MODE_COLORS.get(series.mode, "#444444")
            style = _style_for_optim_mode(series.mode)
            if show_std and np.any(series.std > 0):
                ax.fill_between(
                    series.x,
                    series.mean - series.std,
                    series.mean + series.std,
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
            ax.plot(series.x, series.mean, color=color, label=series.label, **style)
        eq_label = int(eq) if float(eq) == int(eq) else eq
        ax.set_title(f"EQUALITY_COEF={eq_label}", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=7, framealpha=0.9)
        _apply_axis_formatters(ax)

    axes[0].set_ylabel(metric_ylabel(metric))
    for ax in axes:
        ax.set_xlabel(x_label)

    title = f"{sweep.title} — {metric}"
    fig.suptitle(title, fontsize=13, y=1.02)
    footer = (
        "N=4 scaffold  |  Blue = independent  |  Red dashed = shared  |  "
        "Green dash-dot = baseline PPO  |  One panel per EQUALITY_COEF"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, color="#555555")
    if sweep.subtitle:
        fig.text(0.5, 0.96, sweep.subtitle, ha="center", va="top", fontsize=10, color="#333333")

    fig.tight_layout(rect=[0, 0.04, 1, 0.92 if sweep.subtitle else 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    flat = [s for group in panels.values() for s in group]
    return SweepPlotBundle(
        sweep_id=sweep.sweep_id,
        metric=metric,
        figure_path=str(out_path),
        series=flat,
        title=title,
        subtitle=sweep.subtitle,
        description=sweep.description,
        sweep_label=sweep.sweep_label,
        n_independent=sum(1 for s in flat if s.mode == "independent"),
        n_shared=sum(1 for s in flat if s.mode == "shared"),
    )


def plot_sweep_metric(
    sweep: SweepDef,
    metric: str,
    series_list: list[AggregatedSeries],
    *,
    out_path,
    x_label: str = EPISODE_XLABEL,
    show_std: bool = True,
    figsize=(9.5, 5.6),
    legend_outside: bool = False,
) -> SweepPlotBundle:
    indep = [s for s in series_list if s.mode == "independent"]
    shared = [s for s in series_list if s.mode == "shared"]
    ordered = indep + shared

    fig, ax = plt.subplots(figsize=figsize)
    color_map = _assign_colors(ordered, sweep_id=sweep.sweep_id)
    seen_legend: set[str] = set()
    # One sweep value (e.g. only N=4): legend names reward modes (line style encodes mode).
    unique_bases = {_legend_label(s.label) for s in ordered}
    mode_only_legend = len(unique_bases) <= 1 and any(
        s.mode == "shared" for s in ordered
    ) and any(s.mode == "independent" for s in ordered)
    _MODE_LEGEND = {
        "independent": ARM_LABELS["independent"],
        "shared": ARM_LABELS["shared"],
    }

    for series in ordered:
        base = _series_base_label(series.label)
        # Baseline-only: clear green. Grey is reserved for comparison overlays.
        color = (
            BASELINE_STANDALONE_COLOR
            if mode_only_legend
            else color_map.get(base, "#444444")
        )
        is_baseline = _is_baseline_label(base)
        style = _style_for_series(
            sweep.sweep_id,
            base,
            series.mode,
            is_baseline=is_baseline,
        )
        if mode_only_legend:
            legend = _MODE_LEGEND.get(series.mode, series.mode)
        elif is_baseline:
            legend = _MODE_LEGEND.get(series.mode, _legend_label(series.label))
        else:
            legend = _legend_label(series.label)
        # Avoid duplicate legend entries when labels collapse to the same param.
        plot_label = legend if legend not in seen_legend else "_nolegend_"
        if plot_label != "_nolegend_":
            seen_legend.add(legend)

        if show_std and np.any(series.std > 0):
            ax.fill_between(
                series.x,
                series.mean - series.std,
                series.mean + series.std,
                color=color,
                alpha=0.18 if series.mode == "independent" else 0.12,
                linewidth=0,
            )

        ax.plot(series.x, series.mean, color=color, label=plot_label, **style)

    ax.set_xlabel(x_label, fontsize=THESIS_FONT["label"])
    ax.set_ylabel(metric_ylabel(metric), fontsize=THESIS_FONT["label"])
    title = f"{sweep.title} — {metric_ylabel(metric)}"
    ax.tick_params(axis="both", labelsize=THESIS_FONT["tick"])
    ax.grid(True, alpha=0.25)
    _apply_axis_formatters(ax)

    legend_kwargs = dict(
        fontsize=THESIS_FONT["legend"],
        framealpha=0.9,
        handlelength=2.6,
        labelspacing=0.35,
    )
    if legend_outside:
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            **legend_kwargs,
        )
    else:
        ax.legend(
            loc="best",
            borderaxespad=0.4,
            **legend_kwargs,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return SweepPlotBundle(
        sweep_id=sweep.sweep_id,
        metric=metric,
        figure_path=str(out_path),
        series=series_list,
        title=title,
        subtitle=sweep.subtitle,
        description=sweep.description,
        sweep_label=sweep.sweep_label,
        n_independent=len(indep),
        n_shared=len(shared),
    )
