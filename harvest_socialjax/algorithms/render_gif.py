"""Periodic GIF rendering of greedy policy rollouts during training.

Runs a full eval episode on the host (outside the jitted training loop),
captures frames via ``env.render(state)``, and saves animated GIFs for the
first ``RENDER_GIF_START_STEPS`` and last ``RENDER_GIF_TAIL_STEPS`` timesteps.
When ``RENDER_EPISODE_DIAG`` is on (default if ``ENABLE_REPUTATION``), also
writes a per-agent full-episode PNG/CSV (harvest, return, zap, reputation,
timeout). This is purely for visual debugging — it does NOT affect training.
"""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as onp

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None  # type: ignore[assignment,misc]
    ImageDraw = None  # type: ignore[assignment,misc]
    ImageFont = None  # type: ignore[assignment,misc]


def _num_updates(config: dict) -> int:
    """Resolve total update count (may not be written yet when GifRenderer inits)."""
    if config.get("NUM_UPDATES") is not None:
        return int(config["NUM_UPDATES"])
    total = int(config.get("TOTAL_TIMESTEPS", 0))
    steps = int(config.get("NUM_STEPS", 0))
    envs = int(config.get("NUM_ENVS", 0))
    if total > 0 and steps > 0 and envs > 0:
        return total // steps // envs
    return 0


def _steps_per_update(config: dict) -> int:
    return int(config.get("NUM_STEPS", 0)) * int(config.get("NUM_ENVS", 0))


def gif_render_updates(config: dict) -> set[int]:
    """Update indices at which to save a GIF (cheap: start / end / handful).

    Default: ``RENDER_GIF_COUNT`` GIFs evenly spaced over training (includes
    update 0 and the final update). Override with ``RENDER_GIF_EVERY`` for a
    fixed interval instead (e.g. every 200 updates).

    ``RENDER_GIF_AT_TIMESTEPS`` — list of env-step milestones (e.g. 1.5M, 3M,
    5M) converted to update indices. ``RENDER_GIF_AT_UPDATES`` — explicit
    update indices (0-based).

    Defaults keep cost low: a handful of short greedy rollouts, not one per
    checkpoint.
    """
    if not config.get("RENDER_GIF", True):
        return set()

    num_updates = _num_updates(config)
    if num_updates <= 0:
        return set()
    last = num_updates - 1

    at_steps = config.get("RENDER_GIF_AT_TIMESTEPS", None)
    if at_steps is not None and len(list(at_steps)) > 0:
        spu = _steps_per_update(config)
        if spu <= 0:
            return set()
        return {
            min(max(0, int(int(t) // spu)), last) for t in at_steps
        }

    at_updates = config.get("RENDER_GIF_AT_UPDATES", None)
    if at_updates is not None and len(list(at_updates)) > 0:
        return {min(max(0, int(u)), last) for u in at_updates}

    every_raw = config.get("RENDER_GIF_EVERY", None)
    if every_raw is not None and int(every_raw) > 0:
        every = int(every_raw)
        return {u for u in range(0, num_updates, every)} | {last}

    count = max(1, int(config.get("RENDER_GIF_COUNT", 5)))
    if count == 1:
        return {last}
    if last == 0 or count == 2:
        return {0, last}
    # Evenly spaced inclusive endpoints; round then uniquify.
    xs = onp.linspace(0, last, num=min(count, last + 1))
    return {int(round(x)) for x in xs}


def should_render_gif(update: int, config: dict) -> bool:
    """True when a GIF should be saved at this update."""
    return int(update) in gif_render_updates(config)


def gif_update_for_chunk(
    chunk_start: int, last_update: int, config: dict
) -> int | None:
    """If any scheduled GIF falls in ``[chunk_start, last_update]``, return one
    update index to label the GIF (the latest hit). Avoids missing mid-chunk
    schedule points when the host only syncs at chunk ends.
    """
    hits = [
        u
        for u in gif_render_updates(config)
        if chunk_start <= u <= last_update
    ]
    return max(hits) if hits else None


def _ensure_pil() -> bool:
    if Image is None:
        print("[render_gif] WARNING: Pillow not installed - skipping GIF render")
        return False
    return True


def _obs_to_batch(obs, agents) -> jnp.ndarray:
    """Stack per-agent spatial obs to ``(N, *obs_shape)``.

    Harvest envs return a stacked array; ReputationWrapper returns
    nested ``{"spatial", "reputation"}`` per agent; some SocialJax wrappers
    return a dict keyed by agent id (str or int).
    """
    if isinstance(obs, dict):
        rows = []
        for a in agents:
            if a in obs:
                entry = obs[a]
            elif str(a) in obs:
                entry = obs[str(a)]
            else:
                raise KeyError(f"agent {a!r} missing from obs keys {list(obs)}")
            if isinstance(entry, dict) and "spatial" in entry:
                rows.append(entry["spatial"])
            else:
                rows.append(entry)
        return jnp.stack(rows, axis=0)
    return jnp.asarray(obs)


def _rep_from_obs(obs, agents, num_agents: int) -> jnp.ndarray | None:
    """Extract absolute ``(N,)`` reputation from nested obs, or None."""
    if not isinstance(obs, dict):
        return None
    sample = obs[agents[0]] if agents[0] in obs else obs.get(str(agents[0]))
    if isinstance(sample, dict) and "reputation" in sample:
        return jnp.asarray(sample["reputation"], dtype=jnp.float32)
    return None


def _unwrap_render_state(state):
    """Drill through LogEnvState / ReputationEnvState to Harvest ``State``."""
    cur = state
    for _ in range(4):
        if hasattr(cur, "agent_locs") and hasattr(cur, "grid"):
            return cur
        if hasattr(cur, "env_state"):
            cur = cur.env_state
        else:
            break
    return state


def _rep_from_state(state) -> jnp.ndarray | None:
    cur = state
    for _ in range(4):
        if hasattr(cur, "rep_state"):
            return jnp.asarray(cur.rep_state.rep)
        # Reputation v2: Good/Bad labels live on BurstWindowState.
        if hasattr(cur, "burst_state") and hasattr(cur.burst_state, "label"):
            return jnp.asarray(cur.burst_state.label)
        if hasattr(cur, "env_state"):
            cur = cur.env_state
        else:
            break
    # Fallback: Harvest State.tag_filter_rep (set by ReputationV2Wrapper).
    cur = state
    for _ in range(4):
        if hasattr(cur, "tag_filter_rep"):
            return jnp.asarray(cur.tag_filter_rep)
        if hasattr(cur, "env_state"):
            cur = cur.env_state
        else:
            break
    return None


# Binary Good / Bad colours for clean GIFs (match env GREEN/RED accents).
_GOOD_AGENT_RGB = (44, 160, 44)
_BAD_AGENT_RGB = (214, 39, 40)
_NEUTRAL_AGENT_RGB = (160, 160, 160)


def canonicalize_render_frame(frame: onp.ndarray) -> onp.ndarray:
    """Undo ``Harvest_open.render``'s ``rot90(..., 2)`` so spawn is BR.

    The env renderer applies a 180° flip after cropping; agent grid coords and
    spawn layout are defined with row 0 at the top and spawn in the bottom-right.
  """
    return onp.rot90(onp.asarray(frame), 2)


def _agent_locs_to_canonical_pixels(
    agent_locs: onp.ndarray,
    *,
    tile_size: int = 32,
) -> list[tuple[int, int]]:
    """Map unpadded grid ``(row, col)`` to pixel ``(x, y)`` on canonical frames.

    Matches ``Harvest_open.render`` crop offsets: padded tile indices
    ``(r + PADDING, c + PADDING)`` minus ``(PADDING - 1)`` per axis, i.e.
    ``(r + 1) * tile_size`` — then the same 180° undo as
    ``canonicalize_render_frame``.
    """
    pts: list[tuple[int, int]] = []
    for loc in onp.asarray(agent_locs):
        r, c = int(loc[0]), int(loc[1])
        x = (c + 1) * tile_size + tile_size // 2
        y = (r + 1) * tile_size + tile_size // 2
        pts.append((x, y))
    return pts


def _overlay_reputation(
    frame: onp.ndarray,
    agent_locs: onp.ndarray,
    rep: onp.ndarray,
    *,
    tagged: onp.ndarray | None = None,
    tile_size: int = 32,
    padding: int = 0,
) -> onp.ndarray:
    """Draw ``i:±0.00`` near each agent; green if rep>0, red if <0.

    Timed-out agents get a gray ``T`` suffix (they are off the board; label
    is still drawn at their last stored location).

    ``frame`` must be canonical (spawn bottom-right); see
    ``canonicalize_render_frame``.
    """
    del padding  # kept for call-site compatibility
    if Image is None or ImageDraw is None:
        return frame
    img = Image.fromarray(onp.asarray(frame, dtype=onp.uint8))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    tagged_arr = None if tagged is None else onp.asarray(tagged).astype(bool)
    for i, (x, y) in enumerate(
        _agent_locs_to_canonical_pixels(agent_locs, tile_size=tile_size)
    ):
        val = float(rep[i])
        frozen = bool(tagged_arr[i]) if tagged_arr is not None and i < tagged_arr.size else False
        if frozen:
            color = (160, 160, 160)
            label = f"{i}:{val:+.2f} T"
        else:
            color = (
                (40, 180, 40)
                if val > 0.02
                else ((200, 40, 40) if val < -0.02 else (220, 220, 220))
            )
            label = f"{i}:{val:+.2f}"
        draw.text((x + 2, max(0, y - 14)), label, fill=color, font=font)
    return onp.asarray(img)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _build_rep_side_panel(
    rep: onp.ndarray,
    *,
    tagged: onp.ndarray | None,
    num_agents: int,
    height: int,
    width: int,
) -> onp.ndarray:
    """Per-agent reputation bars for one GIF timestep (RGB, ``height`` x ``width``)."""
    if Image is None or ImageDraw is None:
        return onp.zeros((height, width, 3), dtype=onp.uint8)

    rep = onp.asarray(rep, dtype=onp.float32).reshape(-1)
    tagged_arr = (
        onp.asarray(tagged).astype(bool).reshape(-1)
        if tagged is not None
        else onp.zeros((num_agents,), dtype=bool)
    )
    n = int(num_agents)
    panel = Image.new("RGB", (width, height), (28, 28, 32))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    margin_x = 8
    margin_y = 6
    title_h = 16
    draw.text((margin_x, 4), "reputation", fill=(220, 220, 220), font=font)
    row_h = max(18, (height - margin_y * 2 - title_h) // max(n, 1))
    bar_left = margin_x + 52
    bar_right = width - margin_x - 4
    bar_w = max(20, bar_right - bar_left)
    mid_y = margin_y + title_h

    for i in range(n):
        y0 = mid_y + i * row_h
        color = _hex_to_rgb(_AGENT_COLORS[i % len(_AGENT_COLORS)])
        val = float(rep[i]) if i < rep.size else 0.0
        frozen = bool(tagged_arr[i]) if i < tagged_arr.size else False
        draw.text((margin_x, y0 + 2), f"a{i}", fill=color, font=font)
        label = f"{val:+.2f}" + (" T" if frozen else "")
        draw.text((margin_x + 18, y0 + 2), label, fill=(210, 210, 210), font=font)
        track_y1 = y0 + row_h - 8
        track_y0 = track_y1 - 6
        draw.rectangle(
            (bar_left, track_y0, bar_left + bar_w, track_y1),
            fill=(55, 55, 60),
        )
        center = bar_left + bar_w // 2
        fill_w = int(abs(val) * (bar_w // 2 - 2))
        if frozen:
            bar_color = (120, 120, 120)
        elif val > 0.02:
            bar_color = (50, 170, 70)
        elif val < -0.02:
            bar_color = (200, 60, 60)
        else:
            bar_color = (150, 150, 150)
        if fill_w > 0:
            if val >= 0:
                draw.rectangle(
                    (center, track_y0, center + fill_w, track_y1),
                    fill=bar_color,
                )
            else:
                draw.rectangle(
                    (center - fill_w, track_y0, center, track_y1),
                    fill=bar_color,
                )
        draw.line((center, track_y0, center, track_y1), fill=(90, 90, 95), width=1)
    return onp.asarray(panel, dtype=onp.uint8)


def compose_frame_with_rep_panel(
    frame: onp.ndarray,
    rep: onp.ndarray,
    *,
    tagged: onp.ndarray | None = None,
    num_agents: int,
    panel_width: int | None = None,
) -> onp.ndarray:
    """Concatenate env RGB (left) with per-agent reputation panel (right)."""
    frame = onp.asarray(frame, dtype=onp.uint8)
    pw = panel_width if panel_width is not None else max(150, frame.shape[1] // 4)
    panel = _build_rep_side_panel(
        rep,
        tagged=tagged,
        num_agents=num_agents,
        height=frame.shape[0],
        width=pw,
    )
    return onp.concatenate([frame, panel], axis=1)


def compose_frame_with_good_bad_legend(
    frame: onp.ndarray,
    *,
    panel_width: int = 110,
) -> onp.ndarray:
    """Append a compact Good/Bad colour legend to the right of the grid."""
    frame = onp.asarray(frame, dtype=onp.uint8)
    h, _w, _ = frame.shape
    if Image is None or ImageDraw is None:
        return frame
    panel = Image.new("RGB", (panel_width, h), (24, 24, 28))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    entries = (
        ("Good", _GOOD_AGENT_RGB),
        ("Bad", _BAD_AGENT_RGB),
    )
    y0 = max(12, h // 2 - 28)
    for i, (label, color) in enumerate(entries):
        y = y0 + i * 36
        draw.rectangle((12, y, 28, y + 16), fill=color)
        draw.text((36, y + 2), label, fill=(230, 230, 230), font=font)
    return onp.concatenate([frame, onp.asarray(panel, dtype=onp.uint8)], axis=1)


def _rep_to_agent_colors(rep: onp.ndarray, num_agents: int) -> list[tuple[int, int, int]]:
    """Map reputation labels to Good/Bad/neutral RGB triples."""
    rep = onp.asarray(rep, dtype=onp.float32).reshape(-1)
    colors: list[tuple[int, int, int]] = []
    for i in range(num_agents):
        val = float(rep[i]) if i < rep.size else 0.0
        if val > 0.0:
            colors.append(_GOOD_AGENT_RGB)
        elif val < 0.0:
            colors.append(_BAD_AGENT_RGB)
        else:
            colors.append(_NEUTRAL_AGENT_RGB)
    return colors


def _facing_triangle_pixels(
    cx: int, cy: int, direction: int, *, size: int
) -> list[tuple[int, int]]:
    """Agent triangle tip points facing ``direction`` (0=up,1=right,2=down,3=left)."""
    d = int(direction) % 4
    # Tip / base-left / base-right relative to facing (matches Harvest_open.render_tile).
    if d == 0:  # up
        return [(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)]
    if d == 1:  # right
        return [(cx + size, cy), (cx - size, cy - size), (cx - size, cy + size)]
    if d == 2:  # down
        return [(cx, cy + size), (cx + size, cy - size), (cx - size, cy - size)]
    return [(cx - size, cy), (cx + size, cy - size), (cx + size, cy + size)]


def paint_agents_good_bad(
    frame: onp.ndarray,
    agent_locs: onp.ndarray,
    rep: onp.ndarray,
    *,
    tagged: onp.ndarray | None = None,
    tile_size: int = 32,
) -> onp.ndarray:
    """Overwrite agent sprites with Good=green / Bad=red facing triangles.

    Applied on canonical frames (spawn bottom-right). More reliable than mutating
    ``PLAYER_COLOURS`` (tile-cache / JAX membership edge cases).
    """
    if Image is None or ImageDraw is None:
        return frame
    frame = onp.asarray(frame, dtype=onp.uint8)
    locs = onp.asarray(agent_locs)
    if locs.ndim != 2 or locs.shape[0] == 0:
        return frame
    n = int(locs.shape[0])
    colors = _rep_to_agent_colors(rep, n)
    tagged_arr = (
        onp.asarray(tagged).astype(bool).reshape(-1)
        if tagged is not None
        else onp.zeros((n,), dtype=bool)
    )
    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    sand = (210, 190, 140)
    half = max(6, tile_size // 2 - 2)
    tri = max(5, tile_size // 2 - 6)
    for i in range(n):
        r, c = int(locs[i, 0]), int(locs[i, 1])
        direction = int(locs[i, 2]) if locs.shape[1] > 2 else 1
        cx = (c + 1) * tile_size + tile_size // 2
        cy = (r + 1) * tile_size + tile_size // 2
        if cx < 0 or cy < 0 or cx >= img.width or cy >= img.height:
            continue
        # Cover the old multi-colour sprite, then paint legend colour.
        draw.ellipse(
            (cx - half, cy - half, cx + half, cy + half),
            fill=sand,
        )
        frozen = bool(tagged_arr[i]) if i < tagged_arr.size else False
        color = _NEUTRAL_AGENT_RGB if frozen else colors[i]
        pts = _facing_triangle_pixels(cx, cy, direction, size=tri)
        draw.polygon(pts, fill=color)
    return onp.asarray(img, dtype=onp.uint8)


# Orchard palette (same family as wandb_sweep_viz.plot.SWEEP_COLORS).
_AGENT_COLORS = (
    "#7aab72",  # soft orchard mid-green
    "#8fb8c9",  # cool mist blue-grey
    "#c4b59a",  # warm sand / spawn-pad
    "#5f8f6a",  # deeper meadow green
    "#6d8f9c",  # slate-teal
    "#a9dc99",  # FOV apple green
    "#b8a67a",  # muted sand
    "#9aab8c",  # sage
)


def _to_agent_vec(value, num_agents: int, agents) -> onp.ndarray:
    """Coerce env reward/info to ``(N,)`` float32."""
    if value is None:
        return onp.zeros((num_agents,), dtype=onp.float32)
    if isinstance(value, dict):
        out = onp.zeros((num_agents,), dtype=onp.float32)
        for i, agent in enumerate(agents):
            if agent in value:
                raw = value[agent]
            elif str(agent) in value:
                raw = value[str(agent)]
            else:
                continue
            out[i] = float(onp.asarray(raw).reshape(-1)[0])
        return out
    arr = onp.asarray(value).reshape(-1)
    out = onp.zeros((num_agents,), dtype=onp.float32)
    n = min(int(arr.size), num_agents)
    out[:n] = arr[:n].astype(onp.float32)
    return out


def _timeout_from_state(state) -> onp.ndarray | None:
    inner = _unwrap_render_state(state)
    if hasattr(inner, "time_out"):
        return onp.asarray(inner.time_out) > 0
    return None


def _rate_ema_from_state(state) -> onp.ndarray | None:
    cur = state
    for _ in range(4):
        if hasattr(cur, "rep_state") and hasattr(cur.rep_state, "rate_ema"):
            return onp.asarray(cur.rep_state.rate_ema, dtype=onp.float32)
        if hasattr(cur, "env_state"):
            cur = cur.env_state
        else:
            break
    return None


def save_episode_diag(
    trace: dict[str, onp.ndarray],
    path: Path | str,
    *,
    r_star: float | None = None,
    title: str = "",
) -> Path:
    """Per-agent episode time series: cum. return, harvest rate, reputation.

    ``trace`` values are ``(T, N)``. Also writes a sibling ``.csv``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    harvest = onp.asarray(trace["harvest"], dtype=onp.float32)
    reward = onp.asarray(trace["reward"], dtype=onp.float32)
    zap_ok = onp.asarray(trace["zap_successful"], dtype=onp.float32)
    rep = onp.asarray(trace["rep"], dtype=onp.float32)
    tagged = onp.asarray(trace["tagged"], dtype=onp.float32)
    rate_ema = onp.asarray(
        trace.get("rate_ema", onp.full_like(harvest, onp.nan)), dtype=onp.float32
    )
    t_len, n_agents = harvest.shape
    steps = onp.arange(1, t_len + 1)
    cum_harvest = onp.cumsum(harvest, axis=0)
    cum_return = onp.cumsum(reward, axis=0)

    # Slightly taller than wide — readable when scaled onto A4.
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    label_fs = 14
    tick_fs = 12
    legend_fs = 11
    lw_main = 2.0

    series = (
        (axes[0], cum_return, lw_main),
        (axes[1], rate_ema, lw_main),
        (axes[2], rep, lw_main),
    )
    for i in range(n_agents):
        color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
        label = f"agent {i}"
        timed_out = tagged[:, i] > 0.5
        for ax, ys, lw in series:
            # Solid = active; dashed = timed out (same color).
            y_active = ys[:, i].astype(onp.float32).copy()
            y_out = ys[:, i].astype(onp.float32).copy()
            y_active[timed_out] = onp.nan
            y_out[~timed_out] = onp.nan
            ax.plot(steps, y_active, color=color, lw=lw, ls="-", label=label)
            ax.plot(steps, y_out, color=color, lw=lw, ls="--", alpha=0.9)

    if r_star is not None:
        axes[1].axhline(float(r_star), color="#6b7280", lw=1.2, ls=":")

    axes[2].axhline(0.0, color="#9ca3af", lw=0.8)
    axes[2].set_ylim(-1.08, 1.08)
    axes[2].set_xlabel("episode step", fontsize=label_fs)
    for ax in axes:
        ax.tick_params(labelsize=tick_fs)
        ax.set_ylabel("")  # drawn as fig.text below for shared alignment
    axes[0].legend(
        loc="upper left",
        ncol=1,
        fontsize=legend_fs,
        frameon=False,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.14)
    # Same figure-x for all y-labels (avoids shift from negative tick width).
    ylabels = ("cum. return", "harvest rate", "reputation")
    for ax, lab in zip(axes, ylabels):
        pos = ax.get_position()
        fig.text(
            0.035,
            pos.y0 + 0.5 * pos.height,
            lab,
            rotation=90,
            va="center",
            ha="center",
            fontsize=label_fs,
        )
    fig.savefig(path, dpi=200)
    plt.close(fig)

    csv_path = path.with_suffix(".csv")
    header = [
        "t",
        "agent",
        "harvest",
        "cum_harvest",
        "reward",
        "cum_return",
        "zap_successful",
        "rep",
        "tagged",
        "rate_ema",
        "burst_count",
        "burst_penalty",
    ]
    raw_rep = onp.asarray(trace.get("raw_rep", onp.full_like(rep, onp.nan)))
    zap_try = onp.asarray(trace.get("zap_attempted", onp.full_like(zap_ok, onp.nan)))
    burst_count = onp.asarray(
        trace.get("burst_count", onp.full_like(harvest, onp.nan)), dtype=onp.float32
    )
    burst_penalty = onp.asarray(
        trace.get("burst_penalty", onp.full_like(harvest, onp.nan)), dtype=onp.float32
    )
    header.extend(["raw_rep", "zap_attempted"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for t_i in range(t_len):
            for a_i in range(n_agents):
                row = [
                    str(t_i + 1),
                    str(a_i),
                    f"{float(harvest[t_i, a_i]):.6g}",
                    f"{float(cum_harvest[t_i, a_i]):.6g}",
                    f"{float(reward[t_i, a_i]):.6g}",
                    f"{float(cum_return[t_i, a_i]):.6g}",
                    f"{float(zap_ok[t_i, a_i]):.6g}",
                    f"{float(rep[t_i, a_i]):.6g}",
                    f"{float(tagged[t_i, a_i]):.6g}",
                    f"{float(rate_ema[t_i, a_i]):.6g}",
                    f"{float(burst_count[t_i, a_i]):.6g}",
                    f"{float(burst_penalty[t_i, a_i]):.6g}",
                    f"{float(raw_rep[t_i, a_i]):.6g}",
                    f"{float(zap_try[t_i, a_i]):.6g}",
                ]
                fh.write(",".join(row) + "\n")
    return path


def save_occupancy_heatmap(
    occupancy: onp.ndarray,
    wall: onp.ndarray,
    path: Path | str,
    *,
    occupancy_police: onp.ndarray | None = None,
    occupancy_agents: onp.ndarray | None = None,
    ep_len: int = 0,
    mode: str = "police",
    match_gif_orientation: bool = True,
    norm_mode: str = "pct",
    vmax_percentile: float = 95.0,
    subtitle: str | None = None,
    write_npz: bool = True,
) -> Path:
    """Thesis-style visit heatmap over a full greedy episode (grid coords).

    `occupancy` / `wall` are `(H, W)` in env grid coordinates (row 0 at
    top of the array). With `match_gif_orientation=True` (default), arrays are
    rotated 180° so spawn is bottom-right — same view as clean Good/Bad GIFs.

    `mode`:
      - `police` — police agents only (default; requires `occupancy_police`)
      - `all` — all agents (sum)
      - `both` — side-by-side police + all
      - `agents` — one panel per agent (`occupancy_agents` shape ``(A, H, W)``)

    `norm_mode`:
      - `pct` — linear, vmax at `vmax_percentile` of positive cells
        (camping saturates; path stays visible). Default.
      - `linear` — linear to true max (camping washes out the path)
      - `log` — LogNorm (sparse 1-visit cells look loud)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    occ = onp.asarray(occupancy, dtype=onp.float32)
    walls = onp.asarray(wall, dtype=bool)
    occ_pol = (
        onp.asarray(occupancy_police, dtype=onp.float32)
        if occupancy_police is not None
        else None
    )
    occ_ag = (
        onp.asarray(occupancy_agents, dtype=onp.float32)
        if occupancy_agents is not None
        else None
    )
    # Persist raw (unrotated) arrays; rotate only for display.
    occ_raw, walls_raw = occ.copy(), walls.copy()
    occ_pol_raw = None if occ_pol is None else occ_pol.copy()
    occ_ag_raw = None if occ_ag is None else occ_ag.copy()
    if match_gif_orientation:
        occ = onp.rot90(occ, 2)
        walls = onp.rot90(walls, 2)
        if occ_pol is not None:
            occ_pol = onp.rot90(occ_pol, 2)
        if occ_ag is not None:
            occ_ag = onp.rot90(occ_ag, 2, axes=(1, 2))

    mode = str(mode or "police").lower()
    if mode == "police" and occ_pol is None:
        raise ValueError(
            "occupancy_police is required for mode='police' (police ids were empty)"
        )
    if mode == "agents" and occ_ag is None:
        raise ValueError("occupancy_agents is required for mode='agents'")
    if mode == "both" and occ_pol is None:
        mode = "all"
    norm_mode = str(norm_mode or "pct").lower()

    # Orchard floor + warm visit ramp (matches thesis layout / sweep greens).
    floor = onp.array([0.933, 0.933, 0.914], dtype=onp.float32)
    wall_rgb = onp.array([0.55, 0.55, 0.57], dtype=onp.float32)
    border = "#8c8c91"
    cmap = LinearSegmentedColormap.from_list(
        "orchard_heat",
        ["#eeeee9", "#cfe8c4", "#a9dc99", "#7aab72", "#5f8f6a", "#c4b59a"],
    )
    cmap_sat = cmap.copy()
    cmap_sat.set_over("#a67c52")  # camp beyond percentile cap

    agent_spine_colors = ["#5f8f6a", "#1f4e79", "#a67c52", "#6d8f9c"]
    if mode == "agents":
        n_a = int(occ_ag.shape[0])
        panels = [(f"Agent {i}", occ_ag[i]) for i in range(n_a)]
        ncols = 2
        nrows = max(1, int(onp.ceil(n_a / ncols)))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.6 * ncols, 4.6 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()
    elif mode == "both":
        panels = [
            ("police", occ_pol),
            ("all agents", occ),
        ]
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5.6 * len(panels), 4.6),
            squeeze=False,
        )
        axes_flat = axes[0]
    elif mode == "all":
        panels = [("all agents", occ)]
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5.6 * len(panels), 4.6),
            squeeze=False,
        )
        axes_flat = axes[0]
    else:
        panels = [("police", occ_pol)]
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5.6 * len(panels), 4.6),
            squeeze=False,
        )
        axes_flat = axes[0]

    # Shared percentile cap for multi-agent panels (fair comparison).
    shared_vmax = None
    if mode == "agents" and occ_ag is not None:
        pos_all = occ_ag[occ_ag > 0]
        if pos_all.size:
            shared_vmax = max(
                float(onp.percentile(pos_all, float(vmax_percentile))), 1.0
            )

    foot_bits: list[str] = []
    if subtitle:
        foot_bits.append(str(subtitle))
    elif ep_len > 0:
        foot_bits.append(f"full episode · T={ep_len}")

    last_im = None
    for panel_i, (title, grid) in enumerate(panels):
        ax = axes_flat[panel_i]
        vis = onp.asarray(grid, dtype=onp.float32).copy()
        pos = vis[vis > 0]
        vmax_true = float(onp.max(vis)) if pos.size else 1.0
        if norm_mode == "log":
            norm = LogNorm(vmin=1.0, vmax=max(vmax_true, 1.0))
            plot_vis = onp.where(vis > 0, vis, onp.nan)
            use_cmap = cmap
            extend = "neither"
            cbar_label = "visits"
        elif norm_mode == "linear":
            norm = Normalize(vmin=0.0, vmax=max(vmax_true, 1.0))
            plot_vis = onp.where(vis > 0, vis, onp.nan)
            use_cmap = cmap
            extend = "neither"
            cbar_label = "visits"
        else:
            # Percentile cap: keep patrol readable; camping hits "over" color.
            pct = float(vmax_percentile)
            if mode == "agents" and shared_vmax is not None:
                vmax = shared_vmax
            elif pos.size:
                vmax = float(onp.percentile(pos, pct))
            else:
                vmax = 1.0
            vmax = max(vmax, 1.0)
            if panel_i == 0 and vmax_true > vmax * 1.01:
                foot_bits.append(
                    f"camp max={vmax_true:.0f} (cbar capped @ p{pct:g}={vmax:.1f})"
                )
            norm = Normalize(vmin=0.0, vmax=vmax)
            plot_vis = onp.where(vis > 0, vis, onp.nan)
            use_cmap = cmap_sat
            extend = "max"
            cbar_label = f"visits (≤p{pct:g})"
        ax.imshow(
            onp.zeros((*vis.shape, 3)) + floor,
            interpolation="nearest",
            origin="upper",
        )
        im = ax.imshow(
            plot_vis,
            cmap=use_cmap,
            norm=norm,
            interpolation="nearest",
            origin="upper",
        )
        last_im = im
        if onp.any(walls):
            wall_overlay = onp.zeros((*vis.shape, 4))
            wall_overlay[walls, :3] = wall_rgb
            wall_overlay[walls, 3] = 1.0
            ax.imshow(wall_overlay, interpolation="nearest", origin="upper")
        spine_color = (
            agent_spine_colors[panel_i % len(agent_spine_colors)]
            if mode == "agents"
            else border
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(spine_color)
            spine.set_linewidth(2.4 if mode == "agents" else 2.0)
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    if mode == "agents":
        for j in range(len(panels), len(axes_flat)):
            axes_flat[j].axis("off")
        if last_im is not None:
            cbar = fig.colorbar(
                last_im,
                ax=axes.ravel().tolist(),
                fraction=0.03,
                pad=0.02,
                extend="max",
            )
            cbar.set_label(cbar_label, fontsize=12)
            cbar.ax.tick_params(labelsize=10)
    else:
        for panel_i, (title, grid) in enumerate(panels):
            ax = axes_flat[panel_i]
            cbar = fig.colorbar(
                ax.images[-1],
                ax=ax,
                fraction=0.046,
                pad=0.04,
                extend=extend if norm_mode == "pct" else "neither",
            )
            cbar.set_label(
                cbar_label if norm_mode == "pct" else "visits",
                fontsize=12,
            )
            cbar.ax.tick_params(labelsize=10)

    if foot_bits:
        fig.text(
            0.5,
            0.01,
            " · ".join(foot_bits),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#555555",
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if write_npz:
        payload = {
            "occupancy": occ_raw,
            "wall": walls_raw.astype(onp.uint8),
            "ep_len": int(ep_len),
        }
        if occ_pol_raw is not None:
            payload["occupancy_police"] = occ_pol_raw
        if occ_ag_raw is not None:
            payload["occupancy_agents"] = occ_ag_raw
        onp.savez_compressed(path.with_suffix(".npz"), **payload)
    return path



def _actions_for_env(actions: jnp.ndarray, agents, obs_was_dict: bool):
    """Match action format to env: dict if obs was dict, else stacked array."""
    if obs_was_dict:
        return {str(a): actions[i] for i, a in enumerate(agents)}
    return actions.astype(jnp.int32)


def _episode_done(done) -> bool:
    if isinstance(done, dict):
        if "__all__" in done:
            return bool(jnp.all(done["__all__"]))
        return bool(jnp.all(jnp.asarray(list(done.values()))))
    return bool(jnp.all(done))


def _safe_token(value, *, default: str = "na") -> str:
    """Filesystem-safe short token."""
    if value is None:
        return default
    s = str(value).strip().replace(" ", "")
    for ch in ("/", "\\", ":", "*", "?", "\"", "<", ">", "|", ".", ","):
        s = s.replace(ch, "-")
    return s or default


def _fmt_lr(lr) -> str:
    try:
        return f"{float(lr):.0e}".replace("+", "")
    except (TypeError, ValueError):
        return "na"


def build_run_slug(config: dict, *, algo: str, seed_i: int = 0) -> str:
    """Compact filesystem slug of the most identifying run params."""
    env_kw = config.get("ENV_KWARGS") or {}
    if not isinstance(env_kw, dict):
        env_kw = {}

    parts = [
        _safe_token(algo),
        _safe_token(config.get("ENV_SCALE", "na")),
        f"N{_safe_token(config.get('NUM_ENVS', 'na'))}",
        f"seed{int(seed_i)}",
    ]

    # Algo-specific knobs that often distinguish sweeps.
    if "ippo" in str(algo).lower():
        parts.append(f"lr{_fmt_lr(config.get('LR'))}")
        if config.get("ENT_COEF") is not None:
            parts.append(f"ent{_safe_token(config.get('ENT_COEF'))}")
    else:
        parts.append(f"lr{_fmt_lr(config.get('LR'))}")
        if config.get("EPS_FINISH") is not None:
            parts.append(f"eps{_safe_token(config.get('EPS_FINISH'))}")

    cfg_name = config.get("_CFG_NAME") or config.get("CFG_NAME")
    if cfg_name:
        parts.insert(1, _safe_token(cfg_name))

    return "_".join(parts)


def build_title_lines(
    config: dict,
    *,
    algo: str,
    update: int,
    seed_i: int = 0,
    n_frames: int | None = None,
    window: str | None = None,
) -> list[str]:
    """Human-readable title lines drawn onto every GIF frame."""
    env_kw = config.get("ENV_KWARGS") or {}
    if not isinstance(env_kw, dict):
        env_kw = {}

    scale = config.get("ENV_SCALE", "?")
    n_agents = env_kw.get("num_agents", config.get("num_agents", "?"))
    T = env_kw.get("num_inner_steps", "?")
    shared = env_kw.get("shared_rewards", "?")
    cfg_name = config.get("_CFG_NAME") or config.get("CFG_NAME") or "?"

    line1 = (
        f"{algo} | cfg={cfg_name} | scale={scale} | "
        f"update={update} | seed={seed_i}"
    )
    if window:
        line1 += f" | {window}"
    line2 = (
        f"N_envs={config.get('NUM_ENVS', '?')} steps={config.get('NUM_STEPS', '?')} "
        f"agents={n_agents} T={T} shared_rew={shared} "
        f"LR={config.get('LR', '?')} SEED={config.get('SEED', '?')}"
    )
    if "ippo" in str(algo).lower():
        line2 += (
            f" ent={config.get('ENT_COEF', '?')} "
            f"clip={config.get('CLIP_EPS', '?')}"
        )
    else:
        line2 += (
            f" eps={config.get('EPS_START', '?')}->{config.get('EPS_FINISH', '?')} "
            f"buf={config.get('BUFFER_SIZE', '?')}"
        )
    if n_frames is not None:
        line2 += f" | frames={n_frames}"
    return [line1, line2]


def _overlay_title(frame: onp.ndarray, lines: list[str]) -> onp.ndarray:
    """Paste a dark title bar above the frame with the given text lines."""
    img = Image.fromarray(onp.asarray(frame, dtype=onp.uint8))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    line_h = 14
    pad = 4
    bar_h = pad * 2 + line_h * max(len(lines), 1)
    canvas = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate(lines):
        # Truncate extremely long lines so they don't wrap off the canvas.
        max_chars = max(20, img.width // 6)
        if len(text) > max_chars:
            text = text[: max_chars - 1] + "…"
        draw.text(
            (6, pad + i * line_h),
            text,
            fill=(240, 240, 240),
            font=font,
        )
    return onp.asarray(canvas)


class GifRenderer:
    """Stateless renderer: give it params, get a GIF on disk."""

    def __init__(
        self,
        config: dict,
        env_unwrapped,
        gif_dir: Path | str,
        *,
        max_steps: int | None = None,
        fps: int = 10,
        cfg_name: str | None = None,
    ):
        self.config = dict(config)
        if cfg_name:
            self.config["_CFG_NAME"] = str(cfg_name)
        self.env = env_unwrapped
        self.gif_dir = Path(gif_dir)
        self.gif_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.start_window_steps = int(config.get("RENDER_GIF_START_STEPS", 250))
        self.tail_window_steps = int(
            config.get(
                "RENDER_GIF_TAIL_STEPS",
                config.get("RENDER_GIF_STEPS", 100),
            )
        )
        self.capture_tail = bool(self.config.get("RENDER_GIF_TAIL", True))
        self.episode_steps = int(
            self.config.get("ENV_KWARGS", {}).get(
                "num_inner_steps", self.start_window_steps
            )
        )
        self.num_agents = env_unwrapped.num_agents
        self.agents = list(env_unwrapped.agents)
        # Prefer live agent count over stale config for titles.
        env_kw = dict(self.config.get("ENV_KWARGS") or {})
        env_kw.setdefault("num_agents", self.num_agents)
        self.config["ENV_KWARGS"] = env_kw
        self.config["num_agents"] = self.num_agents
        self.save_episode_diag = bool(
            self.config.get(
                "RENDER_EPISODE_DIAG",
                bool(self.config.get("ENABLE_REPUTATION", False))
                or bool(self.config.get("ENABLE_REPUTATION_V2", False)),
            )
        )
        self.save_occupancy_heatmap = bool(
            self.config.get("RENDER_OCCUPANCY_HEATMAP", False)
        )
        self.want_gif = bool(self.config.get("RENDER_GIF", True))
        # Cap RGB capture at N frames (0 = use start_window_steps / tail as before).
        self.gif_max_frames = int(self.config.get("RENDER_GIF_MAX_FRAMES", 0) or 0)
        self.canonical_orient = bool(
            self.config.get("RENDER_GIF_CANONICAL_ORIENT", True)
        )
        enable_rep = bool(self.config.get("ENABLE_REPUTATION", False)) or bool(
            self.config.get("ENABLE_REPUTATION_V2", False)
        )
        self.rep_side_panel = bool(
            self.config.get("RENDER_GIF_REP_PANEL", enable_rep and self.want_gif)
        )
        self.overlay_rep_on_map = bool(
            self.config.get("RENDER_GIF_OVERLAY_REP", False)
        )
        # Clean mode: grid only + Good/Bad agent colours + compact legend.
        # Turns off title bar, numeric rep panel, and on-map i:±rep labels.
        self.clean_good_bad = bool(
            self.config.get("RENDER_GIF_CLEAN_GOOD_BAD", False)
        )
        if self.clean_good_bad:
            self.rep_side_panel = False
            self.overlay_rep_on_map = False
        self._tile_size = 32
        self._padding = int(getattr(env_unwrapped, "PADDING", 0))
        self._default_player_colours = None
        self._last_good_bad_colours = None
        harvest = self._harvest_env()
        if hasattr(harvest, "PLAYER_COLOURS"):
            self._default_player_colours = list(harvest.PLAYER_COLOURS)

        schedule = sorted(gif_render_updates(self.config))
        if self.want_gif and self.gif_max_frames > 0:
            if self.save_episode_diag:
                horizon_note = (
                    f" + full episode T={self.episode_steps} for diag plots"
                )
            else:
                horizon_note = " (GIF-only; stop after frame budget)"
            print(
                f"[render_gif] short GIF mode: {self.gif_max_frames} frames"
                f"{horizon_note} -> {self.gif_dir}"
            )
        elif schedule and self.want_gif:
            tail_note = (
                f" + last {self.tail_window_steps} steps"
                if self.capture_tail and self.episode_steps > self.tail_window_steps
                else ""
            )
            n_gifs = len(schedule) * (2 if tail_note else 1)
            diag_note = (
                " + per-agent episode plots"
                if self.save_episode_diag
                else ""
            )
            panel_note = (
                " + Good/Bad legend"
                if self.clean_good_bad
                else (" + rep side panel" if self.rep_side_panel else "")
            )
            print(
                f"[render_gif] will save ~{n_gifs} GIFs at updates "
                f"{schedule} (first {self.start_window_steps} steps{tail_note}, "
                f"episode T={self.episode_steps}{diag_note}{panel_note}) -> {self.gif_dir}"
            )
        elif self.save_episode_diag and not self.want_gif:
            print(
                f"[render_gif] diag-only mode (no GIFs) "
                f"episode T={self.episode_steps} -> {self.gif_dir}"
            )
        else:
            print("[render_gif] RENDER_GIF disabled - no GIFs scheduled")

    def _slice_params_for_seed(self, params, seed_i: int):
        sample_leaf = jax.tree_util.tree_leaves(params)[0]
        if sample_leaf.ndim > 0 and sample_leaf.shape[0] > 1:
            try:
                return jax.tree_util.tree_map(lambda x: x[seed_i], params)
            except (IndexError, TypeError):
                pass
        return params

    def _harvest_env(self):
        """Drill wrappers to the Harvest env that owns PLAYER_COLOURS / render."""
        env = self.env
        for _ in range(6):
            if hasattr(env, "PLAYER_COLOURS") and hasattr(env, "render_tile"):
                return env
            if hasattr(env, "_env"):
                env = env._env
            else:
                break
        return self.env

    def _apply_good_bad_agent_colours(self, rep: onp.ndarray) -> None:
        """Tint agent triangles green/red from current reputation labels.

        Tile cache keys include RGB (see ``Harvest_open.render_tile``), so we
        only swap ``PLAYER_COLOURS`` — no per-frame cache flush.
        """
        harvest = self._harvest_env()
        if not hasattr(harvest, "PLAYER_COLOURS"):
            return
        colors = _rep_to_agent_colors(rep, self.num_agents)
        if getattr(self, "_last_good_bad_colours", None) == colors:
            return
        harvest.PLAYER_COLOURS = colors
        self._last_good_bad_colours = colors

    def _restore_default_agent_colours(self) -> None:
        if self._default_player_colours is None:
            return
        harvest = self._harvest_env()
        if hasattr(harvest, "PLAYER_COLOURS"):
            harvest.PLAYER_COLOURS = list(self._default_player_colours)
            self._last_good_bad_colours = None

    def _rep_tagged_for_state(self, state) -> tuple[onp.ndarray, onp.ndarray]:
        n = self.num_agents
        rep = _rep_from_state(state)
        if rep is None:
            rep_arr = onp.zeros((n,), dtype=onp.float32)
        else:
            rep_arr = onp.asarray(rep, dtype=onp.float32).reshape(-1)[:n]
        tagged = _timeout_from_state(state)
        if tagged is None:
            tagged_arr = onp.zeros((n,), dtype=onp.float32)
        else:
            tagged_arr = onp.asarray(tagged, dtype=onp.float32).reshape(-1)[:n]
        return rep_arr, tagged_arr

    def _capture_env_frame(self, state) -> onp.ndarray:
        frame = onp.asarray(self.env.render(state), dtype=onp.uint8)
        if self.canonical_orient:
            frame = canonicalize_render_frame(frame)
        enable_rep = bool(self.config.get("ENABLE_REPUTATION", False)) or bool(
            self.config.get("ENABLE_REPUTATION_V2", False)
        )
        if self.clean_good_bad:
            # Paint Good/Bad colours onto the RGB frame (do not rely on
            # PLAYER_COLOURS / tile cache — those were not updating reliably).
            inner = _unwrap_render_state(state)
            rep_arr, tagged_arr = self._rep_tagged_for_state(state)
            if hasattr(inner, "agent_locs"):
                frame = paint_agents_good_bad(
                    frame,
                    onp.asarray(inner.agent_locs),
                    rep_arr,
                    tagged=tagged_arr,
                    tile_size=self._tile_size,
                )
        elif self.overlay_rep_on_map and enable_rep:
            inner = _unwrap_render_state(state)
            rep = _rep_from_state(state)
            tagged = _timeout_from_state(state)
            if rep is not None and hasattr(inner, "agent_locs"):
                frame = _overlay_reputation(
                    frame,
                    onp.asarray(inner.agent_locs),
                    onp.asarray(rep),
                    tagged=tagged,
                    tile_size=self._tile_size,
                    padding=self._padding,
                )
        return frame

    def _finalize_gif_frame(
        self, frame: onp.ndarray, rep: onp.ndarray, tagged: onp.ndarray
    ) -> onp.ndarray:
        if self.clean_good_bad:
            return compose_frame_with_good_bad_legend(frame)
        if self.rep_side_panel:
            return compose_frame_with_rep_panel(
                frame,
                rep,
                tagged=tagged,
                num_agents=self.num_agents,
            )
        return frame

    def _collect_greedy_rollout(
        self,
        params,
        *,
        rng: jax.Array,
        pick_actions: Callable,
    ) -> tuple[list[onp.ndarray], list[onp.ndarray], int, dict[str, onp.ndarray]]:
        """Run a greedy eval episode; return (first_window, last_window, ep_len, trace).

        ``pick_actions`` receives ``(params, obs_batch, rep_batch_or_None, obs_was_dict)``
        and returns actions shaped ``(N,)``. ``trace`` arrays are ``(T, N)``.

        Frame lists hold ``(rgb, rep, tagged)`` tuples before GIF encoding.
        """
        rng, reset_key = jax.random.split(rng)
        obs, state = self.env.reset(reset_key)
        n = self.num_agents

        def _frame_bundle(st):
            rep_arr, tagged_arr = self._rep_tagged_for_state(st)
            if self.want_gif:
                rgb = self._capture_env_frame(st)
            else:
                rgb = onp.zeros((1, 1, 3), dtype=onp.uint8)
            return rgb, rep_arr, tagged_arr

        buckets: dict[str, list[onp.ndarray]] = {
            "harvest": [],
            "reward": [],
            "zap_successful": [],
            "zap_attempted": [],
            "rep": [],
            "raw_rep": [],
            "tagged": [],
            "rate_ema": [],
            "burst_count": [],
            "burst_penalty": [],
        }

        start_frames: list[tuple[onp.ndarray, onp.ndarray, onp.ndarray]] = []
        if self.want_gif:
            start_frames.append(_frame_bundle(state))
        frame_budget = (
            self.gif_max_frames
            if self.gif_max_frames > 0
            else (self.start_window_steps + 1)
        )
        capture_tail = self.want_gif and self.capture_tail and self.gif_max_frames <= 0
        # GIF-only short preview: stop once enough frames (skip rest of episode).
        # Keep running for episode-diag / occupancy heatmap (full T).
        gif_only_short = (
            self.want_gif
            and (not self.save_episode_diag)
            and (not self.save_occupancy_heatmap)
            and self.gif_max_frames > 0
            and (not capture_tail)
        )
        collect_trace = bool(self.save_episode_diag)
        collect_occ = bool(self.save_occupancy_heatmap)
        tail_frames: deque[tuple[onp.ndarray, onp.ndarray, onp.ndarray]] = deque(
            maxlen=self.tail_window_steps + 1 if capture_tail else 1
        )

        # Police indices for police-only occupancy panel.
        ekw = self.config.get("ENV_KWARGS") or {}
        if not isinstance(ekw, dict):
            ekw = {}
        police_ids = ekw.get("police_agent_ids")
        n_pol = int(self.config.get("NUM_POLICE_AGENTS", 0) or 0)
        n_harv = int(self.config.get("ENV_NUM_AGENTS", 0) or 0)
        enable_police = bool(self.config.get("ENABLE_POLICE_AGENT", False))
        if not police_ids:
            # Prefer harvester-count split; else last NUM_POLICE agents.
            if n_pol > 0 and 0 < n_harv < n:
                police_ids = list(range(n_harv, min(n, n_harv + n_pol)))
            elif n_pol > 0 or enable_police:
                n_pol = max(n_pol, 1 if enable_police else 0)
                police_ids = list(range(max(0, n - n_pol), n))
            else:
                police_ids = []
        police_ids = [int(i) for i in police_ids if 0 <= int(i) < n]
        police_set = set(police_ids)
        if collect_occ:
            print(
                f"[render_gif] occupancy police_ids={police_ids} "
                f"n_agents={n} ENV_NUM_AGENTS={n_harv} NUM_POLICE={n_pol}"
            )
            if not police_set:
                print(
                    "[render_gif] WARNING: no police ids resolved — "
                    "police heatmap panel will be unavailable"
                )

        occ_all = None
        occ_police = None
        occ_agents = None
        wall_mask = None

        def _bump_occupancy(st) -> None:
            nonlocal occ_all, occ_police, occ_agents, wall_mask
            if not collect_occ:
                return
            inner = _unwrap_render_state(st)
            if not hasattr(inner, "agent_locs") or not hasattr(inner, "grid"):
                return
            grid = onp.asarray(inner.grid)
            if wall_mask is None:
                # Items.wall == 1 in harvest_open.
                wall_mask = grid == 1
                occ_all = onp.zeros(grid.shape, dtype=onp.float32)
                occ_agents = onp.zeros((n, *grid.shape), dtype=onp.float32)
                # Always allocate police map when we have ids (even if empty path).
                if police_set:
                    occ_police = onp.zeros(grid.shape, dtype=onp.float32)
            locs = onp.asarray(inner.agent_locs)
            h, w = occ_all.shape
            for a_i, loc in enumerate(locs):
                r, c = int(loc[0]), int(loc[1])
                if 0 <= r < h and 0 <= c < w and not wall_mask[r, c]:
                    occ_all[r, c] += 1.0
                    if occ_agents is not None and a_i < occ_agents.shape[0]:
                        occ_agents[a_i, r, c] += 1.0
                    if occ_police is not None and a_i in police_set:
                        occ_police[r, c] += 1.0

        _bump_occupancy(state)

        ep_len = 0
        for _ in range(self.episode_steps):
            was_dict = isinstance(obs, dict)
            obs_batch = _obs_to_batch(obs, self.agents)
            rep_batch = _rep_from_obs(obs, self.agents, self.num_agents)
            actions = pick_actions(params, obs_batch, rep_batch, was_dict)
            step_actions = actions.astype(jnp.int32)
            rng, step_key = jax.random.split(rng)
            obs, state, reward, done, info = self.env.step(
                step_key, state, step_actions
            )
            need_frame = self.want_gif and (
                len(start_frames) < frame_budget or capture_tail
            )
            frame_bundle = _frame_bundle(state) if need_frame else None
            ep_len += 1
            _bump_occupancy(state)
            if collect_trace:
                info = info or {}
                buckets["harvest"].append(
                    _to_agent_vec(
                        info.get("environmental_rewards")
                        if isinstance(info, dict)
                        else None,
                        n,
                        self.agents,
                    )
                )
                buckets["reward"].append(_to_agent_vec(reward, n, self.agents))
                buckets["zap_successful"].append(
                    _to_agent_vec(
                        info.get("zap_successful")
                        if isinstance(info, dict)
                        else None,
                        n,
                        self.agents,
                    )
                )
                buckets["zap_attempted"].append(
                    _to_agent_vec(
                        info.get("zap_attempted")
                        if isinstance(info, dict)
                        else None,
                        n,
                        self.agents,
                    )
                )
                tagged = _to_agent_vec(
                    info.get("tagged") if isinstance(info, dict) else None,
                    n,
                    self.agents,
                )
                buckets["tagged"].append(tagged.astype(onp.float32))
                if isinstance(info, dict) and "rep" in info:
                    buckets["rep"].append(_to_agent_vec(info["rep"], n, self.agents))
                else:
                    rep_now = _rep_from_state(state)
                    buckets["rep"].append(
                        onp.asarray(rep_now, dtype=onp.float32)
                        if rep_now is not None
                        else onp.zeros((n,), dtype=onp.float32)
                    )
                if isinstance(info, dict) and "raw_rep" in info:
                    buckets["raw_rep"].append(
                        _to_agent_vec(info["raw_rep"], n, self.agents)
                    )
                else:
                    buckets["raw_rep"].append(onp.zeros((n,), dtype=onp.float32))
                ema = _rate_ema_from_state(state)
                buckets["rate_ema"].append(
                    onp.asarray(ema, dtype=onp.float32)
                    if ema is not None
                    else onp.full((n,), onp.nan, dtype=onp.float32)
                )
                if isinstance(info, dict) and "burst_count" in info:
                    buckets["burst_count"].append(
                        _to_agent_vec(info["burst_count"], n, self.agents)
                    )
                else:
                    buckets["burst_count"].append(
                        onp.full((n,), onp.nan, dtype=onp.float32)
                    )
                if isinstance(info, dict) and "burst_penalty" in info:
                    buckets["burst_penalty"].append(
                        _to_agent_vec(info["burst_penalty"], n, self.agents)
                    )
                else:
                    buckets["burst_penalty"].append(
                        onp.full((n,), onp.nan, dtype=onp.float32)
                    )
            if self.want_gif:
                if len(start_frames) < frame_budget:
                    start_frames.append(frame_bundle)
                if capture_tail:
                    tail_frames.append(frame_bundle)
            if gif_only_short and len(start_frames) >= frame_budget:
                break
            if _episode_done(done):
                break

        start_out = start_frames if self.want_gif else []
        tail_out = list(tail_frames) if capture_tail else []
        trace = {
            key: (
                onp.stack(vals, axis=0)
                if vals
                else onp.zeros((0, n), dtype=onp.float32)
            )
            for key, vals in buckets.items()
        }
        if collect_occ and occ_all is not None:
            trace["_occupancy"] = occ_all
            trace["_wall"] = wall_mask.astype(onp.uint8)
            if occ_agents is not None:
                trace["_occupancy_agents"] = occ_agents
            if occ_police is not None:
                trace["_occupancy_police"] = occ_police
        return start_out, tail_out, ep_len, trace

    def _save_rollout_gifs(
        self,
        start_frames: list[tuple[onp.ndarray, onp.ndarray, onp.ndarray]],
        tail_frames: list[tuple[onp.ndarray, onp.ndarray, onp.ndarray]],
        update: int,
        label: str,
        *,
        seed_i: int = 0,
        ep_len: int = 0,
        trace: dict[str, onp.ndarray] | None = None,
    ) -> list[Path]:
        paths: list[Path] = []
        if self.want_gif:
            if start_frames:
                short = self.gif_max_frames > 0
                paths.append(
                    self._save_gif(
                        start_frames,
                        update,
                        label,
                        seed_i=seed_i,
                        suffix="preview" if short else "start",
                        window=(
                            f"{len(start_frames)} frames (ep_len={ep_len})"
                            if short
                            else f"first {len(start_frames) - 1} steps (ep_len={ep_len})"
                        ),
                    )
                )
            if (
                self.capture_tail
                and self.gif_max_frames <= 0
                and ep_len > self.tail_window_steps
                and len(tail_frames) > 1
            ):
                paths.append(
                    self._save_gif(
                        tail_frames,
                        update,
                        label,
                        seed_i=seed_i,
                        suffix="end",
                        window=f"last {len(tail_frames) - 1} steps (ep_len={ep_len})",
                    )
                )
        if (
            self.save_episode_diag
            and trace is not None
            and trace["harvest"].shape[0] > 0
        ):
            slug = build_run_slug(self.config, algo=label, seed_i=seed_i)
            diag_path = self.gif_dir / f"{slug}_u{update:06d}_episode_diag.png"
            r_star = (self.config.get("REPUTATION") or {}).get("r_star")
            overlay = bool(self.config.get("OVERLAY_REPUTATION", False))
            saved = save_episode_diag(
                trace,
                diag_path,
                r_star=float(r_star) if r_star is not None else None,
                title=(
                    f"{label} episode diag | update={update} | seed={seed_i} "
                    f"| T={ep_len} | ENABLE_REPUTATION="
                    f"{bool(self.config.get('ENABLE_REPUTATION', False))}"
                    + (" | overlay" if overlay else "")
                ),
            )
            paths.append(saved)
            print(f"[render_gif] saved episode diag {saved}")
        if (
            self.save_occupancy_heatmap
            and trace is not None
            and "_occupancy" in trace
        ):
            slug = build_run_slug(self.config, algo=label, seed_i=seed_i)
            occ_mode = str(
                self.config.get("RENDER_OCCUPANCY_MODE", "auto") or "auto"
            ).lower()
            occ_pol = trace.get("_occupancy_police")
            occ_ag = trace.get("_occupancy_agents")
            if occ_mode == "auto":
                occ_mode = (
                    "police"
                    if occ_pol is not None
                    else ("agents" if occ_ag is not None else "all")
                )
            suffix = "occupancy_heatmap"
            if occ_mode == "agents":
                suffix = "occupancy_agents_heatmap"
            heat_path = self.gif_dir / f"{slug}_u{update:06d}_{suffix}.png"
            if occ_mode == "police" and occ_pol is None:
                raise RuntimeError(
                    "RENDER_OCCUPANCY_MODE=police but no police occupancy was "
                    "collected (check ENABLE_POLICE_AGENT / police_agent_ids)."
                )
            if occ_mode == "agents" and occ_ag is None:
                raise RuntimeError(
                    "RENDER_OCCUPANCY_MODE=agents but per-agent occupancy missing."
                )
            saved_heat = save_occupancy_heatmap(
                trace["_occupancy"],
                trace["_wall"],
                heat_path,
                occupancy_police=occ_pol,
                occupancy_agents=occ_ag,
                ep_len=ep_len,
                mode=occ_mode,
            )
            paths.append(saved_heat)
            print(
                f"[render_gif] saved {occ_mode} occupancy heatmap {saved_heat} "
                f"(all visits={float(onp.sum(trace['_occupancy'])):.0f})"
            )
        return [p for p in paths if p is not None]

    # ------------------------------------------------------------------
    # DQN (shared): single Q-network, all agents share params
    # ------------------------------------------------------------------
    def render_dqn_shared(
        self,
        params,
        update: int,
        network_apply: Callable,
        *,
        rng: jax.Array | None = None,
        seed_i: int = 0,
        label: str = "dqn",
    ) -> list[Path] | None:
        """Greedy rollout with a shared Q-network."""
        if not _ensure_pil():
            return None

        if rng is None:
            rng = jax.random.PRNGKey(update + 9999)

        params = self._slice_params_for_seed(params, seed_i)

        def pick_actions(p, obs_batch, _was_dict):
            q = network_apply(p, obs_batch)
            return jnp.argmax(q, axis=-1).astype(jnp.int32)

        def pick_actions_wrapped(p, obs_batch, _rep, was_dict):
            return pick_actions(p, obs_batch, was_dict)

        start_frames, tail_frames, ep_len, trace = self._collect_greedy_rollout(
            params, rng=rng, pick_actions=pick_actions_wrapped
        )
        return self._save_rollout_gifs(
            start_frames,
            tail_frames,
            update,
            label,
            seed_i=seed_i,
            ep_len=ep_len,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # DQN (independent): per-agent Q-networks, params stacked on axis 0
    # ------------------------------------------------------------------
    def render_dqn_indep(
        self,
        params,
        update: int,
        network_apply: Callable,
        *,
        rng: jax.Array | None = None,
        seed_i: int = 0,
        label: str = "dqn_indep",
    ) -> list[Path] | None:
        """Greedy rollout with independent per-agent Q-networks."""
        if not _ensure_pil():
            return None

        if rng is None:
            rng = jax.random.PRNGKey(update + 9999)

        # Slice seed if multi-seed batched.
        sample_leaf = jax.tree_util.tree_leaves(params)[0]
        expected_agent_dim = self.num_agents
        if sample_leaf.ndim >= 2 and sample_leaf.shape[0] != expected_agent_dim:
            params = self._slice_params_for_seed(params, seed_i)

        def pick_actions(p, obs_batch, _was_dict):
            actions_list = []
            for i in range(self.num_agents):
                params_i = jax.tree_util.tree_map(lambda x: x[i], p)
                obs_i = obs_batch[i][None, ...]
                q_i = network_apply(params_i, obs_i)
                actions_list.append(jnp.argmax(q_i, axis=-1).astype(jnp.int32)[0])
            return jnp.stack(actions_list, axis=0)

        def pick_actions_wrapped(p, obs_batch, _rep, was_dict):
            return pick_actions(p, obs_batch, was_dict)

        start_frames, tail_frames, ep_len, trace = self._collect_greedy_rollout(
            params, rng=rng, pick_actions=pick_actions_wrapped
        )
        return self._save_rollout_gifs(
            start_frames,
            tail_frames,
            update,
            label,
            seed_i=seed_i,
            ep_len=ep_len,
            trace=trace,
        )

    # ------------------------------------------------------------------
    # IPPO: per-agent actor-critics, params stacked on axis 0
    # ------------------------------------------------------------------
    def render_ippo(
        self,
        params,
        update: int,
        network_apply: Callable,
        *,
        rng: jax.Array | None = None,
        seed_i: int = 0,
        label: str = "ippo",
    ) -> list[Path] | None:
        """Greedy (argmax logits) rollout with independent actor-critics."""
        if self.want_gif and not _ensure_pil():
            return None

        if rng is None:
            rng = jax.random.PRNGKey(update + 9999)

        # Slice seed if multi-seed batched.
        sample_leaf = jax.tree_util.tree_leaves(params)[0]
        expected_agent_dim = self.num_agents
        if sample_leaf.ndim >= 2 and sample_leaf.shape[0] != expected_agent_dim:
            params = self._slice_params_for_seed(params, seed_i)

        def pick_actions(p, obs_batch, _rep_batch, _was_dict):
            actions_list = []
            for i in range(self.num_agents):
                params_i = jax.tree_util.tree_map(lambda x: x[i], p)
                obs_i = obs_batch[i][None, ...]
                pi, _ = network_apply(params_i, obs_i)
                actions_list.append(jnp.argmax(pi.logits, axis=-1).astype(jnp.int32)[0])
            return jnp.stack(actions_list, axis=0)

        start_frames, tail_frames, ep_len, trace = self._collect_greedy_rollout(
            params, rng=rng, pick_actions=pick_actions
        )
        return self._save_rollout_gifs(
            start_frames,
            tail_frames,
            update,
            label,
            seed_i=seed_i,
            ep_len=ep_len,
            trace=trace,
        )

    # ------------------------------------------------------------------
    def _save_gif(
        self,
        frames: list[tuple[onp.ndarray, onp.ndarray, onp.ndarray]],
        update: int,
        label: str,
        *,
        seed_i: int = 0,
        suffix: str = "",
        window: str | None = None,
    ) -> Path | None:
        if not frames:
            return None

        if self.clean_good_bad:
            composed = [
                self._finalize_gif_frame(rgb, rep, tagged)
                for rgb, rep, tagged in frames
            ]
        else:
            title_lines = build_title_lines(
                self.config,
                algo=label,
                update=update,
                seed_i=seed_i,
                n_frames=len(frames),
                window=window,
            )
            composed = [
                _overlay_title(
                    self._finalize_gif_frame(rgb, rep, tagged),
                    title_lines,
                )
                for rgb, rep, tagged in frames
            ]

        slug = build_run_slug(self.config, algo=label, seed_i=seed_i)
        suffix_part = f"_{suffix}" if suffix else ""
        clean_tag = "_goodbad" if self.clean_good_bad else ""
        path = self.gif_dir / f"{slug}_u{update:06d}{suffix_part}{clean_tag}.gif"
        duration_ms = int(1000 / self.fps)
        imgs = [Image.fromarray(f) for f in composed]
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"[render_gif] saved {path} ({len(frames)} frames)")
        if self.clean_good_bad:
            print("[render_gif] clean Good/Bad grid + legend (no title bar)")
        else:
            title_lines = build_title_lines(
                self.config,
                algo=label,
                update=update,
                seed_i=seed_i,
                n_frames=len(frames),
                window=window,
            )
            print(f"[render_gif] title: {title_lines[0]}")
            print(f"[render_gif]         {title_lines[1]}")
        return path
