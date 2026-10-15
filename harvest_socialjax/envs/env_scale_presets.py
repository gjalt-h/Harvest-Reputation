"""Paper vs test vs tiny environment scale presets (Lupu & Precup 2020 / Perolat).

Single source of truth for geometry that should apply to **all** paper-scale
runs and **all** test-scale runs. Training YAMLs select a scale with::

    ENV_SCALE: paper   # or paper_perolat | test | tiny

Train scripts call ``apply_env_scale(config)`` before ``socialjax.make`` so
``ENV_KWARGS`` receive these values. Gifting mode and algorithm hyperparams
stay in the per-run YAML.

Settled measurements (Opus, from paper figures — not Melting Pot 16×22):
  - Both Lupu Fig. 1 and Perolat Fig. 6b/1a: **20×40** playable interior
  - Lupu: 180 apples (22.5%), plus-shaped patches, N=10, bottom-right spawn
  - Paper obs/beam: 20 ahead / 1 back / 10 side; beam 20×5 (Perolat App. A)
  - Test: half-scale Lupu 12×24, N=4, obs/beam scaled by 12/20 = 0.6
  - Obs/beam depth counts **include the agent cell** (no backward obs rows).
  - Tiny: half of test dims (6×12, N=2), beam 6×3, spaced plus-patch apples

Configs **without** ``ENV_SCALE`` keep today's constructor defaults (16×22,
SocialJax map, beam module defaults) — nothing that already works breaks.
"""
from __future__ import annotations

from typing import Any, Literal, Mapping, MutableMapping

import math

import numpy as onp

EnvScale = Literal["paper", "paper_perolat", "test", "tiny"]
AppleLayoutMode = Literal["scaffold", "plus_patch", "auto"]

# ---------------------------------------------------------------------------
# PAPER_* — current thesis paper preset (Lupu Fig. 1 on a 20×40 open map)
# ---------------------------------------------------------------------------
PAPER_GRID_SIZE: tuple[int, int] = (20, 40)
PAPER_NUM_AGENTS: int = 10
PAPER_NUM_INNER_STEPS: int = 1000
PAPER_TAG_TIMEOUT: int = 25
# Depth includes the agent cell (19 forward + agent = 20 rows); no backward rows.
PAPER_OBS_FORWARD: int = 19
PAPER_OBS_BACKWARD: int = 0
PAPER_OBS_SIDE: int = 10
PAPER_BEAM_LENGTH: int = 20
PAPER_BEAM_WIDTH: int = 5

# Apple layout traced from Lupu & Precup (2020) Fig. 1 (MiniGrid re-impl).
# Spawn: bottom-right 5×5 (rows 15–19, cols 35–39). Two cells are ``Q`` so the
# existing Harvest reset path (expects ≥2 inside spawns) keeps working; the
# rest are ``P``.
_PAPER_APPLES_ONLY: list[str] = [
    "                         A     AAA AAA  ",
    "          A     A     A AAA     A   A   ",
    "  A    A AAA   AAA   AAA A A            ",
    " AAA  AAA A     A     A   AAA           ",
    "  A    A     A             A            ",
    "            AAA             A   A   A   ",
    "          A  A    A   A    AAA AAA AAA  ",
    " A     A AAA     AAA AAA    A  AA   A   ",
    "AAA   AAA A       A   AAA     AAAA      ",
    " A     A A             A       AA  A    ",
    "AAA     AAA           A  A    A   AAA   ",
    " A       A A     AA  AAAAAA  AAA   A    ",
    "   A  A   AAA   AAAA  A  A    A         ",
    "  AAAAAA   A     AA                     ",
    "   A  A                                 ",
    "                                        ",
    "                 A    A    A            ",
    "   A    A    A  AAA  AAA  AAA           ",
    "  AAA  AAA  AAA  A    A    A            ",
    "   A    A    A                          ",
]


def _with_spawn_corner(
    rows: list[str],
    *,
    row_slice: range,
    col_slice: range,
) -> list[str]:
    """Fill a rectangular corner with ``P``, then mark two cells as ``Q``.

    Rows shorter than ``col_slice.stop`` are right-padded with spaces so
    hand-edited ASCII maps with uneven trailing whitespace still load.
    """
    width = max(len(r) for r in rows)
    width = max(width, col_slice.stop)
    grid = [list(r.ljust(width)) for r in rows]
    coords = [(r, c) for r in row_slice for c in col_slice]
    for r, c in coords:
        if grid[r][c] not in (" ", "P", "Q"):
            raise ValueError(
                f"spawn corner collides with {grid[r][c]!r} at {(r, c)}"
            )
        grid[r][c] = "P"
    # Prefer the two most top-left cells of the corner for inside-spawns.
    grid[coords[0][0]][coords[0][1]] = "Q"
    grid[coords[1][0]][coords[1][1]] = "Q"
    return ["".join(r) for r in grid]


PAPER_MAP_ASCII: list[str] = _with_spawn_corner(
    _PAPER_APPLES_ONLY,
    row_slice=range(15, 20),
    col_slice=range(35, 40),
)

# ---------------------------------------------------------------------------
# PEROLAT_FIG6B_* — Perolat et al. 2017 Fig. 1a / Fig. 6b open map (20×40)
# ---------------------------------------------------------------------------
# Cell-centre trace of the paper figure (not Melting Pot 16×22). Apple cells
# match gifting_test_outputs/_paper_extract/perolat_fig1a_recon.txt.
# Spawn: 10 empty BR tiles where agents stand in the PDF (P/Q for env reset).
PEROLAT_FIG6B_GRID_SIZE: tuple[int, int] = (20, 40)
PEROLAT_FIG6B_OPEN_MAP_ASCII: list[str] = [
    "............A......AAAAA...........A....",
    "........A...AA.........AAA........A..A..",
    ".....A.AAA..AAA....A....A..A.AAA.AAAA...",
    ".A..AAA.A....A..A.AAA..A..AAA.A...A.A...",
    "AAA..A.A....A..AAA.A..AAA..A.A.......A..",
    ".A.A..AAA..AAA..A.A....A.A..AAA...AA.AA.",
    "..AAA..A....A.A..AAA....AAA..A.....AAA..",
    "...A.........AAA..A......A..........A...",
    "..AAA....A....A...............A....AAA..",
    "...A..A.AAA....A..A.AAA......A..AAA.A...",
    "..A..AAA.A....A..AAA.A......AAA..A.A....",
    ".AAA..A.A....AAA..A.A........A.A..AAA...",
    "..A.A..AAA....A.A..AAA........AAA..A....",
    "...AAA..A......AAA..A....A..A.AAA.......",
    "....A.......A..A.AAA....A..AAA.A........",
    "A..AAA..A..A..AAA.A....AAA..A.A.....Q...",
    "....A.A...AAA..A.A......A.A..AAA...P..P.",
    ".....AAA...A.A..AAA......AAA..A..P......",
    ".A....A.....AAA..A........A.......P.P.P.",
    ".............A..................P..P..Q.",
]

# ---------------------------------------------------------------------------
# PAPER_PEROLAT_* — Melting Pot foundation-code open layout (16×22, not Fig 6b)
# ---------------------------------------------------------------------------
PAPER_PEROLAT_GRID_SIZE: tuple[int, int] = (16, 22)
PAPER_PEROLAT_NUM_AGENTS: int = 10
PAPER_PEROLAT_NUM_INNER_STEPS: int = 1000
PAPER_PEROLAT_TAG_TIMEOUT: int = 25
PAPER_PEROLAT_OBS_FORWARD: int = 19
PAPER_PEROLAT_OBS_BACKWARD: int = 0
PAPER_PEROLAT_OBS_SIDE: int = 10
PAPER_PEROLAT_BEAM_LENGTH: int = 20
PAPER_PEROLAT_BEAM_WIDTH: int = 5

# Apples/spawns copied exactly from:
# foundation_code/meltingpot-main/meltingpot/configs/substrates/commons_harvest__open.py
PAPER_PEROLAT_MAP_ASCII: list[str] = [
    "AAA    A      A    AAA",
    "AA    AAA    AAA    AA",
    "A    AAAAA  AAAAA    A",
    "      AAA    AAA      ",
    "       A      A       ",
    "  A                A  ",
    " AAA  Q        Q  AAA ",
    "AAAAA            AAAAA",
    " AAA              AAA ",
    "  A                A  ",
    "                      ",
    "                      ",
    "                      ",
    "  PPPPPPPPPPPPPPPPPP  ",
    " PPPPPPPPPPPPPPPPPPPP ",
    "PPPPPPPPPPPPPPPPPPPPPP",
]

# ---------------------------------------------------------------------------
# TEST_* — half-scale Lupu (12×24), obs/beam/agents scaled by 12/20 = 0.6
# ---------------------------------------------------------------------------
TEST_GRID_SIZE: tuple[int, int] = (12, 24)
TEST_NUM_AGENTS: int = 4
TEST_NUM_INNER_STEPS: int = 1000
TEST_TAG_TIMEOUT: int = 25
# 12 rows incl. agent (11 ahead); 13 cols incl. agent (6 per side).
TEST_OBS_FORWARD: int = 11
TEST_OBS_BACKWARD: int = 0
TEST_OBS_SIDE: int = 6
TEST_BEAM_LENGTH: int = 12
TEST_BEAM_WIDTH: int = 3  # keep odd so the beam stays centered

# Mixed layout (paper-like): distant orchards for travel + near-spawn apples
# for early learning. ~73 A scaffold; live count = apple_density × 18 × N
# (default density=1 → 90 at N=5, 72 at N=4). BR 3×3 empty (rows 9-11, cols 21-23).
_TEST_APPLES_ONLY: list[str] = [
    "A  A         A  AAAA    ",  # 0  outer top / TR
    " AAA   A    AAA   A A   ",  # 1
    "  A   AAA    A   A A    ",  # 2
    "   A   A       A        ",  # 3
    "  AAA   A     AAA       ",  # 4
    "   A   AAA    AAA       ",  # 5
    "  A     A   A  A        ",  # 6
    " AAA   A   AAA          ",  # 7
    "  A   AAA   A    AAA    ",  # 8  approach to spawn
    " A A   A         AAA    ",  # 9  near-spawn (left of BR)
    "  A   AAA    A    A     ",  # 10 near-spawn
    "     A  A               ",  # 11 near-spawn; BR cols empty
]

TEST_MAP_ASCII: list[str] = _with_spawn_corner(
    _TEST_APPLES_ONLY,
    row_slice=range(9, 12),
    col_slice=range(21, 24),
)

# ---------------------------------------------------------------------------
# TINY_* — half of TEST linear dims / agents / obs; beam 6×3 (odd width).
# Apple layout is hand-placed plus-patches (not a raw 2×2 subsample) so
# travel time / depletion can better track paper-relative ~10% near-empty.
# ---------------------------------------------------------------------------
TINY_GRID_SIZE: tuple[int, int] = (TEST_GRID_SIZE[0] // 2, TEST_GRID_SIZE[1] // 2)  # 6×12
TINY_NUM_AGENTS: int = TEST_NUM_AGENTS // 2  # 2
TINY_NUM_INNER_STEPS: int = 600  # live tiny YAML horizon; not forced by ENV_SCALE
TINY_TAG_TIMEOUT: int = TEST_TAG_TIMEOUT
TINY_OBS_FORWARD: int = TEST_OBS_FORWARD // 2  # 5 → 6 rows incl. agent
TINY_OBS_BACKWARD: int = 0
TINY_OBS_SIDE: int = TEST_OBS_SIDE // 2  # 3
TINY_BEAM_LENGTH: int = TEST_BEAM_LENGTH // 2  # 6
TINY_BEAM_WIDTH: int = 3  # odd (env requires centered beam); same width as test

# Mixed layout: outer TL/TR for travel + near-spawn cluster for learning.
# ~18 apples; BR 2×2 empty (rows 4-5, cols 10-11).
_TINY_APPLES_ONLY: list[str] = [
    "A  A     AA ",  # 0  TL + TR outer
    " AAA      A ",  # 1
    "  A         ",  # 2
    " A A  AA    ",  # 3  mid
    "  A   AAA   ",  # 4  near-spawn (left of BR)
    "     A      ",  # 5  near-spawn
]

TINY_MAP_ASCII: list[str] = _with_spawn_corner(
    _TINY_APPLES_ONLY,
    row_slice=range(TINY_GRID_SIZE[0] - 2, TINY_GRID_SIZE[0]),
    col_slice=range(TINY_GRID_SIZE[1] - 2, TINY_GRID_SIZE[1]),
)

# Geometry keys written into ENV_KWARGS when ENV_SCALE is set.
# ``num_inner_steps`` is intentionally NOT included so smoke configs can keep
# short horizons (e.g. 64) while still using the test map/beam/obs/agents.
# (cnn / jit / shared_rewards stay in the per-run YAML.)
# ``apple_density`` / ``regrowth_scale`` are also NOT geometry keys — they are
# optional ENV_KWARGS (or top-level ENV_APPLE_DENSITY) that scale the orchard.
# ``apple_density`` is a multiplier on apples-per-agent (see APPLES_PER_AGENT).
_GEOMETRY_KEYS: tuple[str, ...] = (
    "num_agents",
    "grid_size",
    "map_ASCII",
    "tag_timeout",
    "obs_forward",
    "obs_backward",
    "obs_side",
    "beam_length",
    "beam_width",
)

# Fixed seed for ``scale_apple_density`` tie-breaks — orchard layout must not
# vary with training / reset PRNG seeds.
APPLE_LAYOUT_SEED: int = 0

# Default orchard budget: 18 apples per agent at apple_density=1.0.
# Examples (density × 18 × N): N=5 → 90; N=4 → 72; N=10 → 180 (matches paper map).
# density=0.5 → 9/agent (45 at N=5, 36 at N=4); density=2 → 36/agent (180 / 144).
APPLES_PER_AGENT: float = 18.0
# Standard test-scale orchard (N=4, plus_patch): 98 apples (24.5/agent).
TEST_PLUS_PATCH_APPLE_COUNT: int = 98
TEST_DEFAULT_APPLE_DENSITY: float = (
    TEST_PLUS_PATCH_APPLE_COUNT / (APPLES_PER_AGENT * TEST_NUM_AGENTS)
)


def target_apple_count(
    num_agents: int,
    density: float = 1.0,
    *,
    apples_per_agent: float = APPLES_PER_AGENT,
) -> int:
    """``round(density * apples_per_agent * num_agents)``, at least 1."""
    n = int(round(float(density) * float(apples_per_agent) * int(num_agents)))
    return max(1, n)


def _apple_neighbor_count(grid: list[list[str]], r: int, c: int) -> int:
    h, w = len(grid), len(grid[0])
    n = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == "A":
                n += 1
    return n


# Classic plus only (radius 1 = 5 cells). Incomplete pluses use a subset of
# these offsets — never radius-2/3 "long" crosses (those looked huddled/weird
# on scarce densities).
_PLUS_ARM_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0),  # centre first
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)
_FULL_PLUS_SIZE: int = 5


def _spawn_forbidden(rows: list[list[str]], *, pad: int = 2) -> set[tuple[int, int]]:
    """Cells near player spawns where plus centres should not land."""
    h, w = len(rows), len(rows[0])
    forbidden: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if rows[r][c] not in ("P", "Q"):
                continue
            for dr in range(-pad, pad + 1):
                for dc in range(-pad, pad + 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        forbidden.add((rr, cc))
    return forbidden


def _scaffold_apple_components(
    grid: list[list[str]],
) -> list[list[tuple[int, int]]]:
    """4-connected apple blobs from the hand-authored scaffold."""
    h, w = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != "A" or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, ccc = cr + dr, cc + dc
                    if (
                        0 <= rr < h
                        and 0 <= ccc < w
                        and grid[rr][ccc] == "A"
                        and (rr, ccc) not in seen
                    ):
                        seen.add((rr, ccc))
                        stack.append((rr, ccc))
            comps.append(cells)
    return comps


def _component_centroid(cells: list[tuple[int, int]]) -> tuple[int, int]:
    rs = sum(r for r, _ in cells) / len(cells)
    cs = sum(c for _, c in cells) / len(cells)
    return int(round(rs)), int(round(cs))


def _stratified_pick(
    items: list[tuple[int, int]],
    k: int,
    *,
    seed: int,
) -> list[tuple[int, int]]:
    """Spatially stratified subsample of centres (preserves spread)."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    ordered = sorted(items)
    offset = int(seed) % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    keep_idx = {int(round(i)) for i in onp.linspace(0, len(rotated) - 1, num=k)}
    keep = [rotated[i] for i in sorted(keep_idx)]
    if len(keep) < k:
        for cell in rotated:
            if cell not in keep:
                keep.append(cell)
            if len(keep) >= k:
                break
    return keep[:k]


def _maximin_pick(
    items: list[tuple[int, int]],
    k: int,
    *,
    seed: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    """Greedy farthest-point subsample — keeps scarce orchards evenly spread."""
    if k <= 0:
        return []
    if k >= len(items):
        return list(items)
    # Prefer starting near the map centre so the first pick is not a corner.
    mid_r, mid_c = (height - 1) / 2.0, (width - 1) / 2.0
    start = min(
        items,
        key=lambda rc: (
            abs(rc[0] - mid_r) + abs(rc[1] - mid_c),
            hash((rc[0], rc[1], seed)) & 0xFFFF,
        ),
    )
    chosen: list[tuple[int, int]] = [start]
    remaining = [p for p in items if p != start]
    while len(chosen) < k and remaining:
        best = max(
            remaining,
            key=lambda p: (
                min(abs(p[0] - c[0]) + abs(p[1] - c[1]) for c in chosen),
                -hash((p[0], p[1], seed, len(chosen))) & 0xFFFF,
            ),
        )
        chosen.append(best)
        remaining.remove(best)
    return chosen


def _farthest_point_centers(
    height: int,
    width: int,
    n_extra: int,
    existing: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    *,
    seed: int,
) -> list[tuple[int, int]]:
    """Add centres that maximise min distance to existing ones (even spread)."""
    added: list[tuple[int, int]] = []
    occupied = list(existing)
    for i in range(n_extra):
        best: tuple[int, int] | None = None
        best_score = (-1.0, 0)
        # Coarse scan then refine — enough for 12×24 / 20×40.
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if (r, c) in forbidden:
                    continue
                if occupied:
                    md = min(abs(r - er) + abs(c - ec) for er, ec in occupied)
                else:
                    md = 999
                tie = hash((r, c, seed, i)) & 0xFFFF
                score = (float(md), -tie)
                if score > best_score:
                    best_score = score
                    best = (r, c)
        if best is None:
            break
        added.append(best)
        occupied.append(best)
    return added


def _plus_cells_of_size(
    grid: list[list[str]],
    center_r: int,
    center_c: int,
    size: int,
    occupied: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    """Place a full (size=5) or incomplete (1–4) plus at ``(center_r, center_c)``."""
    size = max(1, min(int(size), _FULL_PLUS_SIZE))
    h, w = len(grid), len(grid[0])
    cells: list[tuple[int, int]] = []
    for dr, dc in _PLUS_ARM_ORDER[:size]:
        r, c = center_r + dr, center_c + dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        if (r, c) in forbidden or (r, c) in occupied:
            return None
        if grid[r][c] not in (" ", "A"):
            return None
        cells.append((r, c))
    return cells


def _plan_plus_sizes(target: int) -> list[int]:
    """Maximize full pluses (5). Avoid lone size-1 remainders.

    ``5k + 1`` becomes ``5(k-1) + 4 + 2`` so every patch is at least a
    2-cell incomplete plus (except the trivial target=1 case).
    """
    target = int(target)
    if target <= 0:
        return []
    n_full, rem = divmod(target, _FULL_PLUS_SIZE)
    if rem == 0:
        return [_FULL_PLUS_SIZE] * n_full
    if rem == 1:
        if n_full >= 1:
            return [_FULL_PLUS_SIZE] * (n_full - 1) + [4, 2]
        return [1]
    return [_FULL_PLUS_SIZE] * n_full + [rem]


def _max_centers_at_sep(
    height: int,
    width: int,
    forbidden: set[tuple[int, int]],
    *,
    min_sep: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Pack as many centres as fit with Manhattan ``min_sep``."""
    budget = max(1, (height - 2) * (width - 2))
    return _farthest_point_with_sep(
        height,
        width,
        budget,
        forbidden,
        min_sep=min_sep,
        seed=seed,
    )


def _snap_free(
    r: int,
    c: int,
    height: int,
    width: int,
    forbidden: set[tuple[int, int]],
    taken: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Nearest free interior cell to ``(r, c)`` (Manhattan), or None."""
    r = min(height - 2, max(1, r))
    c = min(width - 2, max(1, c))
    if (r, c) not in forbidden and (r, c) not in taken:
        return (r, c)
    for rad in range(1, max(height, width)):
        best: tuple[int, int] | None = None
        best_d = 10**9
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if abs(dr) + abs(dc) != rad:
                    continue
                rr, cc = r + dr, c + dc
                if not (1 <= rr <= height - 2 and 1 <= cc <= width - 2):
                    continue
                if (rr, cc) in forbidden or (rr, cc) in taken:
                    continue
                d = abs(dr) + abs(dc)
                if d < best_d:
                    best_d = d
                    best = (rr, cc)
        if best is not None:
            return best
    return None


def _span_grid_centers(
    height: int,
    width: int,
    n_patches: int,
    forbidden: set[tuple[int, int]],
    *,
    seed: int,
) -> list[tuple[int, int]]:
    """Exactly ``n_patches`` centres on an aspect-aware grid spanning the map.

    Uses a staggered (brick) row layout so gaps are more uniform than a
    plain rectangular lattice subset, then snaps away from spawn pads.
    """
    if n_patches <= 0:
        return []
    r_min, r_max = 1, height - 2
    c_min, c_max = 1, width - 2
    play_h = max(r_max - r_min, 1)
    play_w = max(c_max - c_min, 1)
    aspect = play_w / play_h
    n_cols = max(1, int(round(math.sqrt(n_patches * aspect))))
    n_rows = max(1, int(math.ceil(n_patches / n_cols)))
    while n_cols * n_rows < n_patches:
        if play_w / n_cols >= play_h / n_rows:
            n_cols += 1
        else:
            n_rows += 1
    # Prefer a slightly flatter grid when both shapes fit.
    while n_cols > 1 and (n_cols - 1) * n_rows >= n_patches:
        alt_cols = n_cols - 1
        alt_rows = int(math.ceil(n_patches / alt_cols))
        cur_err = abs(n_cols / n_rows - aspect)
        alt_err = abs(alt_cols / max(alt_rows, 1) - aspect)
        if alt_err <= cur_err:
            n_cols, n_rows = alt_cols, alt_rows
        else:
            break

    if n_rows == 1:
        rows = [(r_min + r_max) // 2]
    else:
        rows = [
            int(round(r_min + i * (r_max - r_min) / (n_rows - 1)))
            for i in range(n_rows)
        ]
    if n_cols == 1:
        base_cols = [(c_min + c_max) // 2]
        col_step = 0.0
    else:
        col_step = (c_max - c_min) / (n_cols - 1)
        base_cols = [
            int(round(c_min + j * col_step)) for j in range(n_cols)
        ]

    raw: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        stagger = int(round(col_step * 0.5)) if (i % 2 == 1 and n_cols > 1) else 0
        cols = [
            min(c_max, max(c_min, c + stagger)) for c in base_cols
        ]
        # Drop the last cell on staggered rows when that would overflow count
        # later — collect all then trim with maximin if needed.
        for c in cols:
            raw.append((r, c))

    # Keep only n_patches: walk row-major in a space-filling order that
    # spreads leftovers (omit every (total/excess)-th lattice site).
    if len(raw) > n_patches:
        keep = set(
            int(round(i * (len(raw) - 1) / (n_patches - 1)))
            if n_patches > 1
            else 0
            for i in range(n_patches)
        )
        # If rounding collided, fall back to maximin on the lattice.
        if len(keep) < n_patches:
            raw = _maximin_pick(
                raw, n_patches, seed=seed, height=height, width=width
            )
        else:
            raw = [raw[i] for i in sorted(keep)]
    elif len(raw) < n_patches:
        raw.extend(
            _farthest_point_centers(
                height,
                width,
                n_patches - len(raw),
                raw,
                forbidden,
                seed=seed,
            )
        )

    centers: list[tuple[int, int]] = []
    taken: set[tuple[int, int]] = set()
    for r, c in raw:
        snapped = _snap_free(r, c, height, width, forbidden, taken)
        if snapped is None:
            continue
        centers.append(snapped)
        taken.add(snapped)
    if len(centers) < n_patches:
        centers.extend(
            _farthest_point_centers(
                height,
                width,
                n_patches - len(centers),
                centers,
                forbidden,
                seed=seed,
            )
        )
    return centers[:n_patches]


def _farthest_point_with_sep(
    height: int,
    width: int,
    n_patches: int,
    forbidden: set[tuple[int, int]],
    *,
    min_sep: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Maximin centres with a hard Manhattan floor ``min_sep`` between them."""
    sep = max(1, int(min_sep))
    interior = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (r, c) not in forbidden
    ]
    if not interior or n_patches <= 0:
        return []
    cr, cc = height // 2, width // 2
    start = min(
        interior,
        key=lambda rc: (
            abs(rc[0] - cr) + abs(rc[1] - cc),
            hash((rc[0], rc[1], seed)) & 0xFFFF,
        ),
    )
    chosen: list[tuple[int, int]] = [start]
    for i in range(1, n_patches):
        best: tuple[int, int] | None = None
        best_score = (-1.0, 0)
        for r, c in interior:
            md = min(abs(r - er) + abs(c - ec) for er, ec in chosen)
            if md < sep:
                continue
            tie = hash((r, c, seed, i)) & 0xFFFF
            score = (float(md), -tie)
            if score > best_score:
                best_score = score
                best = (r, c)
        if best is None:
            break
        chosen.append(best)
    return chosen


def _adaptive_lattice_centers(
    height: int,
    width: int,
    n_patches: int,
    forbidden: set[tuple[int, int]],
    *,
    seed: int,
) -> tuple[list[tuple[int, int]], int]:
    """Pick ``n_patches`` centres with even packing and map-wide coverage.

    Prefers centre Manhattan ≥ 4 (one empty cell between classic pluses).
    Falls back to 3, then 2, only when more patches are needed. Uses a
    span grid when it already meets the separation target so corners and
    edges are covered (pure maximin from the centre can leave empty bands).
    """
    play = max((height - 2) * (width - 2), 1)
    ideal = max(2, int(math.floor(math.sqrt(play / max(n_patches, 1)))))
    # ≥4 → visible floor between arm tips; ≥3 → arms may touch, not overlap.
    start_sep = max(ideal, 4)

    span = _span_grid_centers(
        height, width, n_patches, forbidden, seed=seed
    )

    def _min_pair(pts: list[tuple[int, int]]) -> int:
        if len(pts) < 2:
            return 10**9
        return min(
            abs(a[0] - b[0]) + abs(a[1] - b[1])
            for i, a in enumerate(pts)
            for b in pts[i + 1 :]
        )

    for sep in range(start_sep, 1, -1):
        if len(span) >= n_patches and _min_pair(span[:n_patches]) >= sep:
            return span[:n_patches], sep
        packed = _farthest_point_with_sep(
            height,
            width,
            n_patches,
            forbidden,
            min_sep=sep,
            seed=seed,
        )
        if len(packed) >= n_patches:
            return packed[:n_patches], sep

    centers = list(span)
    if len(centers) < n_patches:
        centers.extend(
            _farthest_point_centers(
                height,
                width,
                n_patches - len(centers),
                centers,
                forbidden,
                seed=seed,
            )
        )
    return centers[:n_patches], 2


def resolve_apple_layout_mode(
    mode: str | None,
    *,
    num_agents: int,
    density: float,
) -> AppleLayoutMode:
    """Resolve layout mode.

    ``auto`` (default): plus_patch for all agent counts and densities.
    Use ``scaffold`` explicitly for the legacy hand-authored N=4 × 18 apa map.
    """
    raw = "auto" if mode is None or mode == "" else str(mode).lower()
    if raw in ("scaffold", "plus_patch"):
        return raw  # type: ignore[return-value]
    if raw != "auto":
        raise ValueError(
            f"apple_layout_mode must be 'auto'|'scaffold'|'plus_patch', got {mode!r}"
        )
    return "plus_patch"


def scale_apples_plus_patch(
    rows: list[str] | tuple[str, ...],
    target: int,
    *,
    seed: int = 0,
) -> list[str]:
    """Pack as many full classic pluses as possible, efficiently.

    Centres are packed hierarchically at the largest separation first
    (6 → 5 → 4 → 3 → 2), so sparse orchards keep big gaps and tip-touch
    only appears when the map truly cannot fit more patches with a floor
    gap. Pairwise overlap (sep ≤ 2) is last resort; a cell may sit in at
    most two plus footprints. Sizes maximize full (5-cell) pluses;
    remainder 1 becomes ``4 + 2`` (no planned singles).
    """
    target = int(target)
    if target <= 0:
        raise ValueError(f"target apple count must be > 0, got {target}")

    width = max(len(r) for r in rows)
    source = [list(r.ljust(width)) for r in rows]
    height = len(source)
    forbidden = _spawn_forbidden(source, pad=1)

    sizes = _plan_plus_sizes(target)
    n_patches = len(sizes)

    def _min_pair(pts: list[tuple[int, int]]) -> int:
        if len(pts) < 2:
            return 10**9
        return min(
            abs(a[0] - b[0]) + abs(a[1] - b[1])
            for i, a in enumerate(pts)
            for b in pts[i + 1 :]
        )

    def _add_farthest(centers: list[tuple[int, int]], sep: int) -> bool:
        """Append the farthest free cell with Manhattan ≥ ``sep`` to all centres."""
        best = None
        best_score = (-1.0, 0)
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if (r, c) in forbidden:
                    continue
                if centers:
                    md = min(abs(r - er) + abs(c - ec) for er, ec in centers)
                else:
                    # First centre: prefer map middle.
                    md = 1000 - (abs(r - height // 2) + abs(c - width // 2))
                if centers and md < sep:
                    continue
                tie = hash((r, c, seed, sep, len(centers))) & 0xFFFF
                score = (float(md), -tie)
                if score > best_score:
                    best_score = score
                    best = (r, c)
        if best is None:
            return False
        centers.append(best)
        return True

    # Hierarchical pack: fill at the largest separation first so sparse
    # orchards stay open; only tighten when more patches are required.
    centers: list[tuple[int, int]] = []
    for sep in (6, 5, 4, 3, 2):
        while len(centers) < n_patches:
            if not _add_farthest(centers, sep):
                break
        if len(centers) >= n_patches:
            break
    if len(centers) < n_patches:
        packed = _max_centers_at_sep(
            height, width, forbidden, min_sep=2, seed=seed
        )
        centers = packed
        cap = _FULL_PLUS_SIZE * len(centers)
        sizes = _plan_plus_sizes(min(target, cap))
        if len(sizes) > len(centers):
            sizes = sizes[: len(centers)]
            while sum(sizes) < min(target, cap):
                grew = False
                for i in range(len(sizes)):
                    if sizes[i] < _FULL_PLUS_SIZE:
                        sizes[i] += 1
                        grew = True
                        if sum(sizes) >= min(target, cap):
                            break
                if not grew:
                    break
        centers = centers[: len(sizes)]
    else:
        centers = centers[:n_patches]

    min_sep = max(2, min(_min_pair(centers), 6)) if centers else 2

    # Placement visit order: maximin so we don't fill one corner first.
    if len(centers) > 1:
        place_order_centers = _maximin_pick(
            centers,
            len(centers),
            seed=seed,
            height=height,
            width=width,
        )
    else:
        place_order_centers = list(centers)

    grid = [list(row) for row in source]
    for r in range(height):
        for c in range(width):
            if grid[r][c] == "A":
                grid[r][c] = " "

    occupied: set[tuple[int, int]] = set()
    claim_count: dict[tuple[int, int], int] = {}
    owner: dict[tuple[int, int], int] = {}
    placed_centers: list[tuple[int, int]] = []
    patch_id = 0
    allow_overlap = min_sep <= 2

    def sep_ok(center: tuple[int, int]) -> bool:
        cr, cc = center
        return all(
            abs(cr - pr) + abs(cc - pc) >= min_sep
            for pr, pc in placed_centers
        )

    def adjacent_owners(cells: list[tuple[int, int]]) -> set[int]:
        found: set[int] = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nbr = (r + dr, c + dc)
                if nbr in owner:
                    found.add(owner[nbr])
        return found

    def build_cells(
        center: tuple[int, int], size: int
    ) -> list[tuple[int, int]] | None:
        cr, cc = center
        cells: list[tuple[int, int]] = []
        for dr, dc in _PLUS_ARM_ORDER[:size]:
            r, c = cr + dr, cc + dc
            if not (0 <= r < height and 0 <= c < width):
                return None
            if (r, c) in forbidden:
                return None
            if grid[r][c] not in (" ", "A"):
                return None
            if allow_overlap:
                if claim_count.get((r, c), 0) >= 2:
                    return None
            elif (r, c) in occupied:
                return None
            cells.append((r, c))
        if len(adjacent_owners(cells)) > 1:
            return None
        return cells

    def commit(cells: list[tuple[int, int]], center: tuple[int, int]) -> int:
        nonlocal patch_id
        pid = patch_id
        patch_id += 1
        added = 0
        for r, c in cells:
            claim_count[(r, c)] = claim_count.get((r, c), 0) + 1
            if (r, c) not in occupied:
                grid[r][c] = "A"
                occupied.add((r, c))
                added += 1
            owner[(r, c)] = pid
        placed_centers.append(center)
        return added

    def farthest_candidates() -> list[tuple[int, int]]:
        cands = [
            (r, c)
            for r in range(1, height - 1)
            for c in range(1, width - 1)
            if (r, c) not in forbidden
        ]
        if placed_centers:
            cands.sort(
                key=lambda rc: (
                    min(
                        abs(rc[0] - pr) + abs(rc[1] - pc)
                        for pr, pc in placed_centers
                    ),
                    -hash((rc[0], rc[1], seed)) & 0xFFFF,
                ),
                reverse=True,
            )
        else:
            cands.sort(key=lambda rc: hash((rc[0], rc[1], seed)) & 0xFFFF)
        return cands

    # Match sizes to centres: full pluses first, but visit centres spread-out.
    size_queue = sorted(sizes, reverse=True)
    reserved = list(place_order_centers)

    for size in size_queue:
        placed = False
        for trial_size in [size] + list(range(size - 1, 1, -1)):
            for center in list(reserved) + farthest_candidates():
                if not sep_ok(center):
                    continue
                cells = build_cells(center, trial_size)
                if cells is None:
                    continue
                commit(cells, center)
                if center in reserved:
                    reserved.remove(center)
                placed = True
                break
            if placed:
                break

    need = target - sum(1 for row in grid for ch in row if ch == "A")
    guard = 0
    while need > 0 and guard < target + 5:
        guard += 1
        trial_sizes = (
            list(range(min(_FULL_PLUS_SIZE, max(need, 2)), 1, -1))
            if need >= 2
            else [2, 1]
        )
        placed_any = False
        for trial_size in trial_sizes:
            for center in farthest_candidates():
                if not sep_ok(center):
                    continue
                cells = build_cells(center, trial_size)
                if cells is None:
                    continue
                added = commit(cells, center)
                if added <= 0:
                    continue
                need = target - sum(
                    1 for row in grid for ch in row if ch == "A"
                )
                placed_any = True
                break
            if placed_any:
                break
        if not placed_any:
            free = [
                (r, c)
                for r in range(height)
                for c in range(width)
                if grid[r][c] == " " and (r, c) not in forbidden
            ]

            def attach_score(cell: tuple[int, int]) -> tuple:
                r, c = cell
                nbr_owners: set[int] = set()
                tip = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nbr = (r + dr, c + dc)
                    if nbr in owner:
                        nbr_owners.add(owner[nbr])
                        tip += 1
                if len(nbr_owners) == 1 and tip == 1:
                    return (2, tip, -hash(cell) & 0xFFFF)
                if len(nbr_owners) == 1:
                    return (1, tip, -hash(cell) & 0xFFFF)
                return (0, 0, -hash(cell) & 0xFFFF)

            free.sort(key=attach_score, reverse=True)
            for r, c in free:
                if need <= 0:
                    break
                nbr_owners = {
                    owner[(r + dr, c + dc)]
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if (r + dr, c + dc) in owner
                }
                if len(nbr_owners) > 1:
                    continue
                grid[r][c] = "A"
                occupied.add((r, c))
                if nbr_owners:
                    owner[(r, c)] = next(iter(nbr_owners))
                need -= 1
            break

    n_now = sum(1 for row in grid for ch in row if ch == "A")
    if n_now > target:
        apples = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if grid[r][c] == "A"
        ]
        mid_r, mid_c = height // 2, width // 2
        apples.sort(
            key=lambda rc: (
                abs(rc[0] - mid_r) + abs(rc[1] - mid_c),
                hash(rc) & 0xFFFF,
            ),
            reverse=True,
        )
        for r, c in apples[: n_now - target]:
            grid[r][c] = " "

    return ["".join(row) for row in grid]



def scale_apples_to_count(
    rows: list[str] | tuple[str, ...],
    target: int,
    *,
    seed: int = 0,
) -> list[str]:
    """Thin or densify the orchard to exactly ``target`` ``A`` cells.

    Thinning uses a spatially stratified subsample. Densifying fills empty cells
    next to existing apples first (holes / patch edges), never overwriting
    ``P`` / ``Q`` / ``W``. Deterministic for a given ``seed``.
    """
    target = int(target)
    if target <= 0:
        raise ValueError(f"target apple count must be > 0, got {target}")

    width = max(len(r) for r in rows)
    grid = [list(r.ljust(width)) for r in rows]
    h = len(grid)

    apples = [(r, c) for r in range(h) for c in range(width) if grid[r][c] == "A"]
    n0 = len(apples)
    if n0 == 0:
        raise ValueError("cannot scale apples: map has no A cells to seed from")

    free = [
        (r, c)
        for r in range(h)
        for c in range(width)
        if grid[r][c] == " "
    ]
    max_n = n0 + len(free)
    target = max(1, min(target, max_n))

    if target == n0:
        return ["".join(row) for row in grid]

    if target < n0:
        ordered = sorted(apples)
        offset = int(seed) % n0
        rotated = ordered[offset:] + ordered[:offset]
        keep_idx = {
            int(round(i))
            for i in onp.linspace(0, n0 - 1, num=target)
        }
        keep = {rotated[i] for i in keep_idx}
        if len(keep) < target:
            for cell in rotated:
                if cell not in keep:
                    keep.add(cell)
                if len(keep) >= target:
                    break
        elif len(keep) > target:
            keep = set(sorted(keep)[:target])
        for r, c in apples:
            if (r, c) not in keep:
                grid[r][c] = " "
        return ["".join(row) for row in grid]

    need = target - n0
    free_set = set(free)

    def score(cell: tuple[int, int]) -> tuple:
        r, c = cell
        nbr = _apple_neighbor_count(grid, r, c)
        hsh = hash((r, c, int(seed))) & 0xFFFFFFFF
        return (nbr, -hsh, -r, -c)

    placed = 0
    while placed < need and free_set:
        frontier = [cell for cell in free_set if _apple_neighbor_count(grid, *cell) > 0]
        pool = frontier if frontier else list(free_set)
        best = max(pool, key=score)
        r, c = best
        grid[r][c] = "A"
        free_set.remove(best)
        placed += 1

    return ["".join(row) for row in grid]


def scale_apple_density(
    rows: list[str] | tuple[str, ...],
    density: float,
    *,
    num_agents: int,
    apples_per_agent: float = APPLES_PER_AGENT,
    seed: int = APPLE_LAYOUT_SEED,
    layout_mode: str = "auto",
) -> list[str]:
    """Scale orchard to ``density × apples_per_agent × num_agents`` apples.

    ``layout_mode``:

    - ``auto`` (default): plus_patch for all configs; use ``scaffold`` explicitly
      for the legacy hand-authored N=4 × 18 apa orchard (72 apples).
    - ``plus_patch``: maximize full classic pluses; pack with tip-touch /
      pairwise overlap as needed; avoid planned single-apple leftovers.
    - ``scaffold``: thin/densify the hand-authored ASCII orchard in place.
    """
    density = float(density)
    if density <= 0:
        raise ValueError(f"apple_density must be > 0, got {density}")
    resolved = resolve_apple_layout_mode(
        layout_mode, num_agents=num_agents, density=density
    )
    target = target_apple_count(
        num_agents, density, apples_per_agent=apples_per_agent
    )
    if resolved == "plus_patch":
        return scale_apples_plus_patch(rows, target, seed=seed)
    return scale_apples_to_count(rows, target, seed=seed)


def count_apples(rows: list[str] | tuple[str, ...]) -> int:
    return sum(row.count("A") for row in rows)


def paper_env_kwargs() -> dict[str, Any]:
    """Constructor kwargs for the paper-scale Harvest Open env."""
    return {
        "num_agents": PAPER_NUM_AGENTS,
        "num_inner_steps": PAPER_NUM_INNER_STEPS,
        "grid_size": PAPER_GRID_SIZE,
        "map_ASCII": list(PAPER_MAP_ASCII),
        "tag_timeout": PAPER_TAG_TIMEOUT,
        "obs_forward": PAPER_OBS_FORWARD,
        "obs_backward": PAPER_OBS_BACKWARD,
        "obs_side": PAPER_OBS_SIDE,
        "beam_length": PAPER_BEAM_LENGTH,
        "beam_width": PAPER_BEAM_WIDTH,
    }


def paper_perolat_env_kwargs() -> dict[str, Any]:
    """Constructor kwargs for the exact Perolat foundation-layout Harvest env."""
    return {
        "num_agents": PAPER_PEROLAT_NUM_AGENTS,
        "num_inner_steps": PAPER_PEROLAT_NUM_INNER_STEPS,
        "grid_size": PAPER_PEROLAT_GRID_SIZE,
        "map_ASCII": list(PAPER_PEROLAT_MAP_ASCII),
        "tag_timeout": PAPER_PEROLAT_TAG_TIMEOUT,
        "obs_forward": PAPER_PEROLAT_OBS_FORWARD,
        "obs_backward": PAPER_PEROLAT_OBS_BACKWARD,
        "obs_side": PAPER_PEROLAT_OBS_SIDE,
        "beam_length": PAPER_PEROLAT_BEAM_LENGTH,
        "beam_width": PAPER_PEROLAT_BEAM_WIDTH,
    }


def test_env_kwargs() -> dict[str, Any]:
    """Constructor kwargs for the scaled-down test Harvest Open env."""
    return {
        "num_agents": TEST_NUM_AGENTS,
        "num_inner_steps": TEST_NUM_INNER_STEPS,
        "grid_size": TEST_GRID_SIZE,
        "map_ASCII": list(TEST_MAP_ASCII),
        "tag_timeout": TEST_TAG_TIMEOUT,
        "obs_forward": TEST_OBS_FORWARD,
        "obs_backward": TEST_OBS_BACKWARD,
        "obs_side": TEST_OBS_SIDE,
        "beam_length": TEST_BEAM_LENGTH,
        "beam_width": TEST_BEAM_WIDTH,
    }


def tiny_env_kwargs() -> dict[str, Any]:
    """Constructor kwargs for the half-of-test Harvest Open env."""
    return {
        "num_agents": TINY_NUM_AGENTS,
        "num_inner_steps": TINY_NUM_INNER_STEPS,
        "grid_size": TINY_GRID_SIZE,
        "map_ASCII": list(TINY_MAP_ASCII),
        "tag_timeout": TINY_TAG_TIMEOUT,
        "obs_forward": TINY_OBS_FORWARD,
        "obs_backward": TINY_OBS_BACKWARD,
        "obs_side": TINY_OBS_SIDE,
        "beam_length": TINY_BEAM_LENGTH,
        "beam_width": TINY_BEAM_WIDTH,
    }


def env_kwargs_for_scale(scale: EnvScale) -> dict[str, Any]:
    if scale == "paper":
        return paper_env_kwargs()
    if scale == "paper_perolat":
        return paper_perolat_env_kwargs()
    if scale == "test":
        return test_env_kwargs()
    if scale == "tiny":
        return tiny_env_kwargs()
    raise ValueError(
        f"Unknown ENV_SCALE {scale!r}; expected 'paper', 'paper_perolat', 'test', or 'tiny'"
    )


def apply_env_scale(config: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """If ``config['ENV_SCALE']`` is set, overlay geometry into ``ENV_KWARGS``.

    Preset values win for geometry keys so every paper/test/tiny run stays aligned.
    Other ENV_KWARGS (cnn, jit, shared_rewards, ...) are kept.
    """
    scale = config.get("ENV_SCALE")
    if scale is None or scale == "":
        print(
            "[env_scale] WARNING: ENV_SCALE is not set. "
            "Using ENV_KWARGS / constructor defaults (legacy 16x22 unless overridden). "
            "Set ENV_SCALE: paper|paper_perolat|test|tiny in the YAML for reproducible layouts."
        )
        return config
    if scale not in ("paper", "paper_perolat", "test", "tiny"):
        raise ValueError(
            f"Unknown ENV_SCALE {scale!r}; expected 'paper', 'paper_perolat', 'test', or 'tiny'"
        )

    preset = env_kwargs_for_scale(scale)
    env_kwargs = dict(config.get("ENV_KWARGS") or {})
    # Preserve explicit ENV_KWARGS geometry (Hydra CLI / yaml) over preset defaults.
    explicit_geom = {
        k: env_kwargs[k] for k in _GEOMETRY_KEYS if k in env_kwargs
    }
    for key in _GEOMETRY_KEYS:
        env_kwargs[key] = preset[key]
    for key, val in explicit_geom.items():
        env_kwargs[key] = val

    # Optional agent-count override (keeps map+beam+obs, raises pressure).
    # Example YAML: ENV_SCALE: test + ENV_NUM_AGENTS: 6
    n_override = config.get("ENV_NUM_AGENTS")
    if n_override is not None and n_override != "":
        env_kwargs["num_agents"] = int(n_override)
        print(
            f"[env_scale] ENV_NUM_AGENTS override -> num_agents={env_kwargs['num_agents']} "
            f"(preset was {preset['num_agents']})"
        )

    # Optional orchard-density override (same role as ENV_KWARGS.apple_density).
    # Example YAML: ENV_SCALE: paper + ENV_APPLE_DENSITY: 0.5
    dens_override = config.get("ENV_APPLE_DENSITY")
    if dens_override is not None and dens_override != "":
        env_kwargs["apple_density"] = float(dens_override)
        print(
            f"[env_scale] ENV_APPLE_DENSITY override -> apple_density="
            f"{env_kwargs['apple_density']}"
        )

    config["ENV_KWARGS"] = env_kwargs

    print(
        f"[env_scale] ENV_SCALE={scale} -> "
        f"grid={preset['grid_size']} agents={env_kwargs['num_agents']} "
        f"obs=({preset['obs_forward']}/{preset['obs_backward']}/{preset['obs_side']}) "
        f"beam={preset['beam_length']}x{preset['beam_width']}"
        + (
            f" apple_density={env_kwargs['apple_density']}"
            if "apple_density" in env_kwargs
            else ""
        )
    )
    return config


def print_run_env_banner(config: Mapping[str, Any], env: Any) -> None:
    """Print the live environment factors at the start of a training run.

    Prefer values read from the constructed ``env`` so the log matches what
    actually runs (after ``ENV_SCALE`` overlay), not stale YAML leftovers.
    """
    ekw = config.get("ENV_KWARGS") or {}
    scale = config.get("ENV_SCALE", "(unset)")
    grid = (
        getattr(env, "GRID_SIZE_ROW", None),
        getattr(env, "GRID_SIZE_COL", None),
    )
    n_agents = getattr(env, "num_agents", ekw.get("num_agents"))
    obs = (
        getattr(env, "OBS_FORWARD", ekw.get("obs_forward")),
        getattr(env, "OBS_BACKWARD", ekw.get("obs_backward")),
        getattr(env, "OBS_SIDE", ekw.get("obs_side")),
    )
    obs_shape = (
        getattr(env, "OBS_H", None),
        getattr(env, "OBS_W", None),
    )
    beam = (
        getattr(env, "beam_length", ekw.get("beam_length")),
        getattr(env, "beam_width", ekw.get("beam_width")),
    )
    apples = getattr(getattr(env, "SPAWNS_APPLE", None), "shape", (None,))[0]
    spawn_p = getattr(getattr(env, "SPAWNS_PLAYERS", None), "shape", (None,))[0]
    spawn_q = getattr(getattr(env, "SPAWNS_PLAYER_IN", None), "shape", (None,))[0]
    tag_timeout = getattr(env, "tag_timeout", ekw.get("tag_timeout"))
    inner = ekw.get("num_inner_steps", getattr(env, "num_inner_steps", "?"))
    total = config.get("TOTAL_TIMESTEPS", "?")
    seed = config.get("SEED", "?")

    expected = None
    if scale == "paper":
        expected = PAPER_GRID_SIZE
    elif scale == "paper_perolat":
        expected = PAPER_PEROLAT_GRID_SIZE
    elif scale == "test":
        expected = TEST_GRID_SIZE
    elif scale == "tiny":
        expected = TINY_GRID_SIZE

    print("=" * 64)
    print("[run] Environment check")
    print(f"  ENV_SCALE          : {scale}")
    print(f"  ENV_NAME           : {config.get('ENV_NAME', '?')}")
    print(
        f"  shared_rewards     : "
        f"{getattr(env, 'shared_rewards', ekw.get('shared_rewards', '?'))}"
    )
    print(f"  num_agents         : {n_agents}")
    print(f"  grid_size (H x W)  : {grid[0]} x {grid[1]}")
    print(f"  apples / P / Q     : {apples} / {spawn_p} / {spawn_q}")
    print(f"  apple_density      : {ekw.get('apple_density', getattr(env, 'apple_density', 1.0))}")
    print(
        f"  apple_layout_mode  : "
        f"{ekw.get('apple_layout_mode', getattr(env, 'apple_layout_mode', 'auto'))}"
    )
    print(f"  obs fwd/back/side  : {obs[0]}/{obs[1]}/{obs[2]}  -> window {obs_shape[0]} x {obs_shape[1]}")
    print(f"  beam length x width: {beam[0]} x {beam[1]}")
    print(f"  tag_timeout        : {tag_timeout}")
    print(f"  regrowth_scale     : {ekw.get('regrowth_scale', getattr(env, 'regrowth_scale', 1.0))}")
    print(
        f"  terminate_on_barren: "
        f"{ekw.get('terminate_on_barren', getattr(env, 'terminate_on_barren', False))}"
    )
    print(f"  num_inner_steps T  : {inner}")
    print(f"  TOTAL_TIMESTEPS   : {total}")
    print(f"  SEED               : {seed}")
    if expected is not None and grid != expected:
        print(
            f"  WARNING: live grid {grid} does not match ENV_SCALE={scale} "
            f"expected {expected}"
        )
    elif expected is not None:
        print(f"  layout check       : OK (matches {scale} preset {expected})")
    print("=" * 64)


def validate_maps() -> Mapping[str, Any]:
    """Cheap host-side checks used by unit/smoke tests."""
    out: dict[str, Any] = {}
    for name, rows, grid, n_agents in (
        ("paper", PAPER_MAP_ASCII, PAPER_GRID_SIZE, PAPER_NUM_AGENTS),
        (
            "perolat_fig6b",
            PEROLAT_FIG6B_OPEN_MAP_ASCII,
            PEROLAT_FIG6B_GRID_SIZE,
            PAPER_NUM_AGENTS,
        ),
        (
            "paper_perolat",
            PAPER_PEROLAT_MAP_ASCII,
            PAPER_PEROLAT_GRID_SIZE,
            PAPER_PEROLAT_NUM_AGENTS,
        ),
        ("test", TEST_MAP_ASCII, TEST_GRID_SIZE, TEST_NUM_AGENTS),
        ("tiny", TINY_MAP_ASCII, TINY_GRID_SIZE, TINY_NUM_AGENTS),
    ):
        h, w = grid
        assert len(rows) == h, f"{name}: expected {h} rows, got {len(rows)}"
        assert all(len(r) == w for r in rows), f"{name}: ragged row widths"
        joined = "".join(rows)
        out[name] = {
            "apples": joined.count("A"),
            "spawn_P": joined.count("P"),
            "spawn_Q": joined.count("Q"),
            "density": joined.count("A") / (h * w),
        }
        assert out[name]["spawn_Q"] >= 2, f"{name}: need >=2 Q for reset"
        assert out[name]["spawn_P"] + out[name]["spawn_Q"] >= n_agents, (
            f"{name}: need >=N spawn cells"
        )
    assert out["tiny"]["apples"] > 0, "tiny: expected apples on hand-placed layout"
    # Tiny keeps half of test in linear dims / agent count.
    assert TINY_GRID_SIZE == (TEST_GRID_SIZE[0] // 2, TEST_GRID_SIZE[1] // 2)
    assert TINY_NUM_AGENTS == TEST_NUM_AGENTS // 2
    return out
