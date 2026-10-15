from enum import IntEnum
import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass
import colorsys

from socialjax.environments.multi_agent_env import MultiAgentEnv
from socialjax.environments import spaces


from socialjax.environments.common_harvest.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)
from harvest_socialjax.envs import beam_geometry
from harvest_socialjax.envs.bandits import (
    BANDIT_REP,
    apply_bandit_zap_respawn,
    bandit_config_from_kwargs,
    bandit_occupancy_grid,
    beam_hit_bandits,
    clear_bandits_from_grid,
    extended_zap_labels_and_hits,
    init_bandits_on_reset,
    move_bandits,
    paint_bandits_on_grid,
    resolve_bandit_respawn_delay,
)
from harvest_socialjax.envs.env_scale_presets import (
    count_apples,
    resolve_apple_layout_mode,
    APPLE_LAYOUT_SEED,
    scale_apple_density,
)
from harvest_socialjax.envs.reputation import (
    clamp_enforcement_strictness,
    sample_reputation_enforcement,
)
from harvest_socialjax.envs.police.mechanics import (
    apply_peer_focus_on_bad_zap_hits,
    apply_police_focus_on_bad_zap_hits,
    mask_zap_hits_on_police,
    normalize_police_agent_ids,
    police_agent_mask,
    police_binary_zap_reward,
    split_beam_victims,
)

NUM_TYPES = 4  # empty (0), red (1), blue, red coin, blue coin, wall, interact
NUM_COIN_TYPES = 1
INTERACT_THRESHOLD = 0
TAG_TIMEOUT_STEPS = 25  # Perolat et al. (2017): tagged agents removed for 25 steps
# Spatial reputation planes appended in ``combine_channels`` (after agent features).
_REP_OBS_CHANNELS = 2  # ch 15=self_reputation, 16=other_reputation (baseline layout)

# Perolat / Melting Pot apple regrowth neighborhood: L2 disk radius 2 (12 cells).
# Same offsets as Harvest_open.count_apple — used for local-last detection.
_L2_APPLE_NEIGHBOR_OFFSETS = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def l2_disk_live_apple_count(
    apple_map: jnp.ndarray,
    r: jnp.ndarray,
    c: jnp.ndarray,
    n_rows: int,
    n_cols: int,
) -> jnp.ndarray:
    """Count live apples in the L2≤2 disk around ``(r, c)``, excluding the center.

    Matches Perolat regrowth neighbor stock (``Harvest_open.count_apple``).
    """
    total = jnp.int32(0)
    for dr, dc in _L2_APPLE_NEIGHBOR_OFFSETS:
        rr = r + dr
        cc = c + dc
        in_bounds = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
        rr_safe = jnp.clip(rr, 0, n_rows - 1)
        cc_safe = jnp.clip(cc, 0, n_cols - 1)
        live = apple_map[rr_safe, cc_safe] == 1
        total = total + jnp.where(in_bounds & live, jnp.int32(1), jnp.int32(0))
    return total


@dataclass
class State:
    agent_locs: jnp.ndarray
    agent_invs: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    # Live-apple presence layer (H, W): 1 if an apple occupies the cell, including
    # under a standing agent (Melting Pot ``canRegrowIfOccupied``). Grid sprites
    # show agents on top; this layer drives regrowth neighbor counts + eating.
    apples: jnp.ndarray

    freeze: jnp.ndarray
    reborn_locs: jnp.ndarray
    # Per-agent tag time-out counter (Perolat et al. 2017): 0 = active/in-play,
    # >0 = number of remaining steps the agent is removed from the game.
    time_out: jnp.ndarray
    # Timeout cause flags (persist while time_out > 0; cleared on respawn).
    # Agent beam / police beam / reputation-enforcement timeouts are tracked
    # separately so peace / police_peace / system_peace do not mix causes.
    timeout_from_agent: jnp.ndarray
    timeout_from_enforcement: jnp.ndarray
    timeout_from_police: jnp.ndarray
    # Pre-step reputation for spatial ch15/ch16 (set by ReputationWrapper).
    tag_filter_rep: jnp.ndarray
    # --- Bandits (env-side only; not policy slots) ---
    bandit_locs: jnp.ndarray  # (B, 3) row, col, facing
    bandit_active: jnp.ndarray  # (B,) on-grid and zappable
    bandit_respawn_cd: jnp.ndarray  # (B,) steps until respawn
    bandit_visual_id: jnp.ndarray  # (B,) index into PLAYER_COLOURS / agent grid id
    # Completed inner episodes (for bandit count anneal); shared across envs via
    # schedule_env_multiplier when set on BanditConfig.
    episodes_completed: jnp.ndarray  # scalar int32

@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int

class Actions(IntEnum):
    turn_left = 0
    turn_right = 1
    left = 2
    right = 3
    up = 4
    down = 5
    stay = 6
    zap_forward = 7

class Items(IntEnum):
    empty = 0
    wall = 1
    interact = 2
    apple = 3
    spawn_point = 4
    inside_spawn_point = 5
    
char_to_int = {
    'W': 1,
    ' ': 0,  # 空格字符映射为 0
    'A': 3,
    'P': 4,
    'Q': 5
}

ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # left
        [0, 0, 0],  # right
        [0, 0, 0],  # up
        [0, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [1, 0, 0],  # up
        [0, 1, 0],  # right
        [-1, 0, 0],  # down
        [0, -1, 0],  # left
    ],
    dtype=jnp.int8,
)

STEP_MOVE = jnp.array(
    [
        [0, 0, 0],  # turn left
        [0, 0, 0],  # turn right
        [0, 1, 0],  # left
        [0, -1, 0],  # right
        [1, 0, 0],  # up
        [-1, 0, 0],  # down
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap
    ],
    dtype=jnp.int8,
)


def ascii_map_to_matrix(map_ASCII, char_to_int):
    """
    Convert ASCII map to a JAX numpy matrix using the given character mapping.
    
    Args:
    map_ASCII (list): List of strings representing the ASCII map
    char_to_int (dict): Dictionary mapping characters to integer values
    
    Returns:
    jax.numpy.ndarray: 2D matrix representation of the ASCII map
    """
    # Determine matrix dimensions
    height = len(map_ASCII)
    width = max(len(row) for row in map_ASCII)
    
    # Create matrix filled with zeros
    matrix = jnp.zeros((height, width), dtype=jnp.int32)
    
    # Fill matrix with mapped values
    for i, row in enumerate(map_ASCII):
        for j, char in enumerate(row):
            matrix = matrix.at[i, j].set(char_to_int.get(char, 0))
    
    return matrix

def generate_agent_colors(num_agents):
    colors = []
    for i in range(num_agents):
        hue = i / num_agents
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)  # Saturation and Value set to 0.8
        colors.append(tuple(int(x * 255) for x in rgb))
    return colors

GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)
###################################################

class Harvest_open(MultiAgentEnv):
    """
    JAX Compatible n-agent version of *inTheMatix environment.

    ``jit=True`` (default) wraps ``step_env`` / ``reset`` in ``jax.jit`` for GPU
    throughput. Use ``jit=False`` only when debugging and you need to inspect
    intermediate values without compilation.
    """

    # used for caching
    tile_cache: Dict[Tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps=1000,
        num_outer_steps=1,
        num_agents=7,
        shared_rewards=True,
        inequity_aversion=False,
        inequity_aversion_target_agents=None,
        inequity_aversion_alpha=5,
        inequity_aversion_beta=0.05,
        enable_smooth_rewards=False,
        svo=False,
        svo_target_agents=None,
        svo_w=0.5,
        svo_ideal_angle_degrees=45,
        interest=False,
        s_interest=0.5,
        s_interest_schedule=None,
        s_interest_change_every=30000000,
        grid_size=(16,22),
        jit=True,
        obs_size=11,
        obs_forward=20,
        obs_backward=1,
        obs_side=10,
        cnn=True,
        map_ASCII = [
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
            ],
        tag_timeout=TAG_TIMEOUT_STEPS,
        beam_length: int = beam_geometry.BEAM_LENGTH,
        beam_width: int = beam_geometry.BEAM_WIDTH,
        # Reputation enforcement: each step, active agents with pre-step
        # ``tag_filter_rep < 0`` are timed out with
        # ``P = clip(-strictness * rep, 0, 1)`` (same ``tag_timeout`` /
        # respawn path as agent-beam tags). Off unless reputation training
        # injects these kwargs (see ``ENABLE_REPUTATION``).
        reputation_enforcement: bool = False,
        reputation_enforcement_strictness: float = 0.01,
        # Multiplier on Perolat local-stock regrowth probs (1.0 = paper).
        # <1 makes the commons scarcer / tragedy easier without raising N.
        regrowth_scale: float = 1.0,
        # Scale initial orchard by apples-per-agent ratio (default 18/agent).
        # apple_density=1 → 18×N apples; 0.5 → 9×N; 2 → 36×N.
        apple_density: float = 1.0,
        # How density maps to spatial layout (see env_scale_presets).
        # auto → plus_patch (see env_scale_presets.resolve_apple_layout_mode).
        apple_layout_mode: str = "auto",
        # If True, end the inner episode as soon as live apple stock hits 0.
        # Safe when L=0 is irreversible (Perolat): no further apple reward is
        # possible. Speeds ToC runs by starting a fresh episode instead of
        # simulating barren padding to T. Off by default (paper-faithful fixed T).
        terminate_on_barren: bool = False,
        # Optional police agents (no harvest; immune to timeout; own zap reward).
        police_agent_ids: Optional[Union[Sequence[int], int]] = None,
        police_beam_length: int | None = None,
        police_beam_width: int | None = None,
        police_timeout_duration: int | None = None,
        police_focus_on_bad: bool = False,
        # Peer/harvester zaps only freeze Bad (label < 0) when True.
        peer_focus_on_bad: bool = False,
        # If True, paint ch15/ch16 as zeros in obs while ``tag_filter_rep`` /
        # labels still drive timeouts, peer-punish, and metrics.
        mask_reputation_obs: bool = False,
        # --- Bandits (tunable ENV_KWARGS; see bandits.BanditConfig) ---
        bandit_count: int = 0,
        bandit_respawn_delay: int = 0,
        bandit_respawn_delay_anneal: bool = False,
        bandit_respawn_delay_start: int = -1,
        bandit_respawn_delay_end: int = 0,
        bandit_respawn_delay_anneal_episodes: int = 50000,
        bandit_count_anneal: bool = False,
        bandit_anneal_mode: str = "linear",
        bandit_count_start: int = -1,
        bandit_count_end: int = 0,
        bandit_count_anneal_episodes: int = 20000,
        bandit_step_counts: str | list | tuple = "",
        bandit_step_episodes: str | list | tuple = "",
        bandit_schedule_env_multiplier: int = 1,
    ):
        super().__init__(num_agents=num_agents)
        self.agents = list(range(num_agents))#, dtype=jnp.int16)
        self._agents = jnp.array(self.agents, dtype=jnp.int32) + len(Items)
        self.shared_rewards = shared_rewards
        self.cnn = cnn
        self.regrowth_scale = float(regrowth_scale)
        self.apple_density = float(apple_density)
        self.apple_layout_mode = resolve_apple_layout_mode(
            apple_layout_mode,
            num_agents=self.num_agents,
            density=self.apple_density,
        )
        self.terminate_on_barren = bool(terminate_on_barren)
        self.inequity_aversion = inequity_aversion
        self.inequity_aversion_target_agents = inequity_aversion_target_agents
        self.inequity_aversion_alpha = inequity_aversion_alpha
        self.inequity_aversion_beta = inequity_aversion_beta
        self.enable_smooth_rewards = enable_smooth_rewards
        self.svo = svo
        self.svo_target_agents = svo_target_agents
        self.svo_w = svo_w
        self.svo_ideal_angle_degrees = svo_ideal_angle_degrees
        self.smooth_rewards = enable_smooth_rewards
        self.interest = interest
        self.s_interest = s_interest
        # Convert schedule to JAX array for JIT compatibility
        if s_interest_schedule is not None:
            self.s_interest_schedule = jnp.array(s_interest_schedule)
        else:
            self.s_interest_schedule = None
        self.s_interest_change_every = s_interest_change_every

        # Loud, host-side notice: the inequity-aversion / SVO / interest reward
        # shaping branches still scale the apple reward by ``num_agents`` and are
        # therefore NOT aligned with the Perolat/Lupu paper (+1 per apple). The
        # plain individual-reward path used by the paper-faithful DQN/IPPO harvest
        # configs is unaffected. This prints once at env construction so it shows
        # up in any run log that enables one of these modes.
        _nonpaper_modes = [
            name
            for name, on in (
                ("inequity_aversion", inequity_aversion),
                ("svo", svo),
                ("interest", interest),
            )
            if on
        ]
        if _nonpaper_modes:
            print(
                "[harvest_open][REWARD WARNING] Non-paper reward shaping "
                f"active: {', '.join(_nonpaper_modes)}. These branches scale the "
                "apple reward by num_agents (NOT +1/apple as in Perolat et al. "
                "2017 / Lupu & Precup 2020). Reported returns are not directly "
                "comparable to the paper.",
                flush=True,
            )

        # self.agents = [str(i) for i in list(range(num_agents))]

        self.tag_timeout = int(tag_timeout)
        self.reputation_enforcement = bool(reputation_enforcement)
        self.reputation_enforcement_strictness = clamp_enforcement_strictness(
            reputation_enforcement_strictness
        )
        if int(beam_width) % 2 == 0:
            raise ValueError(
                f"beam_width must be odd so the beam is centered (got {beam_width})"
            )
        self.beam_length = int(beam_length)
        self.beam_width = int(beam_width)

        _police_ids = normalize_police_agent_ids(
            police_agent_ids, num_agents=self.num_agents
        )
        self._police_agent_ids = _police_ids
        self._police_mask = police_agent_mask(self.num_agents, _police_ids)
        # Public alias for trainers / wrappers that unwrap the env stack.
        self.police_mask = self._police_mask
        if _police_ids:
            print(
                f"[harvest_open] police agents (no apple harvest, timeout-immune): "
                f"{list(_police_ids)}",
                flush=True,
            )

        _pbl = (
            int(police_beam_length)
            if police_beam_length is not None
            else self.beam_length
        )
        _pbw = (
            int(police_beam_width)
            if police_beam_width is not None
            else self.beam_width
        )
        if _pbw % 2 == 0:
            raise ValueError(
                f"police_beam_width must be odd so the beam is centered (got {_pbw})"
            )
        self._police_beam_length = _pbl
        self._police_beam_width = _pbw
        self._use_police_beam = (
            self._police_beam_length != self.beam_length
            or self._police_beam_width != self.beam_width
        )
        self._police_timeout = (
            int(police_timeout_duration)
            if police_timeout_duration is not None
            else self.tag_timeout
        )
        self._police_focus_on_bad = bool(police_focus_on_bad)
        self._peer_focus_on_bad = bool(peer_focus_on_bad)
        self._mask_reputation_obs = bool(mask_reputation_obs)
        if self._use_police_beam and _police_ids:
            print(
                f"[harvest_open] police beam "
                f"{self._police_beam_length}x{self._police_beam_width} "
                f"(harvester {self.beam_length}x{self.beam_width})",
                flush=True,
            )
        if self._police_focus_on_bad and _police_ids:
            print(
                "[harvest_open] police_focus_on_bad=true "
                "(police zaps only freeze Bad / label<0 targets)",
                flush=True,
            )
        if self._peer_focus_on_bad:
            print(
                "[harvest_open] peer_focus_on_bad=true "
                "(peer zaps only freeze Bad; beam/rewards still see Good hits)",
                flush=True,
            )
        if self._mask_reputation_obs:
            _n_pol = int(jnp.sum(self._police_mask))
            print(
                "[harvest_open] mask_reputation_obs=true "
                "(ch15/ch16 zeroed for ALL agents"
                + (f" incl. {_n_pol} police" if _n_pol > 0 else "")
                + "; tag_filter_rep still used for mechanics/rewards)",
                flush=True,
            )

        self._bandit_cfg = bandit_config_from_kwargs(
            {
                "bandit_count": bandit_count,
                "bandit_respawn_delay": bandit_respawn_delay,
                "bandit_respawn_delay_anneal": bandit_respawn_delay_anneal,
                "bandit_respawn_delay_start": bandit_respawn_delay_start,
                "bandit_respawn_delay_end": bandit_respawn_delay_end,
                "bandit_respawn_delay_anneal_episodes": (
                    bandit_respawn_delay_anneal_episodes
                ),
                "bandit_count_anneal": bandit_count_anneal,
                "bandit_anneal_mode": bandit_anneal_mode,
                "bandit_count_start": bandit_count_start,
                "bandit_count_end": bandit_count_end,
                "bandit_count_anneal_episodes": bandit_count_anneal_episodes,
                "bandit_step_counts": bandit_step_counts,
                "bandit_step_episodes": bandit_step_episodes,
                "bandit_schedule_env_multiplier": bandit_schedule_env_multiplier,
            }
        )
        self._bandit_count = int(self._bandit_cfg.bandit_count)
        if self._bandit_count > 0:
            print(
                f"[harvest_open] bandits={self._bandit_count} "
                f"respawn_delay={self._bandit_cfg.bandit_respawn_delay} "
                f"delay_anneal={self._bandit_cfg.bandit_respawn_delay_anneal} "
                f"anneal={self._bandit_cfg.bandit_count_anneal} "
                f"mode={self._bandit_cfg.bandit_anneal_mode} "
                f"step_counts={self._bandit_cfg.bandit_step_counts} "
                f"step_eps={self._bandit_cfg.bandit_step_episodes} "
                f"sched_mult={self._bandit_cfg.schedule_env_multiplier}",
                flush=True,
            )

        self.PLAYER_COLOURS = generate_agent_colors(num_agents)
        self.GRID_SIZE_ROW = grid_size[0]
        self.GRID_SIZE_COL = grid_size[1]
        # Egocentric field of view: depth includes the agent cell (no backward
        # rows). A centered square is extracted, rotated to a canonical frame
        # (forward = +row), then cropped to (OBS_H, OBS_W).
        self.OBS_FORWARD = int(obs_forward)
        self.OBS_BACKWARD = int(obs_backward)
        self.OBS_SIDE = int(obs_side)
        self.OBS_RADIUS = max(self.OBS_FORWARD, self.OBS_BACKWARD, self.OBS_SIDE)
        self.OBS_SIZE = 2 * self.OBS_RADIUS + 1  # centered extraction square
        self.OBS_H = self.OBS_FORWARD + self.OBS_BACKWARD + 1
        self.OBS_W = 2 * self.OBS_SIDE + 1
        self.PADDING = self.OBS_RADIUS
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

        self._num_actions = 8
        self._obs_extra_channels = 10 + _REP_OBS_CHANNELS

        self.ROTATIONS = ROTATIONS
        self.STEP_MOVE = STEP_MOVE

        GRID = jnp.zeros(
            (self.GRID_SIZE_ROW + 2 * self.PADDING, self.GRID_SIZE_COL + 2 * self.PADDING),
            dtype=jnp.int16,
        )

        # First layer of padding is Wall
        wall_pad = jnp.int16(Items.inside_spawn_point)
        GRID = GRID.at[self.PADDING - 1, :].set(wall_pad)
        GRID = GRID.at[self.GRID_SIZE_ROW + self.PADDING, :].set(wall_pad)
        GRID = GRID.at[:, self.PADDING - 1].set(wall_pad)
        self.GRID = GRID.at[:, self.GRID_SIZE_COL + self.PADDING].set(wall_pad)

        def find_positions(grid_array, letter):
            a_positions = jnp.array(jnp.where(grid_array == letter)).T
            # Ensure rank-2 (N, 2) even when there are zero matches.
            return jnp.reshape(a_positions, (-1, 2)).astype(jnp.int32)

        # Scale orchard to density × 18 × N (ASCII map is a spatial scaffold).
        map_rows = list(map_ASCII)
        n_map = count_apples(map_rows)
        map_rows = scale_apple_density(
            map_rows,
            self.apple_density,
            num_agents=self.num_agents,
            seed=APPLE_LAYOUT_SEED,
            layout_mode=self.apple_layout_mode,  # type: ignore[arg-type]
        )
        n_scaled = count_apples(map_rows)
        if n_scaled != n_map or abs(self.apple_density - 1.0) > 1e-9:
            per_agent = n_scaled / max(self.num_agents, 1)
            print(
                f"[harvest_open] apple_density={self.apple_density:g} "
                f"layout={self.apple_layout_mode}: "
                f"{n_map} scaffold -> {n_scaled} apples "
                f"({per_agent:.1f}/agent, N={self.num_agents})",
                flush=True,
            )
        self._map_ASCII_effective = map_rows

        nums_map = ascii_map_to_matrix(map_rows, char_to_int)
        self.SPAWNS_APPLE = find_positions(nums_map, 3)
        self.SPAWNS_PLAYER_IN = find_positions(nums_map, 5)
        self.SPAWNS_PLAYERS = find_positions(nums_map, 4)
        self.SPAWNS_WALL = find_positions(nums_map, 1)
        # Host-side spawn sets for RGB overlay. P/Q are parsed into lookup
        # tables and never written onto ``state.grid`` (Melting Pot keeps them
        # on an alternate logic layer), so without this they are invisible.
        self._spawn_p_cells = frozenset(
            (int(r), int(c))
            for r, c in onp.asarray(self.SPAWNS_PLAYERS).reshape(-1, 2)
        )
        self._spawn_q_cells = frozenset(
            (int(r), int(c))
            for r, c in onp.asarray(self.SPAWNS_PLAYER_IN).reshape(-1, 2)
        )
        n_p = int(self.SPAWNS_PLAYERS.shape[0])
        n_q = int(self.SPAWNS_PLAYER_IN.shape[0])
        if n_p + n_q < num_agents:
            raise ValueError(
                f"map has P={n_p} Q={n_q} spawn cells but num_agents={num_agents}"
            )
        if num_agents >= 2 and n_q < 2:
            raise ValueError(
                f"reset for N={num_agents} needs >=2 Q inside-spawns; map has {n_q}"
            )
        if num_agents >= 2 and n_p < num_agents - 2:
            raise ValueError(
                f"reset fills N-2 agents from P; map has P={n_p} for N={num_agents}"
            )
        # Tag-respawn samples this pool. Prefer P; fall back to Q∪P when
        # P < N so permutation[:N] stays rank-2 with N rows.
        if n_p >= num_agents:
            self._RESPAWN_POOL = self.SPAWNS_PLAYERS
        else:
            self._RESPAWN_POOL = jnp.concatenate(
                (self.SPAWNS_PLAYER_IN, self.SPAWNS_PLAYERS), axis=0
            )


        def rand_interaction(
                key: int,
                conflicts: jnp.ndarray,
                conflicts_matrix: jnp.ndarray,
                step_arr: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function for randomly choosing between conflicting interactions.
            
            Args:
                - key: jax PRNGKey for randomisation.
                - conflicts: jnp.ndarray of bools where True if agent is in a
                conflicting interaction, False otherwise.
                - conflicts_matrix: jnp.ndarray matrix of bools of agents in
                conflicting interactions.
                - step_arr: jnp.ndarray, where each index is the index of an
                agent, and the element at each index is the item found at that
                agent's respective target location in the grid.

                
            Returns:
                - jnp.ndarray array of final interactions, where each index is
                an agent, and each element is caught in its interaction beam.
            '''
            def scan_fn(
                    state,
                    idx
            ):

                key, conflicts, conflicts_matrix, step_arr = state

                return jax.lax.cond(
                    conflicts[idx] > 0,
                    lambda: _rand_interaction(
                        key,
                        conflicts,
                        conflicts_matrix,
                        step_arr
                    ),
                    lambda: (state, step_arr.astype(jnp.int16))
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, conflicts, conflicts_matrix, step_arr.astype(jnp.int16)),
                jnp.arange(self.num_agents)
            )

            final_itxs = ys[-1]
            return final_itxs

        def _rand_interaction(
                key: int,
                conflicts: jnp.ndarray,
                conflicts_matrix: jnp.ndarray,
                step_arr: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function for randomly choosing between conflicting interactions.
            
            Args:
                - key: jax PRNGKey for randomisation.
                - conflicts: jnp.ndarray of bools where True if agent is in a
                conflicting interaction, False otherwise.
                - conflicts_matrix: jnp.ndarray matrix of bools of agents in
                conflicting interactions.
                - step_arr: jnp.ndarray, where each index is the index of an
                agent, and the element at each index is the item found at that
                agent's respective target location in the grid.

                
            Returns:
                - jnp.ndarray array of final interactions, where each index is
                an agent, and each element is caught in its interaction beam.
            '''
            conflict_idx = jnp.nonzero(
                conflicts,
                size=self.num_agents,
                fill_value=-1
            )[0][0]

            agent_conflicts = conflicts_matrix[conflict_idx]

            agent_conflicts_idx = jnp.nonzero(
                agent_conflicts,
                size=self.num_agents,
                fill_value=-1
            )[0]
            max_rand = jnp.sum(agent_conflicts_idx > -1)

            # preparing random agent selection
            k1, k2 = jax.random.split(key, 2)
            random_number = jax.random.randint(
                k1,
                (1,),
                0,
                max_rand
            )

            # index in main matrix of agent of successful interaction
            rand_agent_idx = agent_conflicts_idx[random_number]

            # set that agent's bool to False, for inversion later
            new_agent_conflict = agent_conflicts.at[rand_agent_idx].set(False)

            # set all remaining True agents' values as the "empty" item
            step_arr = jnp.where(
                new_agent_conflict,
                Items.empty,
                step_arr
            ).astype(jnp.int16)

            # update conflict bools to reflect the post-conflict state
            _conflicts = conflicts.at[agent_conflicts_idx].set(0)
            conflicts = jnp.where(
                agent_conflicts,
                0,
                conflicts
            )
            conflicts = conflicts.at[conflict_idx].set(0)
            conflicts_matrix = jax.vmap(
                lambda c, x: jnp.where(c, x, jnp.array([False]*conflicts.shape[0]))
            )(conflicts, conflicts_matrix)

            # deal with next conflict
            return ((
                k2,
                conflicts,
                conflicts_matrix,
                step_arr
            ), step_arr)
        
        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        def check_collision(
                new_agent_locs: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check agent collisions.
            
            Args:
                - new_agent_locs: jnp.ndarray, the agent locations at the 
                current time step.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.all(x[:2] == y[:2]),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(new_agent_locs, new_agent_locs)

            return collisions
        
        # first attempt at func - needs improvement
        # inefficient due to double-checking collisions
        def check_interaction_conflict(
                items: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Function to check conflicting interaction targets.
            
            Args:
                - items: jnp.ndarray, the agent itemss at the interaction
                targets.
                
            Returns:
                - jnp.ndarray matrix of bool of agents in collision.
            '''
            matcher = jax.vmap(
                lambda x,y: jnp.logical_and(
                    jnp.all(x == y),
                    jnp.isin(x, self._agents)
                ),
                in_axes=(0, None)
            )

            collisions = jax.vmap(
                matcher,
                in_axes=(None, 0)
            )(items, items)

            return collisions

        def fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Function defining multi-collision logic.

            Args:
                - key: jax key for randomisation
                - collided_moved: jnp.ndarray, the agents which moved in the
                last time step and caused collisions.
                - collision_matrix: jnp.ndarray, the agents currently in
                collisions
                - agent_locs: jnp.ndarray, the agent locations at the previous
                time step.
                - new_agent_locs: jnp.ndarray, the agent locations at the
                current time step.

            Returns:
                - jnp.ndarray of the final positions after collisions are
                managed.
            """
            def scan_fn(
                    state,
                    idx
            ):
                key, collided_moved, collision_matrix, agent_locs, new_agent_locs = state

                return jax.lax.cond(
                    collided_moved[idx] > 0,
                    lambda: _fix_collisions(
                        key,
                        collided_moved,
                        collision_matrix,
                        agent_locs,
                        new_agent_locs
                    ),
                    lambda: (state, new_agent_locs.astype(jnp.int32))
                )

            _, ys = jax.lax.scan(
                scan_fn,
                (key, collided_moved, collision_matrix, agent_locs, new_agent_locs),
                jnp.arange(self.num_agents)
            )

            final_locs = ys[-1]

            return final_locs

        def _fix_collisions(
            key: jnp.ndarray,
            collided_moved: jnp.ndarray,
            collision_matrix: jnp.ndarray,
            agent_locs: jnp.ndarray,
            new_agent_locs: jnp.ndarray
        ) -> Tuple[Tuple, jnp.ndarray]:
            def select_random_true_index(key, array):
                # Calculate the cumulative sum of True values
                cumsum_array = jnp.cumsum(array)

                # Count the number of True values
                true_count = cumsum_array[-1]

                # Generate a random index in the range of the number of True
                # values
                rand_index = jax.random.randint(
                    key,
                    (1,),
                    0,
                    true_count
                )

                # Find the position of the random index within the cumulative
                # sum
                chosen_index = jnp.argmax(cumsum_array > rand_index)

                return chosen_index
            # Pick one from all who collided & moved
            colliders_idx = jnp.argmax(collided_moved)

            collisions = collision_matrix[colliders_idx]

            # Check whether any of collision participants didn't move
            collision_subjects = jnp.where(
                collisions,
                collided_moved,
                collisions
            )
            collision_mask = collisions == collision_subjects
            stayed = jnp.all(collision_mask)
            stayed_mask = jnp.logical_and(~stayed, ~collision_mask)
            stayed_idx = jnp.where(
                jnp.max(stayed_mask) > 0,
                jnp.argmax(stayed_mask),
                0
            )

            # Prepare random agent selection
            k1, k2 = jax.random.split(key, 2)
            rand_idx = select_random_true_index(k1, collisions)
            collisions_rand = collisions.at[rand_idx].set(False) # <<<< PROBLEM LINE        
            new_locs_rand = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_rand,
                agent_locs,
                new_agent_locs
            )

            collisions_stayed = jax.lax.select(
                jnp.max(stayed_mask) > 0,
                collisions.at[stayed_idx].set(False),
                collisions_rand
            )
            new_locs_stayed = jax.vmap(
                lambda p, l, n: jnp.where(p, l, n)
            )(
                collisions_stayed,
                agent_locs,
                new_agent_locs
            )

            # Choose between the two scenarios - revert positions if
            # non-mover exists, otherwise choose random agent if all moved
            new_agent_locs = jnp.where(
                stayed,
                new_locs_rand,
                new_locs_stayed
            )

            # Update move bools to reflect the post-collision positions
            collided_moved = jnp.clip(collided_moved - collisions, 0, 1)
            collision_matrix = collision_matrix.at[colliders_idx].set(
                [False] * collisions.shape[0]
            )
            return ((k2, collided_moved, collision_matrix, agent_locs, new_agent_locs), new_agent_locs)


        
        def combine_channels(
                grid: jnp.ndarray,
                agent: int,
                angles: jnp.ndarray,
                agent_pickups: jnp.ndarray,
                state: State,
                rep_vector: jnp.ndarray,
                bandit_occ: jnp.ndarray,
            ):
            '''
            Function to enforce symmetry in observations & generate final
            feature representation; current agent is permuted to first 
            position in the feature dimension.
            
            Args:
                - grid: jax ndarray of current agent's obs grid
                - agent: int, an index indicating current agent number
                - angles: jnp.ndarray of current agents' relative orientation
                in current agent's obs grid
                - agent_pickups: jnp.ndarray of agents able to interact
                - state: State, the env state obj
                - bandit_occ: bool grid (same HxW as ``grid``) — env-side bandits
            Returns:
                - grid with current agent [x] permuted to 1st position (after
                the first 4 "Items" features, so 5th position overall) in the
                feature dimension, "other" [x] agent 2nd, angle [x, x, x, x]
                3rd, pick-up [x] bool 4th, inventory [x, x] 5th, frozen [x]
                6th, for a final obs grid of shape (5, 5, 14) (additional 4
                one-hot places for 5 possible items)
            '''
            def move_and_collapse(
                    x: jnp.ndarray,
                    angle: jnp.ndarray,
                    is_bandit: jnp.ndarray,
                ) -> jnp.ndarray:

                # ``x`` is one-hot(grids - 1): index j = grid_value - 1. Agent i has
                # grid value len(Items)+i, so observer one-hot index = agent - 1
                # (equivalently (len(Items) - 1) + (agent - len(Items))).
                agent_oh_idx = agent - 1
                agent_element = jnp.array([jnp.int8(x[agent_oh_idx])])

                # mask to check if any other agent exists there
                mask = x[len(Items)-1:] > 0

                # does an agent exist which is not the subject?
                other_agent = jnp.int8(
                    jnp.logical_and(
                        jnp.any(mask),
                        jnp.logical_not(
                            agent_element
                        )
                    )
                )

                # what is the class of the item in cell (one-hot index, not grid id)
                item_idx = jnp.where(
                    x,
                    size=1
                )[0]
                cell_agent_i = item_idx - (len(Items) - 1)

                # check if agent is frozen and can observe inventories
                show_inv_bool = jnp.logical_and(
                        state.freeze[
                            agent-len(Items)
                        ].max(axis=-1) > 0,
                        item_idx >= (len(Items) - 1)
                )

                show_inv_idxs = jnp.where(
                    state.freeze[agent],
                    size=12, # since, in a setting where simultaneous interac-
                    fill_value=-1 # -tions can happen, only a max of 12 can
                )[0] # happen at once (zap logic), regardless of pop size

                inv_to_show = jnp.where(
                    jnp.logical_or(
                        jnp.logical_and(
                            show_inv_bool,
                            jnp.isin(cell_agent_i, show_inv_idxs),
                        ),
                        agent_element
                    ),
                    state.agent_invs[cell_agent_i],
                    jnp.array([0, 0], dtype=jnp.int8)
                )[0]

                # Timed-out (tagged) agents: Perolat freeze / removal. Drive the
                # obs channel from ``time_out`` (the live counter). The vestigial
                # ``state.freeze`` matrix is never updated and was always off.
                other_agent_i = cell_agent_i
                frozen = jnp.where(
                    other_agent,
                    state.time_out[other_agent_i] > 0,
                    0,
                )

                # get pickup/inv info
                pick_up_idx = jnp.where(
                    jnp.any(mask),
                    jnp.nonzero(mask, size=1)[0],
                    jnp.int8(-1)
                )
                picked_up = jnp.where(
                    pick_up_idx > -1,
                    agent_pickups[pick_up_idx],
                    jnp.int8(0)
                )

                # Bandits impersonate an agent grid ID for RGB but must collapse
                # to other+Bad in CNN obs (never self, never that agent's label).
                is_b = jnp.asarray(is_bandit, dtype=jnp.bool_)
                agent_element = jnp.where(
                    is_b, jnp.array([jnp.int8(0)]), agent_element
                )
                other_agent = jnp.where(is_b, jnp.int8(1), other_agent)
                inv_to_show = jnp.where(
                    is_b, jnp.array([0, 0], dtype=jnp.int8), inv_to_show
                )
                frozen = jnp.where(is_b, jnp.int8(0), frozen)
                picked_up = jnp.where(is_b, jnp.int8(0), picked_up)

                # Binary / continuous reputation planes (ch15 self, ch16 other).
                # Empty cell = 0. Agent cells carry the label from ``rep_vector``:
                # classic continuous tanh ∈ (−1,1); v2 binary uses +1=Good, −1=Bad
                # (never 0 on an occupied agent cell so empty ≠ Bad).
                observer_idx = agent - len(Items)
                is_self = (~is_b) & (x[agent_oh_idx] > 0)
                is_other = other_agent > 0
                self_rep = jnp.where(
                    is_self,
                    rep_vector[observer_idx],
                    jnp.float32(0.0),
                )
                other_rep = jnp.where(
                    is_b,
                    BANDIT_REP,
                    jnp.where(
                        is_other,
                        rep_vector[other_agent_i],
                        jnp.float32(0.0),
                    ),
                )
                # Ablation: keep label mechanics, hide rep channels from policy.
                self_rep = jnp.where(
                    self._mask_reputation_obs, jnp.float32(0.0), self_rep
                )
                other_rep = jnp.where(
                    self._mask_reputation_obs, jnp.float32(0.0), other_rep
                )
                # build extension (agent features unchanged; rep planes follow)
                extension = jnp.concatenate(
                    [
                        agent_element,
                        other_agent,
                        angle,
                        picked_up,
                        inv_to_show,
                        frozen,
                        jnp.reshape(self_rep, (1,)),
                        jnp.reshape(other_rep, (1,)),
                    ],
                    axis=-1
                )

                # build final feature vector (float32 so rep in [-1, 1] is exact)
                final_vec = jnp.concatenate(
                    [
                        x[: len(Items) - 1].astype(jnp.float32),
                        extension.astype(jnp.float32),
                    ],
                    axis=-1,
                )

                return final_vec

            new_grid = jax.vmap(
                jax.vmap(
                    move_and_collapse
                )
            )(grid, angles, bandit_occ)
            return new_grid
        
        def check_relative_orientation(
                agent: int,
                agent_locs: jnp.ndarray,
                grid: jnp.ndarray
            ) -> jnp.ndarray:
            '''
            Check's relative orientations of all other agents in view of
            current agent.
            
            Args:
                - agent: int, an index indicating current agent number
                - agent_locs: jax ndarray of agent locations (x, y, direction)
                - grid: jax ndarray of current agent's obs grid
                
            Returns:
                - grid with 1) int -1 in places where no agent exists, or
                where the agent is the current agent, and 2) int in range
                0-3 in cells of opposing agents indicating relative
                orientation to current agent.
            '''
            # we decrement by num of Items when indexing as we incremented by
            # 5 in constructor call (due to 5 non-agent Items enum & locations
            # are indexed from 0)
            idx = agent - len(Items)
            agents = jnp.delete(
                self._agents,
                idx,
                assume_unique_indices=True
            )
            curr_agent_dir = agent_locs[idx, 2]

            def calc_relative_direction(cell):
                cell_agent = cell - len(Items)
                cell_direction = agent_locs[cell_agent, 2]
                return (cell_direction - curr_agent_dir) % 4

            angle = jnp.where(
                jnp.isin(grid, agents),
                jax.vmap(calc_relative_direction)(grid),
                -1
            )

            return angle
        
        def rotate_grid(agent_loc: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
            '''
            Rotates agent's observation grid k * 90 degrees, depending on agent's
            orientation.

            Args:
                - agent_loc: jax ndarray of agent's x, y, direction
                - grid: jax ndarray of agent's obs grid

            Returns:
                - jnp.ndarray of new rotated grid.

            '''
            grid = jnp.where(
                agent_loc[2] == 1,
                jnp.rot90(grid, k=1, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 2,
                jnp.rot90(grid, k=2, axes=(0, 1)),
                grid,
            )
            grid = jnp.where(
                agent_loc[2] == 3,
                jnp.rot90(grid, k=3, axes=(0, 1)),
                grid,
            )

            return grid

        def _get_obs_point(agent_loc: jnp.ndarray) -> jnp.ndarray:
            '''
            Obtain the position of top-left corner of obs map using
            agent's current location & orientation.

            Args: 
                - agent_loc: jnp.ndarray, agent x, y, direction.
            Returns:
                - x, y: ints of top-left corner of agent's obs map.
            '''
            
            x, y, direction = agent_loc

            # Centered square extraction; ``rotate_grid`` then canonicalises
            # orientation and ``_get_obs`` crops the forward-biased rectangle.
            x, y = x + self.PADDING, y + self.PADDING

            x = x - self.OBS_RADIUS
            y = y - self.OBS_RADIUS

            return x, y

        def _get_obs(state: State) -> jnp.ndarray:
            '''
            Obtain the agent's observation of the grid.

            Args: 
                - state: State object containing env state.
            Returns:
                - jnp.ndarray of grid observation.
            '''
            # create state
            grid = jnp.pad(
                state.grid,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=Items.wall,
            )

            # obtain all agent obs-points
            agent_start_idxs = jax.vmap(_get_obs_point)(state.agent_locs)

            dynamic_slice = partial(
                jax.lax.dynamic_slice,
                operand=grid,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE)
            )

            # obtain agent obs grids
            grids = jax.vmap(dynamic_slice)(start_indices=agent_start_idxs)

            # Bandit occupancy (same FOV + rotation as grid) for Bad other_rep.
            b_occ = bandit_occupancy_grid(
                state.bandit_locs,
                state.bandit_active,
                self.GRID_SIZE_ROW,
                self.GRID_SIZE_COL,
            )
            b_occ_pad = jnp.pad(
                b_occ,
                ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)),
                constant_values=False,
            )
            b_slice = partial(
                jax.lax.dynamic_slice,
                operand=b_occ_pad,
                slice_sizes=(self.OBS_SIZE, self.OBS_SIZE),
            )
            b_views = jax.vmap(b_slice)(start_indices=agent_start_idxs)
            b_views = jax.vmap(rotate_grid)(state.agent_locs, b_views)

            # rotate agent obs grids
            grids = jax.vmap(rotate_grid)(state.agent_locs, grids)

            angles = jax.vmap(
                check_relative_orientation,
                in_axes=(0, None, 0)
            )(
                self._agents,
                state.agent_locs,
                grids
            )

            angles = jax.nn.one_hot(angles, 4)

            # one-hot (drop first channel as its empty blocks)
            grids = jax.nn.one_hot(
                grids - 1,
                num_agents + len(Items) - 1, # will be collapsed into a
                dtype=jnp.int8 # [Items, self, other, extra features] representation
            )

            # check agents that can interact
            inventory_sum = jnp.sum(state.agent_invs, axis=-1)
            agent_pickups = jnp.where(
                inventory_sum > INTERACT_THRESHOLD,
                True,
                False
            )

            # make index len(Item) always the current agent
            # and sum all others into an "other" agent
            # ``rep_vector`` is ``state.tag_filter_rep`` (pre-step, set by
            # ReputationWrapper). Painted here inside ``combine_channels`` so
            # self/other rep share the same rotate_grid + forward-crop path as
            # ch5/ch6 occupancy (rotation already applied to ``grids`` above).
            grids = jax.vmap(
                combine_channels,
                in_axes=(0, 0, 0, None, None, None, 0)
            )(
                grids,
                self._agents,
                angles,
                agent_pickups,
                state,
                state.tag_filter_rep,
                b_views,
            )

            # Crop the centered, rotation-canonicalised window. In the canonical
            # frame the agent sits at row OBS_BACKWARD, col OBS_SIDE of the crop
            # (back-centre); "forward" is the +row axis.
            r0 = self.OBS_RADIUS - self.OBS_BACKWARD
            r1 = self.OBS_RADIUS + self.OBS_FORWARD + 1
            c0 = self.OBS_RADIUS - self.OBS_SIDE
            c1 = self.OBS_RADIUS + self.OBS_SIDE + 1
            grids = grids[:, r0:r1, c0:c1, :]

            # Belt-and-suspenders: even if combine_channels forgets a path
            # (bandit, self, other), wipe ch15/ch16 after the crop so policy
            # never sees reputation when mask_reputation_obs is on.
            if self._mask_reputation_obs:
                grids = grids.at[..., 15].set(jnp.float32(0.0))
                grids = grids.at[..., 16].set(jnp.float32(0.0))

            return grids

        def _interact(
            key: jnp.ndarray,
            state: State,
            actions: jnp.ndarray,
            active: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, State, jnp.ndarray, jnp.ndarray]:
            '''
            Main interaction logic entry point.

            Args:
                - key: jax key for randomisation.
                - state: State env state object.
                - actions: jnp.ndarray of actions taken by agents.
                - active: ``(N,)`` bool — agents currently in play (not timed out).
            Returns:
                - reborn_players: ``(N,)`` bool victims whose loc is in a zap beam
                - state: updated env State (beam paint on grid)
                - zap_attempted: ``(N,)`` bool — agent took zap action
                - zap_successful: ``(N,)`` bool — zap beam hit ≥1 active agent
            '''
            # if interact
            zaps = jnp.isin(actions,
                jnp.array(
                    [
                        Actions.zap_forward,
                        # Actions.zap_ahead
                    ]
                )
            )

            interact_idx = jnp.int16(Items.interact)

            # remove old interact markers
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx,
                jnp.int16(Items.empty),
                state.grid,
            ))


            # Shared rectangular beam (length x width) for zap.
            beam_positions = beam_geometry.compute_beam_positions(
                state.agent_locs,
                self.GRID_SIZE_ROW,
                self.GRID_SIZE_COL,
                beam_length=self.beam_length,
                beam_width=self.beam_width,
            )
            clipped_beam, beam_in_bounds = beam_geometry.clip_beam_positions(
                beam_positions,
                self.GRID_SIZE_ROW,
                self.GRID_SIZE_COL,
            )
            if self._use_police_beam:
                police_beam_positions = beam_geometry.compute_beam_positions(
                    state.agent_locs,
                    self.GRID_SIZE_ROW,
                    self.GRID_SIZE_COL,
                    beam_length=self._police_beam_length,
                    beam_width=self._police_beam_width,
                )
                clipped_police, police_in_bounds = beam_geometry.clip_beam_positions(
                    police_beam_positions,
                    self.GRID_SIZE_ROW,
                    self.GRID_SIZE_COL,
                )
            else:
                police_beam_positions = beam_positions
                clipped_police = clipped_beam
                police_in_bounds = beam_in_bounds

            num_agents = state.agent_locs.shape[0]
            num_beam_cells = clipped_beam.shape[1]
            num_police_beam_cells = clipped_police.shape[1]
            zaps_squeezed = zaps.squeeze().astype(jnp.bool_).reshape((num_agents,))
            shooters = zaps_squeezed & active.astype(jnp.bool_)

            hit = beam_geometry.beam_hit_matrix(
                state.agent_locs,
                beam_positions,
                self.GRID_SIZE_ROW,
                self.GRID_SIZE_COL,
                active=active.astype(jnp.bool_),
            )
            if self._use_police_beam:
                hit_police = beam_geometry.beam_hit_matrix(
                    state.agent_locs,
                    police_beam_positions,
                    self.GRID_SIZE_ROW,
                    self.GRID_SIZE_COL,
                    active=active.astype(jnp.bool_),
                )
                hit = jnp.where(self._police_mask[:, None], hit_police, hit)
            # (N, N) bool: [i, j] = True ⇒ active shooter i tagged agent j this step.
            zap_hit_matrix = shooters[:, None] & hit
            zap_hit_matrix = mask_zap_hits_on_police(zap_hit_matrix, self._police_mask)
            tag_matrix = apply_police_focus_on_bad_zap_hits(
                zap_hit_matrix,
                self._police_mask,
                state.tag_filter_rep,
                enabled=self._police_focus_on_bad,
            )
            tag_matrix = apply_peer_focus_on_bad_zap_hits(
                tag_matrix,
                self._police_mask,
                state.tag_filter_rep,
                enabled=self._peer_focus_on_bad,
            )
            # Victims for freeze: never include police (also masked above).
            reborn_players = jnp.any(tag_matrix, axis=0) & ~self._police_mask
            zap_attempted = zaps_squeezed
            zap_successful = shooters & jnp.any(tag_matrix, axis=1)

            # Bandit hits: same beams as agents; never freeze/timeout bandits.
            bandit_hit = beam_hit_bandits(
                beam_positions,
                state.bandit_locs,
                state.bandit_active,
                self.GRID_SIZE_ROW,
                self.GRID_SIZE_COL,
            )
            if self._use_police_beam:
                bandit_hit_p = beam_hit_bandits(
                    police_beam_positions,
                    state.bandit_locs,
                    state.bandit_active,
                    self.GRID_SIZE_ROW,
                    self.GRID_SIZE_COL,
                )
                bandit_hit = jnp.where(
                    self._police_mask[:, None], bandit_hit_p, bandit_hit
                )
            bandit_hit_matrix = shooters[:, None] & bandit_hit

            aux_grid = jnp.copy(state.grid)

            def paint_beam_cells(
                active_agents,
                marker_or_keep,
                grid,
                force_marker,
                cells_per_agent,
                in_bounds_per_agent,
                num_cells,
            ):
                """Paint in-bounds beam cells for agents where ``active_agents`` is True."""

                def _paint_one(agent_i, grid_acc):
                    is_on = active_agents[agent_i]
                    cells = cells_per_agent[agent_i]  # (C, 2)
                    valid = in_bounds_per_agent[agent_i]

                    def _cell(cell_j, g):
                        r, c = cells[cell_j, 0], cells[cell_j, 1]
                        should = is_on & valid[cell_j]
                        if force_marker:
                            new_val = jnp.int16(marker_or_keep)
                        else:
                            cur = g[r, c]
                            new_val = jnp.where(cur != 0, cur, interact_idx)
                        return g.at[r, c].set(
                            jnp.where(should, new_val, g[r, c]).astype(jnp.int16)
                        )

                    return jax.lax.fori_loop(0, num_cells, _cell, grid_acc)

                return jax.lax.fori_loop(0, num_agents, _paint_one, grid)

            harvester_shooters = shooters & ~self._police_mask
            police_shooters = shooters & self._police_mask
            aux_grid = paint_beam_cells(
                harvester_shooters,
                interact_idx,
                aux_grid,
                force_marker=False,
                cells_per_agent=clipped_beam,
                in_bounds_per_agent=beam_in_bounds,
                num_cells=num_beam_cells,
            )
            if self._use_police_beam:
                aux_grid = paint_beam_cells(
                    police_shooters,
                    interact_idx,
                    aux_grid,
                    force_marker=False,
                    cells_per_agent=clipped_police,
                    in_bounds_per_agent=police_in_bounds,
                    num_cells=num_police_beam_cells,
                )
            else:
                aux_grid = paint_beam_cells(
                    police_shooters,
                    interact_idx,
                    aux_grid,
                    force_marker=False,
                    cells_per_agent=clipped_beam,
                    in_bounds_per_agent=beam_in_bounds,
                    num_cells=num_beam_cells,
                )

            state = state.replace(
                grid=jnp.where(
                    jnp.any(shooters),
                    aux_grid,
                    state.grid
                )
            )

            return (
                reborn_players,
                state,
                zap_attempted,
                zap_successful,
                tag_matrix,
                zap_hit_matrix,
                bandit_hit_matrix,
            )

        def get_current_s_interest(timestep):
            """Calculate current s_interest based on timestep and schedule."""
            if self.s_interest_schedule is None:
                return self.s_interest

            # Calculate which phase we're in using JAX operations
            phase = timestep // self.s_interest_change_every
            phase_idx = phase % self.s_interest_schedule.shape[0]
            return self.s_interest_schedule[phase_idx]

        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: jnp.ndarray,
            timestep: int = 0
        ):
            """Step the environment."""
            # actions = self.action_set.take(indices=jnp.array([actions["0"], actions["agent_1"]]))
            actions = jnp.array(actions)

            # Tag time-out (Perolat et al. 2017): agents currently removed from the
            # game (time_out > 0) take no action and are held off the board.
            active = state.time_out <= 0
            actions = jnp.where(active, actions, jnp.int32(Actions.stay))

            # regrow apple (Perolat p(L); Melting Pot canRegrowIfOccupied=True)
            apple_map = state.apples

            def count_apple(apple_locs):
                # Neighbor stock from the apple *layer* so apples under agents count.
                r0, c0 = apple_locs[0], apple_locs[1]
                apple_nums = (
                    jnp.where((apple_map[r0 - 1, c0] == 1) & (r0 - 1 >= 0), 1, 0)
                    + jnp.where(
                        (apple_map[r0 + 1, c0] == 1) & (r0 + 1 < self.GRID_SIZE_ROW),
                        1,
                        0,
                    )
                    + jnp.where((apple_map[r0, c0 - 1] == 1) & (c0 - 1 >= 0), 1, 0)
                    + jnp.where(
                        (apple_map[r0, c0 + 1] == 1) & (c0 + 1 < self.GRID_SIZE_COL),
                        1,
                        0,
                    )
                    + jnp.where((apple_map[r0 - 2, c0] == 1) & (r0 - 2 >= 0), 1, 0)
                    + jnp.where(
                        (apple_map[r0 + 2, c0] == 1) & (r0 + 2 < self.GRID_SIZE_ROW),
                        1,
                        0,
                    )
                    + jnp.where((apple_map[r0, c0 - 2] == 1) & (c0 - 2 >= 0), 1, 0)
                    + jnp.where(
                        (apple_map[r0, c0 + 2] == 1) & (c0 + 2 < self.GRID_SIZE_COL),
                        1,
                        0,
                    )
                    + jnp.where(
                        (apple_map[r0 - 1, c0 - 1] == 1)
                        & (r0 - 1 >= 0)
                        & (c0 - 1 >= 0),
                        1,
                        0,
                    )
                    + jnp.where(
                        (apple_map[r0 - 1, c0 + 1] == 1)
                        & (r0 - 1 >= 0)
                        & (c0 + 1 < self.GRID_SIZE_COL),
                        1,
                        0,
                    )
                    + jnp.where(
                        (apple_map[r0 + 1, c0 - 1] == 1)
                        & (r0 + 1 < self.GRID_SIZE_ROW)
                        & (c0 - 1 >= 0),
                        1,
                        0,
                    )
                    + jnp.where(
                        (apple_map[r0 + 1, c0 + 1] == 1)
                        & (r0 + 1 < self.GRID_SIZE_ROW)
                        & (c0 + 1 < self.GRID_SIZE_COL),
                        1,
                        0,
                    )
                )
                return apple_nums

            near_apple_nums = jax.vmap(count_apple)(self.SPAWNS_APPLE)

            def regrow_apple(apple_locs, near_apple, prob):
                # Perolat et al. (2017) local-stock regrowth probability p(L):
                #   L=0 -> 0, L in {1,2} -> 0.01, L in {3,4} -> 0.05, L>4 -> 0.1
                # ``regrowth_scale`` < 1 slows replenishment (cheaper pressure knob
                # than raising N). L=0 stays irreversible (×scale still 0).
                scale = jnp.float32(self.regrowth_scale)
                regrow_prob = scale * jnp.where(
                    near_apple > 4,
                    0.1,
                    jnp.where(
                        near_apple >= 3,
                        0.05,
                        jnp.where(near_apple >= 1, 0.01, 0.0),
                    ),
                )
                r, c = apple_locs[0], apple_locs[1]
                is_live = apple_map[r, c] == 1
                # Melting Pot DensityRegrow canRegrowIfOccupied=True: a dormant
                # site may sprout even if an agent stands on it.
                sprouts = (~is_live) & (prob < regrow_prob)
                return jnp.where(is_live | sprouts, jnp.int16(1), jnp.int16(0))

            prob = jax.random.uniform(key, shape=(len(self.SPAWNS_APPLE),))
            new_apple_flags = jax.vmap(regrow_apple)(
                self.SPAWNS_APPLE, near_apple_nums, prob
            )
            apple_map = apple_map.at[
                self.SPAWNS_APPLE[:, 0], self.SPAWNS_APPLE[:, 1]
            ].set(new_apple_flags)

            # Sync apple sprites onto the grid without overwriting agents/walls.
            # Vectorized (safe when SPAWNS_APPLE is empty — fori_loop tracing is not).
            ar = self.SPAWNS_APPLE[:, 0]
            ac = self.SPAWNS_APPLE[:, 1]
            cur = state.grid[ar, ac]
            is_agent = jnp.any(cur[:, None] == self._agents[None, :], axis=-1)
            is_wall = cur == jnp.int16(Items.wall)
            want = apple_map[ar, ac] == 1
            new_vals = jnp.where(
                is_agent | is_wall,
                cur,
                jnp.where(want, jnp.int16(Items.apple), jnp.int16(Items.empty)),
            ).astype(jnp.int16)
            synced_grid = state.grid.at[ar, ac].set(new_vals)
            state = state.replace(grid=synced_grid, apples=apple_map)

            # moving all agents
            # Always clear each agent's marker from their last cell, then place
            # only *active* agents. Timed-out agents stay off the board (Perolat
            # et al. 2017: removed for tag_timeout steps). Restoring an apple
            # under a vacating agent matches Melting Pot's layered apple/agent.
            def _clear_agent_marker(i, g):
                r, c = state.agent_locs[i, 0], state.agent_locs[i, 1]
                is_me = g[r, c] == self._agents[i]
                under = jnp.where(
                    state.apples[r, c] == 1,
                    jnp.int16(Items.apple),
                    jnp.int16(Items.empty),
                )
                val = jnp.where(is_me, under, g[r, c]).astype(jnp.int16)
                return g.at[r, c].set(val)

            new_grid = jax.lax.fori_loop(
                0, self.num_agents, _clear_agent_marker, state.grid
            )

            def _place_active(i, g):
                r, c = state.reborn_locs[i, 0], state.reborn_locs[i, 1]
                val = jnp.where(active[i], self._agents[i], g[r, c]).astype(jnp.int16)
                return g.at[r, c].set(val)

            new_grid = jax.lax.fori_loop(0, self.num_agents, _place_active, new_grid)
            state = state.replace(grid=new_grid)
            state = state.replace(agent_locs=state.reborn_locs)

            # state = state.replace(reborn_locs=state.agent_locs)

            key, subkey = jax.random.split(key)
            # Rotation actions (turn_left/turn_right) update orientation only.
            # Keep shape (N, 3) — do not squeeze (N=1 would collapse to (3,)).
            all_new_locs = jax.vmap(
                lambda p, a: (p.astype(jnp.int32) + self.ROTATIONS[a].astype(jnp.int32))
                % jnp.array(
                    [self.GRID_SIZE_ROW + 1, self.GRID_SIZE_COL + 1, 4],
                    dtype=jnp.int32,
                )
            )(p=state.agent_locs, a=actions)

            # Egocentric translation (Perolat et al. 2017): the four step actions
            # move relative to the agent's current facing rather than in absolute
            # grid directions. up=forward, down=backward, right/left=strafe.
            def _ego_move_delta(loc, a):
                d = loc[2]
                delta = jnp.select(
                    [
                        a == Actions.up,     # step forward
                        a == Actions.down,   # step backward
                        a == Actions.right,  # strafe right
                        a == Actions.left,   # strafe left
                    ],
                    [
                        STEP[d],
                        STEP[(d + 2) % 4],
                        STEP[(d + 1) % 4],
                        STEP[(d + 3) % 4],
                    ],
                    default=jnp.zeros((3,), dtype=STEP.dtype),
                )
                return delta.astype(jnp.int32)

            move_delta = jax.vmap(_ego_move_delta)(state.agent_locs, actions)
            all_new_locs = all_new_locs + move_delta
            
            all_new_locs = jax.vmap(
                jnp.clip,
                in_axes=(0, None, None)
            )(
                all_new_locs,
                jnp.array([0, 0, 0], dtype=jnp.int32),
                jnp.array(
                    [self.GRID_SIZE_ROW - 1, self.GRID_SIZE_COL - 1, 3],
                    dtype=jnp.int32
                ),
            )

            # if you bounced back to your original space,
            # change your move to stay (for collision logic)
            agents_move = jax.vmap(lambda n, p: jnp.any(n[:2] != p[:2]))(n=all_new_locs, p=state.agent_locs)

            # generate bool mask for agents colliding
            collision_matrix = check_collision(all_new_locs)
            # Timed-out agents are off the board: they neither cause nor suffer
            # collisions.
            collision_matrix = collision_matrix & active[:, None] & active[None, :]

            # sum & subtract "self-collisions"
            collisions = jnp.sum(
                collision_matrix,
                axis=-1,
                dtype=jnp.int8
            ) - 1
            collisions = jnp.minimum(collisions, 1)

            # identify which of those agents made wrong moves
            collided_moved = jnp.maximum(
                collisions - ~agents_move,
                0
            )

            # fix collisions at the correct indices
            new_locs = jax.lax.cond(
                jnp.max(collided_moved) > 0,
                lambda: fix_collisions(
                    key,
                    collided_moved,
                    collision_matrix,
                    state.agent_locs.astype(jnp.int32),
                    all_new_locs.astype(jnp.int32),
                ),
                lambda: all_new_locs.astype(jnp.int32),
            )

            # Cannot move onto walls or another agent marker (apples are walkable).
            # In-place actions (turn / stay) keep the same cell; that cell still
            # holds *this* agent's marker, so treat agent-occupancy as blocking
            # only when the position actually changes. Otherwise turns are
            # silently reverted and agents can never reorient.
            cell = state.grid[new_locs[:, 0], new_locs[:, 1]]
            is_agent_marker = jnp.any(cell[:, None] == self._agents[None, :], axis=-1)
            pos_changed = jnp.any(
                new_locs[:, :2] != state.agent_locs[:, :2], axis=-1
            )
            blocked = (cell == jnp.int16(Items.wall)) | (
                is_agent_marker & pos_changed
            )
            condition_3d = jnp.stack([blocked, blocked, blocked], axis=-1)
            new_locs = jnp.where(condition_3d, state.agent_locs, new_locs)

            # Eat from the apple *layer* only on *entering* a cell (Melting Pot
            # Edible.onEnter). A sprout under a stationary agent does not auto-collect.
            def coin_matcher(p: jnp.ndarray) -> jnp.ndarray:
                return jnp.array([state.apples[p[0], p[1]] == 1])

            apple_matches = jax.vmap(coin_matcher)(p=new_locs)
            moved = jnp.any(new_locs[:, :2] != state.agent_locs[:, :2], axis=-1)
            apple_matches = apple_matches & active[:, None] & moved[:, None]
            # Police never collect apples — layer stays occupied.
            apple_matches = apple_matches & ~self._police_mask[:, None]
            harvested_now = apple_matches.squeeze()
            # Local-last: eaten apple had L=0 neighbors in the Perolat L2≤2 disk
            # (before clear) → that site cannot regrow. Always reported in info;
            # ReputationV2Wrapper may OR it into the burst timeout trigger.
            local_L = jax.vmap(
                lambda loc: l2_disk_live_apple_count(
                    state.apples,
                    loc[0],
                    loc[1],
                    self.GRID_SIZE_ROW,
                    self.GRID_SIZE_COL,
                )
            )(new_locs)
            locally_last_apple = harvested_now & (local_L == 0) & active

            new_invs = state.agent_invs + apple_matches
            state = state.replace(agent_invs=new_invs)

            # Clear eaten apples from the layer.
            eaten_r = new_locs[:, 0]
            eaten_c = new_locs[:, 1]
            apple_map = state.apples

            def _clear_eaten(i, amap):
                r, c = eaten_r[i], eaten_c[i]
                clear = apple_matches[i, 0]
                return amap.at[r, c].set(
                    jnp.where(clear, jnp.int16(0), amap[r, c]).astype(jnp.int16)
                )

            apple_map = jax.lax.fori_loop(0, self.num_agents, _clear_eaten, apple_map)
            state = state.replace(apples=apple_map)

            # update grid: always clear markers, place only active agents;
            # restore uneaten apples under vacated cells.
            old_grid = state.grid

            def _clear_agent_marker_cur(i, g):
                r, c = state.agent_locs[i, 0], state.agent_locs[i, 1]
                is_me = g[r, c] == self._agents[i]
                under = jnp.where(
                    state.apples[r, c] == 1,
                    jnp.int16(Items.apple),
                    jnp.int16(Items.empty),
                )
                val = jnp.where(is_me, under, g[r, c]).astype(jnp.int16)
                return g.at[r, c].set(val)

            new_grid = jax.lax.fori_loop(
                0, self.num_agents, _clear_agent_marker_cur, old_grid
            )

            def _place_active_new(i, g):
                r, c = new_locs[i, 0], new_locs[i, 1]
                val = jnp.where(active[i], self._agents[i], g[r, c]).astype(jnp.int16)
                return g.at[r, c].set(val)

            new_grid = jax.lax.fori_loop(0, self.num_agents, _place_active_new, new_grid)
            state = state.replace(grid=new_grid)

            # update agent locations
            state = state.replace(agent_locs=new_locs)

            # Bandits: clear markers, random cardinal move, then zap/respawn/paint.
            # Kept on-grid during agent move so agent markers block bandit cells.
            state = state.replace(
                grid=clear_bandits_from_grid(
                    state.grid,
                    bandit_locs=state.bandit_locs,
                    bandit_active=state.bandit_active,
                    apples=state.apples,
                    agent_item_base=len(Items),
                    bandit_visual_id=state.bandit_visual_id,
                )
            )
            key, bandit_move_key = jax.random.split(key)
            new_bandit_locs, key = move_bandits(
                bandit_move_key,
                bandit_locs=state.bandit_locs,
                bandit_active=state.bandit_active,
                grid=state.grid,
                apples=state.apples,
                agent_locs=state.agent_locs,
                agent_active=active,
                wall_id=int(Items.wall),
                grid_h=self.GRID_SIZE_ROW,
                grid_w=self.GRID_SIZE_COL,
            )
            state = state.replace(bandit_locs=new_bandit_locs)

            (
                reborn_players,
                state,
                zap_attempted,
                zap_successful,
                tag_matrix,
                zap_hit_matrix,
                bandit_hit_matrix,
            ) = _interact(key, state, actions, active)

            # Zap → delay/respawn (bandits never take peer/system timeouts).
            bandit_zapped = jnp.any(bandit_hit_matrix, axis=0)
            key, bandit_respawn_key = jax.random.split(key)
            (
                b_locs,
                b_active,
                b_cd,
                b_vis,
                key,
            ) = apply_bandit_zap_respawn(
                bandit_respawn_key,
                bandit_locs=state.bandit_locs,
                bandit_active=state.bandit_active,
                bandit_respawn_cd=state.bandit_respawn_cd,
                bandit_visual_id=state.bandit_visual_id,
                zapped=bandit_zapped,
                respawn_delay=resolve_bandit_respawn_delay(
                    state.episodes_completed, self._bandit_cfg
                ),
                grid=state.grid,
                apples=state.apples,
                agent_locs=state.agent_locs,
                agent_active=active,
                num_agents=self.num_agents,
                wall_id=int(Items.wall),
                grid_h=self.GRID_SIZE_ROW,
                grid_w=self.GRID_SIZE_COL,
            )
            state = state.replace(
                bandit_locs=b_locs,
                bandit_active=b_active,
                bandit_respawn_cd=b_cd,
                bandit_visual_id=b_vis,
                grid=paint_bandits_on_grid(
                    state.grid,
                    bandit_locs=b_locs,
                    bandit_active=b_active,
                    bandit_visual_id=b_vis,
                    agent_item_base=len(Items),
                ),
            )

            # --- Reputation enforcement (negative-rep stochastic timeout) ----
            # Uses pre-step ``tag_filter_rep`` (set by ReputationWrapper). Same
            # ``newly_tagged`` → ``time_out`` → scrub → respawn path as beam tags.
            key, enforce_key = jax.random.split(key)
            if self.reputation_enforcement:
                reputation_zapped, reputation_zap_prob = sample_reputation_enforcement(
                    enforce_key,
                    state.tag_filter_rep,
                    self.reputation_enforcement_strictness,
                    active,
                )
                reputation_zapped = reputation_zapped & ~self._police_mask
            else:
                reputation_zapped = jnp.zeros((num_agents,), dtype=jnp.bool_)
                reputation_zap_prob = jnp.zeros((num_agents,), dtype=jnp.float32)

            # --- Tag time-out bookkeeping (Perolat et al. 2017) ---------------
            # Only currently-active agents can be tagged. A tagged agent is
            # removed for ``self.tag_timeout`` steps (or police_timeout when
            # the freeze was caused by a police zap); timed-out agents count
            # down and respawn at a fresh spawn point when their timer expires.
            harvester_hit_victims, police_hit_victims = split_beam_victims(
                tag_matrix, self._police_mask
            )
            agent_beam_hit = harvester_hit_victims & active & ~self._police_mask
            police_beam_hit = police_hit_victims & active & ~self._police_mask
            enforcement_hit = reputation_zapped & active & ~self._police_mask
            newly_tagged = agent_beam_hit | police_beam_hit | enforcement_hit

            harvester_dur = jnp.int16(self.tag_timeout)
            police_dur = jnp.int16(self._police_timeout)
            # Prefer the longer freeze if multiple causes fire on the same step.
            hit_dur = jnp.maximum(
                jnp.where(agent_beam_hit | enforcement_hit, harvester_dur, jnp.int16(0)),
                jnp.where(police_beam_hit, police_dur, jnp.int16(0)),
            )
            time_out_next = jnp.where(
                newly_tagged,
                hit_dur,
                jnp.maximum(state.time_out - 1, 0).astype(jnp.int16),
            )
            # Police are never timed out (belt-and-suspenders).
            time_out_next = jnp.where(self._police_mask, jnp.int16(0), time_out_next)
            just_respawned = state.time_out == 1  # timer expires -> back in play

            # Snapshot cause *before* updating so the just_respawned step (still
            # inactive) is attributed correctly; clear applies to the next step.
            timeout_from_agent_pre = state.timeout_from_agent
            timeout_from_enforcement_pre = state.timeout_from_enforcement
            timeout_from_police_pre = state.timeout_from_police
            timeout_from_agent_next = jnp.where(
                newly_tagged,
                agent_beam_hit,
                jnp.where(just_respawned, False, timeout_from_agent_pre),
            )
            timeout_from_enforcement_next = jnp.where(
                newly_tagged,
                enforcement_hit,
                jnp.where(just_respawned, False, timeout_from_enforcement_pre),
            )
            timeout_from_police_next = jnp.where(
                newly_tagged,
                police_beam_hit,
                jnp.where(just_respawned, False, timeout_from_police_pre),
            )

            re_agents_pos = jax.random.permutation(subkey, self._RESPAWN_POOL)[:num_agents]
            player_dir = jax.random.randint(
                subkey, shape=(num_agents,), minval=0, maxval=3, dtype=jnp.int8
            )
            re_agent_locs = jnp.stack(
                [
                    re_agents_pos[:, 0].astype(jnp.int32),
                    re_agents_pos[:, 1].astype(jnp.int32),
                    player_dir.astype(jnp.int32),
                ],
                axis=1,
            )

            # Next-step positions: agents whose time-out just expired respawn at a
            # random spawn; everyone else keeps their current position (timed-out
            # agents stay parked off-board until they respawn).
            new_re_locs = jnp.where(just_respawned[:, None], re_agent_locs, new_locs)
            state = state.replace(
                reborn_locs=new_re_locs,
                time_out=time_out_next,
                timeout_from_agent=timeout_from_agent_next,
                timeout_from_enforcement=timeout_from_enforcement_next,
                timeout_from_police=timeout_from_police_next,
            )

            # Immediately remove anyone with time_out_next > 0 from the grid so
            # the returned state (and next obs) match Perolat "removed" — not a
            # one-step-delayed ghost sprite. Restore apple under scrubbed agent.
            def _scrub_timed_out(i, g):
                r, c = state.agent_locs[i, 0], state.agent_locs[i, 1]
                is_me = g[r, c] == self._agents[i]
                scrub = (time_out_next[i] > 0) & is_me
                under = jnp.where(
                    state.apples[r, c] == 1,
                    jnp.int16(Items.apple),
                    jnp.int16(Items.empty),
                )
                val = jnp.where(scrub, under, g[r, c]).astype(jnp.int16)
                return g.at[r, c].set(val)

            state = state.replace(
                grid=jax.lax.fori_loop(0, self.num_agents, _scrub_timed_out, state.grid)
            )

            env_rewards_per_agent = apple_matches.squeeze().astype(jnp.float32)

            if self.shared_rewards:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards)

                rewards_sum_all_agents = jnp.zeros((self.num_agents, 1))
                rewards_sum = jnp.sum(original_rewards)
                rewards_sum_all_agents += rewards_sum
                rewards = rewards_sum_all_agents
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.inequity_aversion:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                if self.smooth_rewards:
                    should_smooth = (state.inner_t % 1) == 0
                    new_smooth_rewards = 0.99 * 0.01* state.smooth_rewards + original_rewards
                    rewards, disadvantageous, advantageous = self.get_inequity_aversion_rewards_immediate(new_smooth_rewards, state.inner_t, self.inequity_aversion_target_agents, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    state = state.replace(smooth_rewards=new_smooth_rewards)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "smooth_rewards": state.smooth_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
                else:
                    rewards, disadvantageous, advantageous = self.get_inequity_aversion_rewards_immediate(original_rewards, state.inner_t, self.inequity_aversion_target_agents, self.inequity_aversion_alpha, self.inequity_aversion_beta)
                    info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.svo:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents
                rewards, theta = self.get_svo_rewards(original_rewards, self.svo_w, self.svo_ideal_angle_degrees, self.svo_target_agents)
                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "svo_theta": theta.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                }
            elif self.interest:
                rewards = jnp.zeros((self.num_agents, 1))
                original_rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents

                # Calculate current s_interest based on timestep
                current_s_interest = get_current_s_interest(timestep)
                original_flat = original_rewards.squeeze()

                # Each agent gets s * their_reward + (1-s)/(n-1) * sum_of_others
                total_reward = jnp.sum(original_flat)
                others_reward = total_reward - original_flat  # sum of all other agents' rewards

                rewards = (current_s_interest * original_flat +
                        (1 - current_s_interest) / (self.num_agents - 1) * others_reward).reshape(-1, 1)

                info = {
                    "original_rewards": original_rewards.squeeze(),
                    "shaped_rewards": rewards.squeeze(),
                    "s_interest": current_s_interest,
                }
            else:
                # Truthful Commons Harvest reward: +1 per apple collected, per
                # agent (Lupu & Precup §4.1). The historical ``* self.num_agents``
                # inflated the individual-reward signal by N and made returns /
                # R scale with agent count; removed for paper faithfulness.
                rewards = jnp.zeros((self.num_agents, 1))
                rewards = jnp.where(apple_matches, 1, rewards)
                info = {}

            rewards = rewards.squeeze()
            # Police learning signal: +1 Bad / −1 Good hits (pre-step labels).
            # Bandit columns reuse the same Bad path via extended hit matrix.
            police_hits_ext, police_labels_ext = extended_zap_labels_and_hits(
                tag_matrix,
                state.tag_filter_rep,
                bandit_hit_matrix,
            )
            police_zap_reward = police_binary_zap_reward(
                police_hits_ext,
                police_labels_ext,
                self._police_mask,
            )
            rewards = rewards + police_zap_reward
            info["environmental_rewards"] = env_rewards_per_agent
            info["police_zap_reward"] = police_zap_reward
            # "tagged-out" indicator for metrics: agents removed from play during
            # this step (i.e. serving a tag time-out). Exactly ``tag_timeout``
            # steps are counted per successful tag (Perolat et al. 2017 P metric).
            # Cause-split flags use *pre-step* cause state so the hit step itself
            # still reports tagged=False (agent was active at step start) while
            # subsequent timeout steps attribute inactivity correctly.
            inactive = jnp.logical_not(active)
            info["tagged"] = inactive
            info["agent_tagged"] = inactive & timeout_from_agent_pre
            info["enforcement_tagged"] = inactive & timeout_from_enforcement_pre
            info["police_tagged"] = inactive & timeout_from_police_pre
            info["zap_attempted"] = zap_attempted
            info["zap_successful"] = zap_successful
            # Beam occupancy at attempt time (pre police-focus): [i,j]=True if
            # active shooter i's beam covers active non-police agent j. Used for
            # Bad-targeting diagnostics (attempt-time, includes eventual misses
            # when focus clears a hit). Freeze/timeout agents are already out of
            # ``active`` so they are not eligible targets.
            info["zap_beam_matrix"] = zap_hit_matrix
            # Full matrix after police immunity / focus; used by reputation + metrics.
            info["zap_hit_matrix"] = tag_matrix
            info["police_agent"] = self._police_mask
            info["police_zap_attempted"] = zap_attempted & self._police_mask
            info["police_zap_successful"] = zap_successful & self._police_mask
            info["police_zap_hit_matrix"] = tag_matrix & self._police_mask[:, None]
            info["just_respawned"] = just_respawned
            info["reputation_zapped"] = reputation_zapped
            info["reputation_zap_prob"] = reputation_zap_prob
            # Hit events this step (newly frozen causes) — for timeout-cause metrics.
            info["agent_beam_timeout_hit"] = agent_beam_hit
            info["enforcement_timeout_hit"] = enforcement_hit
            info["police_timeout_hit"] = police_beam_hit
            info["locally_last_apple"] = locally_last_apple
            # Bandits: separate from real-agent zap matrices / accuracy metrics.
            info["bandit_zap_matrix"] = bandit_hit_matrix
            info["bandit_zap_count"] = jnp.sum(
                bandit_hit_matrix.astype(jnp.float32)
            )
            info["bandit_zapped"] = bandit_zapped
            info["bandit_active_count"] = jnp.sum(
                state.bandit_active.astype(jnp.float32)
            )
            info["target_type_bandit"] = jnp.any(bandit_hit_matrix, axis=1)
            # Start-of-step activity + actions after the inactive→stay override.
            # Trainers should store ``executed_actions`` (not the raw policy
            # output) and skip replay for ``~active`` agents.
            info["active"] = active
            info["executed_actions"] = actions

            AppleCount = jnp.sum(state.apples == 1)
            info["AppleCount_info"] = jnp.zeros((self.num_agents, 1)).squeeze(-1) + AppleCount
            
            
            state_nxt = State(
                agent_locs=state.agent_locs,
                agent_invs=state.agent_invs,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                apples=state.apples,
                freeze=state.freeze,
                reborn_locs=state.reborn_locs,
                time_out=state.time_out,
                timeout_from_agent=state.timeout_from_agent,
                timeout_from_enforcement=state.timeout_from_enforcement,
                timeout_from_police=state.timeout_from_police,
                tag_filter_rep=state.tag_filter_rep,
                bandit_locs=state.bandit_locs,
                bandit_active=state.bandit_active,
                bandit_respawn_cd=state.bandit_respawn_cd,
                bandit_visual_id=state.bandit_visual_id,
                episodes_completed=state.episodes_completed,
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            # Fixed horizon (paper) OR optional early end when the commons is
            # fully depleted. Perolat L=0 is irreversible, so with
            # terminate_on_barren the remaining steps would yield no apple reward.
            reset_horizon = inner_t == num_inner_steps
            reset_barren = jnp.logical_and(
                jnp.bool_(self.terminate_on_barren),
                AppleCount == 0,
            )
            reset_inner = jnp.logical_or(reset_horizon, reset_barren)

            # if inner episode is done, return start state for next game
            eps_next = jnp.asarray(state.episodes_completed, dtype=jnp.int32) + jnp.int32(
                1
            )
            state_re = _reset_state(key, episodes_completed=eps_next)

            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree.map(
                lambda x, y: jnp.where(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            # Agent-level done tracks *inner* episode boundaries (T = num_inner_steps).
            # ``__all__`` stays outer-only so multi-outer runs don't terminate early.
            # LogWrapper only watches ``__all__``, so training logs use
            # ``info["inner_episode_done"]`` for episode-return metrics.
            done = {f"{a}": reset_inner for a in self.agents}
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            # Keep terminal-step rewards (incl. the apple that emptied the
            # commons). Zeroing them dropped the last-step learning signal.
            info["terminated_on_barren"] = jnp.full(
                (self.num_agents,), reset_barren, dtype=jnp.bool_
            )

            # Broadcast so CTRolloutManager's vmap yields (NUM_ENVS, N).
            info["inner_episode_done"] = jnp.full(
                (self.num_agents,), reset_inner, dtype=jnp.bool_
            )
            info["inner_t"] = jnp.full(
                (self.num_agents,), inner_t, dtype=jnp.int32
            )

            return (
                obs,
                state,
                rewards,
                done,
                info,
            )

        def _reset_state(
            key: jnp.ndarray,
            episodes_completed: jnp.ndarray = jnp.int32(0),
        ) -> State:
            key, subkey = jax.random.split(key)

            # Find the free spaces in the grid
            grid = jnp.zeros((self.GRID_SIZE_ROW, self.GRID_SIZE_COL), jnp.int16)

            # all_positions = jax.random.permutation(subkey, self.SPAWNS)
            # player_reborn_pos = 
            # total_items = num_agents + NUM_COIN_TYPES * self.NUM_COINS

            inside_players_pos = jax.random.permutation(subkey, self.SPAWNS_PLAYER_IN)
            player_positions = jnp.concatenate((inside_players_pos, self.SPAWNS_PLAYERS))
            # Unique: both Q first, then distinct P cells (not Q∪P, which can
            # duplicate a Q already assigned).
            if num_agents == 1:
                agent_pos = jax.random.permutation(subkey, player_positions)[:1]
            else:
                agent_pos = jnp.concatenate(
                    (
                        inside_players_pos[:2],
                        jax.random.permutation(subkey, self.SPAWNS_PLAYERS)[
                            : num_agents - 2
                        ],
                    )
                )
            wall_pos = self.SPAWNS_WALL

            apple_pos = self.SPAWNS_APPLE
            apple_map = jnp.zeros(
                (self.GRID_SIZE_ROW, self.GRID_SIZE_COL), dtype=jnp.int16
            )
            apple_map = apple_map.at[apple_pos[:, 0], apple_pos[:, 1]].set(
                jnp.int16(1)
            )

            grid = grid.at[
                apple_pos[:, 0],
                apple_pos[:, 1]
            ].set(jnp.int16(Items.apple))

            grid = grid.at[
                wall_pos[:, 0],
                wall_pos[:, 1]
            ].set(jnp.int16(Items.wall))


            player_dir = jax.random.randint(
                subkey, shape=(
                    num_agents,
                    ), minval=0, maxval=3, dtype=jnp.int8
            )

            agent_locs = jnp.stack(
                [
                    agent_pos[:, 0].astype(jnp.int32),
                    agent_pos[:, 1].astype(jnp.int32),
                    player_dir.astype(jnp.int32),
                ],
                axis=1,
            )

            grid = grid.at[
                agent_locs[:, 0],
                agent_locs[:, 1]
            ].set(jnp.int16(self._agents))

            freeze = jnp.array(
                [[-1]*num_agents]*num_agents,
            dtype=jnp.int16
            )

            agent_active0 = jnp.ones((num_agents,), dtype=jnp.bool_)
            key, bandit_key = jax.random.split(key)
            bandit_fields, key = init_bandits_on_reset(
                bandit_key,
                cfg=self._bandit_cfg,
                episodes_completed=jnp.asarray(episodes_completed, dtype=jnp.int32),
                grid=grid,
                apples=apple_map,
                agent_locs=agent_locs,
                agent_active=agent_active0,
                num_agents=num_agents,
                wall_id=int(Items.wall),
                grid_h=self.GRID_SIZE_ROW,
                grid_w=self.GRID_SIZE_COL,
            )
            grid = paint_bandits_on_grid(
                grid,
                bandit_locs=bandit_fields["bandit_locs"],
                bandit_active=bandit_fields["bandit_active"],
                bandit_visual_id=bandit_fields["bandit_visual_id"],
                agent_item_base=len(Items),
            )

            return State(
                agent_locs=agent_locs,
                agent_invs=jnp.array([(0,0)]*num_agents, dtype=jnp.int8),
                inner_t=0,
                outer_t=0,
                grid=grid,
                apples=apple_map,

                freeze=freeze,
                reborn_locs = agent_locs,
                time_out=jnp.zeros((num_agents,), dtype=jnp.int16),
                timeout_from_agent=jnp.zeros((num_agents,), dtype=jnp.bool_),
                timeout_from_enforcement=jnp.zeros((num_agents,), dtype=jnp.bool_),
                timeout_from_police=jnp.zeros((num_agents,), dtype=jnp.bool_),
                tag_filter_rep=jnp.zeros((num_agents,), dtype=jnp.float32),
                bandit_locs=bandit_fields["bandit_locs"],
                bandit_active=bandit_fields["bandit_active"],
                bandit_respawn_cd=bandit_fields["bandit_respawn_cd"],
                bandit_visual_id=bandit_fields["bandit_visual_id"],
                episodes_completed=jnp.asarray(episodes_completed, dtype=jnp.int32),
            )

        def reset(
            key: jnp.ndarray,
            episodes_completed: jnp.ndarray = jnp.int32(0),
        ) -> Tuple[jnp.ndarray, State]:
            state = _reset_state(key, episodes_completed=episodes_completed)
            obs = _get_obs(state)
            return obs, state
        ################################################################################
        # if you want to test whether it can run on gpu, activate following code
        # overwrite Gymnax as it makes single-agent assumptions
        if jit:
            self.step_env = jax.jit(_step)
            self.reset = jax.jit(reset)
            self.get_obs_point = jax.jit(_get_obs_point)
            self.get_obs = jax.jit(_get_obs)
        else:
            # if you want to see values whilst debugging, don't jit
            self.step_env = _step
            self.reset = reset
            self.get_obs_point = _get_obs_point
            self.get_obs = _get_obs
        ################################################################################

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions,
        timestep: int = 0,
        reset_state: Optional[State] = None,
    ):
        """Step with SocialJax auto-reset, preserving the bandit schedule clock.

        Base ``MultiAgentEnv.step`` calls ``reset()`` whenever ``done["__all__"]``
        (true every inner episode when ``num_outer_steps=1``). That would zero
        ``episodes_completed`` and re-spawn the bandit anneal at t=0 forever.
        ``step_env`` already increments the clock on inner boundaries; we keep
        that value across the auto-reset and re-init bandits for it.
        """
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(
            key, state, actions, timestep
        )
        eps_keep = jnp.asarray(states_st.episodes_completed, dtype=jnp.int32)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset, episodes_completed=eps_keep)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            states_re,
            states_st,
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y),
            obs_re,
            obs_st,
        )
        return obs, states, rewards, dones, infos

    @property
    def name(self) -> str:
        """Environment name."""
        return "MGinTheGrid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self._num_actions

    def action_space(
        self, agent_id: Union[int, None] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self._num_actions)

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        _shape_obs = (
            (self.OBS_H, self.OBS_W, (len(Items)-1) + self._obs_extra_channels)
            if self.cnn
            else (self.OBS_H * self.OBS_W * ((len(Items)-1) + self._obs_extra_channels),)
        )

        return spaces.Box(
                low=0, high=1E9, shape=_shape_obs, dtype=jnp.uint8
            ), _shape_obs
    
    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (self.GRID_SIZE_ROW, self.GRID_SIZE_COL, NUM_TYPES + 4)
            if self.cnn
            else (self.GRID_SIZE_ROW* self.GRID_SIZE_COL * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)
    
    def render_tile(
        self,
        obj: int,
        agent_dir: Union[int, None] = None,
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache (plain Python ints — stable across steps).
        # Include agent RGB so recolouring Good/Bad does not require cache clears.
        obj_id = int(obj) if obj is not None else -1
        dir_id = int(agent_dir) if agent_dir is not None else -1
        if obj in self._agents:
            _ac = self.PLAYER_COLOURS[obj - len(Items)]
            color_key = (int(_ac[0]), int(_ac[1]), int(_ac[2]))
        else:
            color_key = (-1, -1, -1)
        key: tuple[int, ...] = (
            obj_id,
            int(agent_hat),
            int(highlight),
            tile_size,
            dir_id,
            *color_key,
        )

        if key in self.tile_cache:
            return self.tile_cache[key]

        img = onp.full(
            shape=(tile_size * subdivs, tile_size * subdivs, 3),
            fill_value=(210, 190, 140),
            dtype=onp.uint8,
        )


    # class Items(IntEnum):

        if obj in self._agents:
            # Draw the agent
            agent_color = self.PLAYER_COLOURS[obj-len(Items)]
        elif obj == Items.apple:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
            )
        # elif obj == Items.blue_coin:
        #     # Draw the blue coin as DEFECT/ RED COIN
        #     fill_coords(
        #         img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
        #     )
        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))

        elif obj == Items.spawn_point:
            # RGB-only marker (not in agent obs). Pale blue pad on sand floor.
            fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (150.0, 170.0, 205.0))

        elif obj == Items.inside_spawn_point:
            fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (70.0, 115.0, 185.0))

        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))

        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))

        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))

        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))

        # Overlay the agent on top
        if agent_dir is not None:
            if agent_hat:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                    0.3,
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(
                    tri_fn,
                    cx=0.5,
                    cy=0.5,
                    theta=0.5 * math.pi * (1 - agent_dir),
                )
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
                0.0,
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
            )
            fill_coords(img, tri_fn, agent_color)

        # # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        self.tile_cache[key] = img
        return img

    def render(
        self,
        state: State,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(self.GRID))

        # Compute the total grid size
        width_px = self.GRID.shape[1] * tile_size
        height_px = self.GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        grid = onp.array(state.grid)
        
        grid = onp.pad(
            grid, ((self.PADDING, self.PADDING), (self.PADDING, self.PADDING)), constant_values=Items.wall
        )
        for a in range(self.num_agents):
            startx, starty = self.get_obs_point(
                state.agent_locs[a]
            )
            # Highlight only the agent's own cell (the extraction square is now
            # large; highlighting all of it would wash out the whole frame).
            highlight_mask[
                startx + self.OBS_RADIUS, starty + self.OBS_RADIUS
            ] = True

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                    # Overlay P/Q that were stripped from the live grid at reset.
                    ri = i - self.PADDING
                    cj = j - self.PADDING
                    if (
                        0 <= ri < self.GRID_SIZE_ROW
                        and 0 <= cj < self.GRID_SIZE_COL
                    ):
                        if (ri, cj) in self._spawn_q_cells:
                            cell = int(Items.inside_spawn_point)
                        elif (ri, cj) in self._spawn_p_cells:
                            cell = int(Items.spawn_point)
                agent_here = []
                for a in self._agents:
                    agent_here.append(cell == a)
                # if cell in [1,2]:
                #     print(f'coordinates: {i},{j}')
                #     print(cell)

                agent_dir = None
                for a in range(self.num_agents):
                    if agent_here[a]:
                        agent_dir = int(state.agent_locs[a, 2])
                
                agent_hat = False
                # for a in range(self.num_agents):
                #     agent_hat = (
                #         bool(state.agent_invs[a].sum() > INTERACT_THRESHOLD)
                #         if agent_here[a]
                #         else agent_hat
                #     )

                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = i * tile_size
                ymax = (i + 1) * tile_size
                xmin = j * tile_size
                xmax = (j + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        img = onp.rot90(
            img[
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                (self.PADDING - 1) * tile_size : -(self.PADDING - 1) * tile_size,
                :,
            ],
            2,
        )

        # time = self.render_time(state, img.shape[1])
        # img = onp.concatenate((img, *agent_inv, time), axis=0)
        # img = onp.concatenate((img, time), axis=0)
        return img

    def render_inventory(self, inventory, width_px) -> onp.array:
        tile_height = 32
        height_px = NUM_COIN_TYPES * tile_height
        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // self.NUM_COINS
        for j in range(0, NUM_COIN_TYPES):
            num_coins = inventory[j]
            for i in range(int(num_coins)):
                cell = None
                if j == 0:
                    cell = 99
                elif j == 1:
                    cell = 100
                tile_img = self.render_tile(cell, tile_size=tile_height)
                ymin = j * tile_height
                ymax = (j + 1) * tile_height
                xmin = i * tile_width
                xmax = (i + 1) * tile_width
                img[ymin:ymax, xmin:xmax, :] = onp.resize(
                    tile_img, (tile_height, tile_width, 3)
                )
        return img

    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img
    
    def get_inequity_aversion_rewards_immediate(self, array, inner_t, target_agents=None, alpha=5, beta=0.05):
        """
        Calculate inequity aversion rewards using immediate rewards, based on equation (3) in the paper
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i^t for each agent
            target_agents: list of agent indices to apply inequity aversion
            alpha: inequity aversion coefficient (when other agents' rewards are greater than self)
            beta: inequity aversion coefficient (when self's rewards are greater than others)
        Returns:
            subjective_rewards: adjusted subjective rewards u_i^t after inequity aversion
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Calculate inequality using immediate rewards
        r_i = array  # [num_agents, 1]
        r_j = jnp.transpose(array)  # [1, num_agents]
        
        # Calculate inequality
        disadvantageous = jnp.maximum(r_j - r_i, 0)  # when other agents' rewards are higher
        advantageous = jnp.maximum(r_i - r_j, 0)     # when self's rewards are higher
        
        # Create mask to exclude self-comparison
        mask = 1 - jnp.eye(self.num_agents)
        disadvantageous = disadvantageous * mask
        advantageous = advantageous * mask
        
        # Calculate inequality penalty
        n_others = self.num_agents - 1
        inequity_penalty = (alpha * jnp.sum(disadvantageous, axis=1, keepdims=True) +
                           beta * jnp.sum(advantageous, axis=1, keepdims=True)) / n_others

        # Calculate subjective rewards u_i^t = r_i^t - inequality penalty
        subjective_rewards = array - inequity_penalty

        subjective_rewards = jnp.where(jnp.all(array == 0), -(alpha + beta) * n_others, subjective_rewards)
        
        # Apply inequity aversion only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, subjective_rewards, array),jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)
        else:
            return subjective_rewards,jnp.sum(disadvantageous, axis=1, keepdims=True),jnp.sum(advantageous, axis=1, keepdims=True)

    def get_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        
        Args:
            array: shape: [num_agents, 1] immediate rewards r_i for each agent
            w: SVO weight to balance self-reward and social value (0 <= w <= 1)
               w=0 means completely selfish, w=1 means completely altruistic
            ideal_angle_degrees: ideal angle in degrees
               - 45 degrees means complete equality
               - 0 degrees means completely selfish
               - 90 degrees means completely altruistic
            target_agents: list of agent indices to apply SVO
        
        Returns:
            shaped_rewards: rewards adjusted by SVO
            theta: reward angle in radians
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)  # [num_agents, num_agents]
        # Modified: use matrix multiplication to calculate other agents' rewards
        others_rewards = jnp.matmul(mask, array)  # [num_agents, 1]
        mean_others = others_rewards / (self.num_agents - 1)  # divide by number of other agents
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array  # [num_agents, 1]
        r_j = mean_others  # [num_agents, 1]
        theta = jnp.arctan2(r_j, r_i)
        
        # Calculate social value oriented utility
        # U(r_i, r_j) = r_i - w * |θ(R) - ideal_angle|
        angle_deviation = jnp.abs(theta - ideal_angle)
        svo_utility = r_i - self.num_agents * w * angle_deviation

        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)  # [num_agents, 1]
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta

    def get_standardized_svo_rewards(self, array, w=0.5, ideal_angle_degrees=45, target_agents=None):
        """
        Reward shaping function based on Social Value Orientation (SVO)
        """
        # Ensure correct input shape
        assert array.shape == (self.num_agents, 1), f"Expected shape ({self.num_agents}, 1), got {array.shape}"
        
        # Convert ideal angle from degrees to radians
        ideal_angle = (ideal_angle_degrees * jnp.pi) / 180.0
        
        # Calculate group average reward r_j (excluding self)
        mask = 1 - jnp.eye(self.num_agents)
        others_rewards = jnp.matmul(mask, array)
        mean_others = others_rewards / (self.num_agents - 1)
        
        # Calculate reward angle θ(R) = arctan(r_j / r_i)
        r_i = array
        r_j = mean_others
        theta = jnp.arctan2(r_j, r_i)
        
        # Convert angle to [0, 2π] range
        theta = (theta + 2 * jnp.pi) % (2 * jnp.pi)
        
        # Calculate angle deviation and normalize to [0, 1] range
        angle_deviation = jnp.abs(theta - ideal_angle)
        angle_deviation = jnp.minimum(angle_deviation, 2 * jnp.pi - angle_deviation)  # take minimum deviation
        normalized_deviation = angle_deviation / jnp.pi  # normalize to [0, 1]
        
        # Use multiplicative form of penalty instead of subtraction
        svo_utility = r_i * (1 - w * normalized_deviation)
        
        # Apply SVO only to target agents if specified
        if target_agents is not None:
            target_agents_array = jnp.array(target_agents)
            agent_mask = jnp.zeros(self.num_agents, dtype=bool)
            agent_mask = agent_mask.at[target_agents_array].set(True)
            agent_mask = agent_mask.reshape(-1, 1)
            return jnp.where(agent_mask, svo_utility, array), theta
        else:
            return svo_utility, theta