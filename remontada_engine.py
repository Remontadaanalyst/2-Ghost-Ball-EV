"""
REMONTADA TACTICAL EV ENGINE - Continuous Off-Ball Scoring Opportunity Model
Hyper-Optimized Pure NumPy Vectorized Physics & Spatial Reasoning Engine
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class EngineOutput:
    """Structured tactical output exported for every frame."""
    pitch_control: np.ndarray          # Shape (104, 68) control probability [0.0, 1.0]
    expected_threat: np.ndarray        # Shape (104, 68) active xT matrix
    expected_value: np.ndarray         # Shape (104, 68) EMA-smoothed EV grid
    max_ev_coord: Tuple[float, float]  # (X, Y) pitch coordinates of max EV target (meters)
    max_ev_value: float                # Maximum EV value at target
    pass_feasibility: float            # Line-of-sight pass feasibility score [0.0, 1.0]
    pressure_metric: float             # Active localized pressure on ball carrier [0.0, 100.0]
    offside_line_x: float              # X-coordinate (meters) of second-last defender
    ball_carrier_id: Optional[int]     # Player ID currently in possession
    ball_pos: Tuple[float, float]      # (X, Y) ball position in meters
    tripwire_blocked: bool             # True if ray to max EV is blocked by defender
    tripwire_block_point: Optional[Tuple[float, float]] # Pitch coords of tripwire impact
    blocked_defender_id: Optional[int] # Defender ID blocking the passing lane
    biomechanical_traps: Dict[int, float] # Defender ID -> orientation turnaround penalty (seconds)


class ContinuousEVEngine:
    """
    ContinuousEVEngine: Hyper-optimized, production-ready, vectorized Python engine
    for continuous off-ball scoring opportunity evaluation.
    
    Zero nested loops over the 104x68 pitch grid (7,072 cells).
    Utilizes flattened contiguous memory arrays and SIMD-friendly broadcasting for >60 FPS performance.
    """

    def __init__(
        self,
        pitch_length: float = 104.0,
        pitch_width: float = 68.0,
        grid_res_x: int = 104,
        grid_res_y: int = 68,
        ball_velocity_scalar: float = 15.0,  # Flat pass speed in m/s
        max_sprint_speed: float = 8.5,       # Outfield sprint speed in m/s
        reaction_time: float = 0.3,          # Outfield reaction time baseline in seconds
        gk_reaction_time: float = 0.15,      # Goalkeeper reaction time in seconds
        gk_dive_speed: float = 10.0,         # Goalkeeper explosive dive speed in m/s
        turnaround_penalty_max: float = 0.85,# Maximum orientation penalty in seconds
        sigmoid_slope: float = 3.2,          # Steepness of logistic probability wave
        defender_reach_radius: float = 1.15, # Defender interception radius in meters
        ema_alpha: float = 0.1818,           # ~10-frame EMA smoothing alpha (2 / (10 + 1))
        attack_direction: int = 1,           # +1 for attacking towards X=104, -1 for X=0
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.grid_res_x = grid_res_x
        self.grid_res_y = grid_res_y
        self.n_grid = grid_res_x * grid_res_y
        self.ball_v = ball_velocity_scalar
        self.v_max = max_sprint_speed
        self.t_react = reaction_time
        self.gk_t_react = gk_reaction_time
        self.gk_v_dive = gk_dive_speed
        self.tau_turn = turnaround_penalty_max
        self.sigmoid_slope = sigmoid_slope
        self.r_reach = defender_reach_radius
        self.ema_alpha = ema_alpha
        self.attack_direction = attack_direction

        # Phase 1: Grid Initialization
        # Pitch coordinates: X in [0, 104], Y in [0, 68]
        self.x_bins = np.linspace(0.5, pitch_length - 0.5, grid_res_x, dtype=np.float32)
        self.y_bins = np.linspace(0.5, pitch_width - 0.5, grid_res_y, dtype=np.float32)
        self.grid_x, self.grid_y = np.meshgrid(self.x_bins, self.y_bins, indexing="ij")
        
        # Flattened grid for high-speed cache-friendly SIMD broadcasting
        self.grid_flat_x = self.grid_x.ravel() # Shape (7072,)
        self.grid_flat_y = self.grid_y.ravel() # Shape (7072,)
        self.grid_flat = np.column_stack([self.grid_flat_x, self.grid_flat_y]) # Shape (7072, 2)

        # Baseline Expected Threat (xT) Positional Matrix
        self.base_xt = self._initialize_xt_matrix()

        # State for temporal smoothing
        self.prev_ev: Optional[np.ndarray] = None
        self.prev_pitch_control: Optional[np.ndarray] = None

    def _initialize_xt_matrix(self) -> np.ndarray:
        """
        Generates a continuous, calibrated Expected Threat (xT) positional matrix
        interpolated across the 104x68 grid.
        """
        x_norm = self.grid_x / self.pitch_length
        goal_x = self.pitch_length if self.attack_direction == 1 else 0.0
        goal_y = self.pitch_width / 2.0

        dist_to_goal_sq = (self.grid_x - goal_x) ** 2 + (self.grid_y - goal_y) ** 2
        dist_to_center_y_sq = (self.grid_y - goal_y) ** 2

        progression_threat = 0.005 + 0.18 * (x_norm ** 2.2 if self.attack_direction == 1 else (1.0 - x_norm) ** 2.2)
        box_bonus = 0.22 * np.exp(- dist_to_goal_sq / (2.0 * (16.0 ** 2)))
        central_bonus = 0.05 * np.exp(- dist_to_center_y_sq / (2.0 * (14.0 ** 2)))

        xt = progression_threat + box_bonus + central_bonus
        return np.clip(xt, 0.003, 0.40).astype(np.float32)

    def reset_state(self) -> None:
        """Reset temporal filters between distinct clips or game segments."""
        self.prev_ev = None
        self.prev_pitch_control = None

    def set_attack_direction(self, direction: int) -> None:
        """
        Update the attacking direction at runtime (e.g. at half-time, when the
        attacking_team's target goal flips from X=104 to X=0 or vice versa).

        Recomputes the baseline xT matrix, since it is orientation-dependent,
        and resets temporal smoothing state, since EMA-blending pitch
        control/EV grids computed under the previous direction would corrupt
        the new grid's spatial meaning.
        """
        if direction not in (1, -1):
            raise ValueError(f"attack_direction must be 1 or -1, got {direction}")
        self.attack_direction = direction
        self.base_xt = self._initialize_xt_matrix()
        self.reset_state()

    def process_frame(
        self,
        player_coords: np.ndarray,      # Shape (N, 2): (X, Y) in pitch meters
        player_velocities: np.ndarray,  # Shape (N, 2): (Vx, Vy) in m/s
        player_teams: np.ndarray,       # Shape (N,): team identifier (e.g. 0 or 1)
        player_ids: np.ndarray,         # Shape (N,): player tracking IDs
        is_goalkeeper: np.ndarray,      # Shape (N,): bool mask for goalkeepers
        ball_coord: np.ndarray,         # Shape (2,): (X, Y) in pitch meters
        attacking_team: int,            # Active attacking team ID
        pass_release: bool = False,     # True when an active pass release event occurs
    ) -> EngineOutput:
        """
        Execute the 5-phase tactical EV physics algorithm for a single frame.
        Guaranteed zero nested loops over the 7,072 grid points.
        """
        num_players = len(player_coords)
        if num_players == 0:
            zero_grid = np.zeros((self.grid_res_x, self.grid_res_y), dtype=np.float32)
            return EngineOutput(
                pitch_control=zero_grid,
                expected_threat=self.base_xt,
                expected_value=zero_grid,
                max_ev_coord=(self.pitch_length / 2, self.pitch_width / 2),
                max_ev_value=0.0,
                pass_feasibility=1.0,
                pressure_metric=0.0,
                offside_line_x=self.pitch_length / 2,
                ball_carrier_id=None,
                ball_pos=(float(ball_coord[0]), float(ball_coord[1])),
                tripwire_blocked=False,
                tripwire_block_point=None,
                blocked_defender_id=None,
                biomechanical_traps={},
            )

        # -------------------------------------------------------------
        # Identify Ball Carrier & Team Masks
        # -------------------------------------------------------------
        dists_to_ball_sq = (player_coords[:, 0] - ball_coord[0]) ** 2 + (player_coords[:, 1] - ball_coord[1]) ** 2
        ball_carrier_idx = int(np.argmin(dists_to_ball_sq))
        has_possession = dists_to_ball_sq[ball_carrier_idx] < (1.75 ** 2)
        ball_carrier_id = int(player_ids[ball_carrier_idx]) if has_possession else None

        att_mask = (player_teams == attacking_team)
        def_mask = ~att_mask

        # -------------------------------------------------------------
        # Phase 2: Game State & The True Offside Rule
        # -------------------------------------------------------------
        def_coords = player_coords[def_mask]
        if len(def_coords) >= 2:
            if self.attack_direction == 1:
                sorted_def_x = np.sort(def_coords[:, 0])[::-1]
                second_last_def_x = float(sorted_def_x[1])
                offside_line_x = max(second_last_def_x, float(ball_coord[0]), self.pitch_length / 2.0)
                is_offside_flat = self.grid_flat_x > offside_line_x
            else:
                sorted_def_x = np.sort(def_coords[:, 0])
                second_last_def_x = float(sorted_def_x[1])
                offside_line_x = min(second_last_def_x, float(ball_coord[0]), self.pitch_length / 2.0)
                is_offside_flat = self.grid_flat_x < offside_line_x
        else:
            offside_line_x = self.pitch_length if self.attack_direction == 1 else 0.0
            is_offside_flat = np.zeros(self.n_grid, dtype=bool)

        active_xt = self.base_xt.copy()
        if pass_release:
            active_xt[is_offside_flat.reshape(self.grid_res_x, self.grid_res_y)] = 0.0

        # -------------------------------------------------------------
        # Phase 3: Continuous Physics Engine (Kinematics & Sigmoid Waves)
        # -------------------------------------------------------------
        # Ball arrival time: t_ball = dist / 15.0 m/s
        ball_dx = self.grid_flat_x - ball_coord[0]
        ball_dy = self.grid_flat_y - ball_coord[1]
        dist_ball_sq = ball_dx * ball_dx + ball_dy * ball_dy
        dist_ball = np.sqrt(dist_ball_sq)
        t_ball = dist_ball / self.ball_v # Shape (7072,)

        # Player-to-grid calculations (N, 7072)
        dx_p = self.grid_flat_x[None, :] - player_coords[:, 0:1] # (N, 7072)
        dy_p = self.grid_flat_y[None, :] - player_coords[:, 1:2] # (N, 7072)
        dist_p_sq = dx_p * dx_p + dy_p * dy_p
        dist_p = np.sqrt(dist_p_sq) # (N, 7072)

        # Sprint speed and reaction time
        v_player = np.where(is_goalkeeper[:, None], self.gk_v_dive, self.v_max)
        t_sprint = dist_p / v_player
        t_react_arr = np.where(is_goalkeeper[:, None], self.gk_t_react, self.t_react)

        # Orientation Turnaround Cost
        vx = player_velocities[:, 0:1]
        vy = player_velocities[:, 1:2]
        v_speed = np.sqrt(vx * vx + vy * vy) # (N, 1)

        dot_prod = vx * dx_p + vy * dy_p # (N, 7072)
        cos_theta = dot_prod / (v_speed * dist_p + 1e-6)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        raw_turn_penalty = (self.tau_turn * 0.5) * (1.0 - cos_theta)
        speed_factor = np.clip(v_speed / 0.4, 0.0, 1.0)
        orient_penalty = raw_turn_penalty * speed_factor

        # Detect Biomechanical Traps for defenders
        biomechanical_traps: Dict[int, float] = {}
        for idx in np.where(def_mask)[0]:
            pid = int(player_ids[idx])
            v_vec = player_velocities[idx]
            spd = float(v_speed[idx, 0])
            if spd > 1.2:
                to_ball_x = ball_coord[0] - player_coords[idx, 0]
                to_ball_y = ball_coord[1] - player_coords[idx, 1]
                dist_to_ball_val = math.hypot(to_ball_x, to_ball_y) + 1e-6
                cos_ang = (v_vec[0] * to_ball_x + v_vec[1] * to_ball_y) / (spd * dist_to_ball_val)
                if cos_ang < -0.25: # Angle > 105 degrees
                    biomechanical_traps[pid] = float(self.tau_turn * 0.5 * (1.0 - cos_ang))

        # Total time to intercept
        t_player = t_react_arr + t_sprint + orient_penalty # (N, 7072)
        delta_t = t_ball[None, :] - t_player # (N, 7072)

        # Sigmoid probability waves
        p_waves = 1.0 / (1.0 + np.exp(-self.sigmoid_slope * delta_t)) # (N, 7072)

        # Hardcoded constraints: Ball carrier 1m control, GK 3m control
        if has_possession:
            p_waves[ball_carrier_idx, dist_p[ball_carrier_idx] <= 1.0] = 1.0
        
        gk_indices = np.where(is_goalkeeper)[0]
        for gk_i in gk_indices:
            p_waves[gk_i, dist_p[gk_i] <= 3.0] = 1.0

        # -------------------------------------------------------------
        # Phase 4: Ray-Casting & Spatial Interference (Vectorized)
        # -------------------------------------------------------------
        # Ray vector from ball to all 7,072 grid points
        U_x = ball_dx # (7072,)
        U_y = ball_dy # (7072,)
        U_sq = dist_ball_sq + 1e-6 # (7072,)

        def_indices = np.where(def_mask)[0]
        if len(def_indices) > 0:
            def_pos = player_coords[def_indices] # (M_def, 2)
            W_x = def_pos[:, 0:1] - ball_coord[0] # (M_def, 1)
            W_y = def_pos[:, 1:2] - ball_coord[1] # (M_def, 1)

            # Dot product W · U: (M_def, 7072)
            dot_WU = W_x * U_x[None, :] + W_y * U_y[None, :]
            t_proj = dot_WU / U_sq[None, :] # (M_def, 7072)
            t_clamped = np.clip(t_proj, 0.0, 1.0)

            # Closest point C on ray segment:
            C_x = ball_coord[0] + t_clamped * U_x[None, :]
            C_y = ball_coord[1] + t_clamped * U_y[None, :]

            # Perpendicular distance squared
            dist_perp_sq = (def_pos[:, 0:1] - C_x) ** 2 + (def_pos[:, 1:2] - C_y) ** 2

            # Lane obstruction mask
            in_lane = (t_proj > 0.05) & (t_proj < 0.98)
            sigma_block_sq = 2.0 * ((self.r_reach / 1.6) ** 2)

            block_factor = np.exp(- dist_perp_sq / sigma_block_sq)
            clearance_per_def = 1.0 - (block_factor * in_lane.astype(np.float32))
            total_pass_feasibility = np.prod(clearance_per_def, axis=0) # (7072,)
        else:
            total_pass_feasibility = np.ones(self.n_grid, dtype=np.float32)

        # Attacking & Defending probability wave sums
        p_att_waves = p_waves[att_mask] # (N_att, 7072)
        p_def_waves = p_waves[def_mask] # (N_def, 7072)

        if len(p_att_waves) > 0:
            p_att_degraded = p_att_waves * total_pass_feasibility[None, :]
            sum_att = np.sum(p_att_degraded, axis=0)
        else:
            sum_att = np.zeros(self.n_grid, dtype=np.float32)

        if len(p_def_waves) > 0:
            sum_def = np.sum(p_def_waves, axis=0)
        else:
            sum_def = np.zeros(self.n_grid, dtype=np.float32)

        pitch_control_flat = np.clip(sum_att / (sum_att + sum_def + 1e-6), 0.0, 1.0)
        pitch_control = pitch_control_flat.reshape((self.grid_res_x, self.grid_res_y))

        # -------------------------------------------------------------
        # Phase 5: Tactical Output & Temporal Smoothing
        # -------------------------------------------------------------
        # 1. Localized Pressure Metric around ball carrier (defenders within 5m)
        pressure_val = 0.0
        if len(def_indices) > 0:
            def_pos = player_coords[def_indices]
            def_vels = player_velocities[def_indices]
            dists_to_carrier = np.hypot(def_pos[:, 0] - ball_coord[0], def_pos[:, 1] - ball_coord[1])
            in_pressure_ring = dists_to_carrier <= 5.0
            if np.any(in_pressure_ring):
                press_dists = dists_to_carrier[in_pressure_ring]
                press_pos = def_pos[in_pressure_ring]
                press_vels = def_vels[in_pressure_ring]

                to_carrier_ux = (ball_coord[0] - press_pos[:, 0]) / (press_dists + 1e-6)
                to_carrier_uy = (ball_coord[1] - press_pos[:, 1]) / (press_dists + 1e-6)
                closing_speeds = np.maximum(0.0, press_vels[:, 0] * to_carrier_ux + press_vels[:, 1] * to_carrier_uy)

                press_contributions = (1.0 + 0.6 * closing_speeds) / (press_dists + 0.4)
                pressure_val = min(100.0, float(np.sum(press_contributions) * 12.5))

        # 2. Dynamic EV Boost for safe backward passing zones under pressure
        safe_pass_boost = np.zeros(self.n_grid, dtype=np.float32)
        if pressure_val > 15.0:
            press_intensity = (pressure_val - 15.0) / 85.0
            if self.attack_direction == 1:
                backward_mask = self.grid_flat_x < (ball_coord[0] - 2.0)
            else:
                backward_mask = self.grid_flat_x > (ball_coord[0] + 2.0)

            lat_sq = (self.grid_flat_y - ball_coord[1]) ** 2
            lat_gaussian = np.exp(- lat_sq / (2.0 * (15.0 ** 2)))
            safe_pass_boost = (0.12 * press_intensity * pitch_control_flat * lat_gaussian * backward_mask.astype(np.float32))

        # 3. Raw EV calculation
        raw_ev = (pitch_control_flat * active_xt.ravel()) + safe_pass_boost
        raw_ev = raw_ev.reshape((self.grid_res_x, self.grid_res_y))

        # 4. Temporal Smoothing via EMA Filter
        if self.prev_ev is None:
            smoothed_ev = raw_ev
            self.prev_pitch_control = pitch_control
        else:
            smoothed_ev = (self.ema_alpha * raw_ev) + ((1.0 - self.ema_alpha) * self.prev_ev)
            pitch_control = (self.ema_alpha * pitch_control) + ((1.0 - self.ema_alpha) * self.prev_pitch_control)

        self.prev_ev = smoothed_ev
        self.prev_pitch_control = pitch_control

        # -------------------------------------------------------------
        # Maximum EV Target Extraction & Tripwire Ray Blocking
        # -------------------------------------------------------------
        carrier_exclusion = (dist_ball_sq < (2.5 ** 2)).reshape(self.grid_res_x, self.grid_res_y)
        eval_ev = smoothed_ev.copy()
        eval_ev[carrier_exclusion] = -1.0

        max_idx = np.unravel_index(np.argmax(eval_ev), eval_ev.shape)
        max_ev_x = float(self.x_bins[max_idx[0]])
        max_ev_y = float(self.y_bins[max_idx[1]])
        max_ev_val = float(smoothed_ev[max_idx])
        feasibility_at_target = float(total_pass_feasibility.reshape(self.grid_res_x, self.grid_res_y)[max_idx])

        # Laser Tripwire evaluation
        tripwire_blocked = False
        tripwire_block_pt: Optional[Tuple[float, float]] = None
        blocked_def_id: Optional[int] = None

        if len(def_indices) > 0:
            target_pt = np.array([max_ev_x, max_ev_y], dtype=np.float32)
            ray_seg = target_pt - ball_coord
            ray_len_sq = float(np.dot(ray_seg, ray_seg)) + 1e-6
            
            def_pos = player_coords[def_indices]
            w_vecs = def_pos - ball_coord[None, :]
            t_vals = np.sum(w_vecs * ray_seg[None, :], axis=1) / ray_len_sq
            t_clamped = np.clip(t_vals, 0.0, 1.0)
            closest_pts = ball_coord[None, :] + t_clamped[:, None] * ray_seg[None, :]
            dists_to_ray = np.hypot(def_pos[:, 0] - closest_pts[:, 0], def_pos[:, 1] - closest_pts[:, 1])

            interceptors = np.where((t_vals > 0.08) & (t_vals < 0.95) & (dists_to_ray < self.r_reach))[0]
            if len(interceptors) > 0:
                first_interceptor = interceptors[np.argmin(t_vals[interceptors])]
                tripwire_blocked = True
                def_real_idx = def_indices[first_interceptor]
                blocked_def_id = int(player_ids[def_real_idx])
                tripwire_block_pt = (float(def_pos[first_interceptor, 0]), float(def_pos[first_interceptor, 1]))

        return EngineOutput(
            pitch_control=pitch_control,
            expected_threat=active_xt,
            expected_value=smoothed_ev,
            max_ev_coord=(max_ev_x, max_ev_y),
            max_ev_value=max_ev_val,
            pass_feasibility=feasibility_at_target,
            pressure_metric=pressure_val,
            offside_line_x=offside_line_x,
            ball_carrier_id=ball_carrier_id,
            ball_pos=(float(ball_coord[0]), float(ball_coord[1])),
            tripwire_blocked=tripwire_blocked,
            tripwire_block_point=tripwire_block_pt,
            blocked_defender_id=blocked_def_id,
            biomechanical_traps=biomechanical_traps,
        )