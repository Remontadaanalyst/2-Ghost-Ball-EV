"""
REMONTADA TACTICAL EV ENGINE - Clean Slate Visualizer
Features: Decoupled YOLO bounding boxes, strict layering (Rings -> Boxes), and Hovering Ghost Ball.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
from remontada_engine import EngineOutput


class RemontadaOverlay:
    def __init__(
        self,
        view_transformer: Optional[Any] = None,
        pitch_length: float = 104.0,
        pitch_width: float = 68.0,
        enable_minimap: bool = True,
        minimap_scale: float = 0.22,
    ):
        self.view_transformer = view_transformer
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.enable_minimap = enable_minimap
        self.minimap_scale = minimap_scale

        self.frame_idx = 0

        # Broadcast Palette (BGR)
        self.COLOR_ATTACK_PRIMARY = (255, 200, 0)     # Electric Cyan / Sky Blue
        self.COLOR_DEFENSE_PRIMARY = (30, 35, 240)    # Crimson Red
        self.COLOR_TRIPWIRE_CLEAR = (0, 230, 255)     # Radiant Gold Laser
        self.COLOR_TRIPWIRE_BLOCKED = (30, 30, 255)   # Warning Red
        self.COLOR_GHOST_BALL = (255, 255, 255)       # Pure Phosphor White
        self.COLOR_CHECKMARK = (50, 255, 120)         # Emerald Green
        self.COLOR_THETA = (0, 215, 255)              # Tactical Gold

    def pitch_to_pixel(self, pitch_x: float, pitch_y: float, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if self.view_transformer is not None:
            try:
                if hasattr(self.view_transformer, "transform_pitch_to_pixel"):
                    px = self.view_transformer.transform_pitch_to_pixel(pitch_x, pitch_y)
                    if px is not None:
                        return int(px[0]), int(px[1])
                elif hasattr(self.view_transformer, "inverse_homography"):
                    pt = np.array([[[pitch_x, pitch_y]]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(pt, self.view_transformer.inverse_homography)
                    return int(transformed[0, 0, 0]), int(transformed[0, 0, 1])
            except Exception:
                pass
        return None

    def _draw_player_box_and_ripples(
        self, overlay: np.ndarray, frame: np.ndarray, center_px: Tuple[int, int], 
        is_attack: bool, dominance: float, bbox: List[float]
    ) -> None:
        """Draws grass rings anchored to true feet FIRST, then thick YOLO brackets over them."""
        color = self.COLOR_ATTACK_PRIMARY if is_attack else self.COLOR_DEFENSE_PRIMARY

        # Calculate exact feet pixels from the raw YOLO bounding box
        x1, y1, x2, y2 = map(int, bbox)
        feet_x = (x1 + x2) // 2
        feet_y = y2

        # 1. WIDE SONAR RIPPLES (Background Layer on Grass, Anchored to YOLO Feet)
        pulse_phase = (self.frame_idx * 0.08) % 1.0
        for r_i in range(3):
            ripple_offset = (pulse_phase + (r_i * 0.33)) % 1.0
            rx = int(18 + (dominance * 10) + (ripple_offset * 25))
            ry = int(8 + (dominance * 5) + (ripple_offset * 12))
            cv2.ellipse(overlay, (feet_x, feet_y), (rx, ry), 0, 0, 360, color, 1, cv2.LINE_AA)

        # 2. TRUE BODY-HUGGING BRACKETS (Foreground Layer)
        arm = 12  
        thick = 3 

        # Top Brackets
        cv2.line(frame, (x1, y1), (x1 + arm, y1), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + arm), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2 - arm, y1), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + arm), color, thick, cv2.LINE_AA)
        # Bottom Brackets
        cv2.line(frame, (x1, y2), (x1 + arm, y2), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - arm), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2 - arm, y2), color, thick, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - arm), color, thick, cv2.LINE_AA)

    def _draw_shatter_effect(self, frame: np.ndarray, block_px: Tuple[int, int]) -> None:
        bx, by = block_px
        shard_angles = [0, 35, 75, 120, 165, 210, 255, 305, 340]
        shard_lengths = [16, 22, 18, 25, 14, 20, 26, 17, 21]

        for ang, length in zip(shard_angles, shard_lengths):
            rad = math.radians(ang)
            ex = int(bx + length * math.cos(rad))
            ey = int(by + length * math.sin(rad))
            cv2.line(frame, (bx, by), (ex, ey), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.line(frame, (bx, by), (ex, ey), (255, 255, 255), 1, cv2.LINE_AA)

        cv2.circle(frame, (bx, by), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 12, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.rectangle(frame, (bx - 32, by - 24), (bx + 32, by - 10), (15, 15, 20), -1)
        cv2.rectangle(frame, (bx - 32, by - 24), (bx + 32, by - 10), (0, 0, 255), 1)
        cv2.putText(frame, "BLOCKED", (bx - 26, by - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 50, 255), 1, cv2.LINE_AA)

    def _draw_floating_ghost_ball(self, overlay: np.ndarray, frame: np.ndarray, target_px: Tuple[int, int], max_ev_val: float) -> None:
        """Hovering Ghost Ball (No rocket line, just pulsing in the pocket)."""
        tx, ty = target_px
        pulse = 0.5 + 0.5 * math.sin(self.frame_idx * 0.25)
        target_radius = int(22 + 6 * pulse)

        # 1. Target Pocket Concentric Sonar Rings
        cv2.circle(overlay, (tx, ty), target_radius, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.circle(overlay, (tx, ty), int(target_radius * 0.55), (50, 255, 120), -1, cv2.LINE_AA)

        # 2. Glowing Ghost Ball Orb Floating in Pocket
        cv2.circle(overlay, (tx, ty), 9, self.COLOR_GHOST_BALL, -1, cv2.LINE_AA)
        cv2.circle(overlay, (tx, ty), 14, (0, 240, 255), 2, cv2.LINE_AA)

        # 3. Emerald Green Checkmark
        chk_start, chk_mid, chk_end = (tx - 7, ty), (tx - 2, ty + 5), (tx + 8, ty - 6)
        cv2.line(frame, chk_start, chk_mid, (15, 30, 15), 4, cv2.LINE_AA)
        cv2.line(frame, chk_mid, chk_end, (15, 30, 15), 4, cv2.LINE_AA)
        cv2.line(frame, chk_start, chk_mid, self.COLOR_CHECKMARK, 2, cv2.LINE_AA)
        cv2.line(frame, chk_mid, chk_end, self.COLOR_CHECKMARK, 2, cv2.LINE_AA)

        # 4. EV Value Badge
        cv2.putText(frame, f"EV +{max_ev_val:.2f}", (tx - 26, ty + target_radius + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)

    def render(
        self, frame: np.ndarray, output: EngineOutput, player_coords: np.ndarray, 
        player_velocities: np.ndarray, player_teams: np.ndarray, player_ids: np.ndarray, 
        attacking_team: int, fps: float = 25.0, player_bboxes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        self.frame_idx += 1
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Render Tactical Player Boxes & Gaussian Rings
        for idx, (pt, team) in enumerate(zip(player_coords, player_teams)):
            px = self.pitch_to_pixel(pt[0], pt[1], (h, w))
            if px is None or not (0 <= px[0] < w and 0 <= px[1] < h):
                continue

            gx, gy = int(np.clip(pt[0], 0, self.pitch_length - 1)), int(np.clip(pt[1], 0, self.pitch_width - 1))
            control_val = output.pitch_control[gx, gy]
            is_att = (team == attacking_team)
            dominance = control_val if is_att else (1.0 - control_val)

            # Use raw YOLO bbox if provided by the adapter; otherwise construct a safe local bounding box
            if player_bboxes is not None and len(player_bboxes) > idx:
                bbox = player_bboxes[idx]
            else:
                bbox = [px[0] - 20, px[1] - 50, px[0] + 20, px[1]]

            self._draw_player_box_and_ripples(overlay, frame, px, is_att, dominance, bbox)

        # 2. Key Reference Points & Ray-Cast Laser
        ball_px = self.pitch_to_pixel(output.ball_pos[0], output.ball_pos[1], (h, w))
        target_px = self.pitch_to_pixel(output.max_ev_coord[0], output.max_ev_coord[1], (h, w))
        
        if ball_px is not None:
            if output.tripwire_blocked and output.tripwire_block_point is not None:
                block_px = self.pitch_to_pixel(output.tripwire_block_point[0], output.tripwire_block_point[1], (h, w))
                if block_px:
                    cv2.line(overlay, ball_px, block_px, self.COLOR_TRIPWIRE_BLOCKED, 5, cv2.LINE_AA)
                    cv2.line(overlay, ball_px, block_px, (255, 255, 255), 2, cv2.LINE_AA)
                    self._draw_shatter_effect(frame, block_px)

        # 3. Floating Ghost Ball (No Rocket Tether)
        if target_px is not None:
            self._draw_floating_ghost_ball(overlay, frame, target_px, output.max_ev_value)

        # Blend Translucent Overlays
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Highlight Real Ball
        if ball_px is not None:
            cv2.circle(frame, ball_px, 8, (0, 240, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, ball_px, 3, (255, 255, 255), -1, cv2.LINE_AA)

        return frame