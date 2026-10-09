"""
REMONTADA TACTICAL EV ENGINE - Seamless Integration into BasicTrackingPipeline
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from remontada_engine import ContinuousEVEngine, EngineOutput
from remontada_overlay import RemontadaOverlay


class RemontadaPipelineAdapter:
    """
    Adapter that consumes the raw `tracks` dictionary frame-by-frame from BasicTrackingPipeline,
    extracts NumPy arrays for player positions, velocities, ball coordinates, and raw bboxes,
    and executes ContinuousEVEngine + RemontadaOverlay in real-time.
    """

    def __init__(
        self,
        engine: Optional[ContinuousEVEngine] = None,
        overlay: Optional[RemontadaOverlay] = None,
        view_transformer: Optional[Any] = None,
        attacking_team: int = 1,
    ):
        self.engine = engine or ContinuousEVEngine()
        self.overlay = overlay or RemontadaOverlay(view_transformer=view_transformer)
        self.attacking_team = attacking_team

    def set_attack_direction(self, direction: int) -> None:
        """Call this at half-time (once you know the second-half kickoff frame_num)
        to flip which goal attacking_team is now shooting towards."""
        self.engine.set_attack_direction(direction)

    def extract_frame_arrays(
        self,
        tracks: Dict[str, Any],
        frame_num: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Safe retrieval for either list- or dict-shaped tracking structures.
        players_data = tracks.get("players", {})
        ball_data = tracks.get("ball", {})

        if isinstance(players_data, list):
            player_frame = players_data[frame_num] if 0 <= frame_num < len(players_data) else {}
        else:
            player_frame = players_data.get(frame_num, {})

        if isinstance(ball_data, list):
            ball_frame = ball_data[frame_num] if 0 <= frame_num < len(ball_data) else {}
        else:
            ball_frame = ball_data.get(frame_num, {})

        # Extract Ball Coordinate (meters)
        if isinstance(ball_frame, dict):
            ball_pos = ball_frame.get("transformed_position")
            if ball_pos is None:
                ball_pos = ball_frame.get("position_transformed")
            if ball_pos is None:
                ball_pos = ball_frame.get("position", [52.0, 34.0])
            ball_coord = np.array(ball_pos[:2], dtype=np.float32)
        elif hasattr(ball_frame, "transformed_position"):
            ball_coord = np.array(ball_frame.transformed_position[:2], dtype=np.float32)
        elif hasattr(ball_frame, "position"):
            ball_coord = np.array(ball_frame.position[:2], dtype=np.float32)
        else:
            ball_coord = np.array([52.0, 34.0], dtype=np.float32)

        coords_list, vels_list, teams_list, ids_list, gk_list, bboxes_list = [], [], [], [], [], []

        for p_id, track_info in player_frame.items():
            if isinstance(track_info, dict):
                pos = track_info.get("transformed_position")
                if pos is None:
                    pos = track_info.get("position_transformed")
                if pos is None:
                    pos = track_info.get("position", [0.0, 0.0])
                vx = track_info.get("speed_x", 0.0)
                vy = track_info.get("speed_y", 0.0)
                team = track_info.get("team", 0)
                is_goalkeeper = track_info.get("is_goalkeeper", False)
                bbox = track_info.get("bbox", [0.0, 0.0, 0.0, 0.0])
            else:
                pos = getattr(
                    track_info,
                    "transformed_position",
                    getattr(track_info, "position_transformed", getattr(track_info, "position", [0.0, 0.0])),
                )
                vx = getattr(track_info, "speed_x", 0.0)
                vy = getattr(track_info, "speed_y", 0.0)
                team = getattr(track_info, "team", 0)
                is_goalkeeper = getattr(track_info, "is_goalkeeper", False)
                bbox = getattr(track_info, "bbox", [0.0, 0.0, 0.0, 0.0])

            coords_list.append([float(pos[0]), float(pos[1])])
            vels_list.append([float(vx), float(vy)])
            teams_list.append(int(team) if team is not None else 0)
            ids_list.append(int(p_id))
            gk_list.append(bool(is_goalkeeper))
            bboxes_list.append(bbox)

        if coords_list:
            player_coords = np.array(coords_list, dtype=np.float32)
            player_vels = np.array(vels_list, dtype=np.float32)
            player_teams = np.array(teams_list, dtype=np.int32)
            player_ids = np.array(ids_list, dtype=np.int64)
            is_gk = np.array(gk_list, dtype=bool)
            player_bboxes = np.array(bboxes_list, dtype=np.float32)
        else:
            player_coords = np.empty((0, 2), dtype=np.float32)
            player_vels = np.empty((0, 2), dtype=np.float32)
            player_teams = np.empty((0,), dtype=np.int32)
            player_ids = np.empty((0,), dtype=np.int64)
            is_gk = np.empty((0,), dtype=bool)
            player_bboxes = np.empty((0, 4), dtype=np.float32)

        return player_coords, player_vels, player_teams, player_ids, is_gk, ball_coord, player_bboxes

    def process_frame(
        self,
        frame_img: np.ndarray,
        tracks: Dict[str, Any],
        frame_num: int,
        pass_release: bool = False,
        fps: float = 25.0,
    ) -> Tuple[np.ndarray, EngineOutput]:
        (
            player_coords,
            player_vels,
            player_teams,
            player_ids,
            is_gk,
            ball_coord,
            player_bboxes,
        ) = self.extract_frame_arrays(tracks, frame_num)

        # 1. Execute Vectorized Physics Engine
        output = self.engine.process_frame(
            player_coords=player_coords,
            player_velocities=player_vels,
            player_teams=player_teams,
            player_ids=player_ids,
            is_goalkeeper=is_gk,
            ball_coord=ball_coord,
            attacking_team=self.attacking_team,
            pass_release=pass_release,
        )

        # 2. Render OpenCV Remontada Overlay with Raw BBoxes
        annotated_frame = self.overlay.render(
            frame=frame_img,
            output=output,
            player_coords=player_coords,
            player_velocities=player_vels,
            player_teams=player_teams,
            player_ids=player_ids,
            attacking_team=self.attacking_team,
            fps=fps,
            player_bboxes=player_bboxes, # THE BRIDGE IS CONNECTED
        )

        return annotated_frame, output
