"""
REMONTADA TACTICAL EV ENGINE - Main Execution & Tracker Integration Pipeline
Connects YOLO/ByteTrack object detection, team classification, velocity estimation,
and pitch homography into RemontadaPipelineAdapter for real-time tactical rendering.
"""

import sys
import os
import argparse
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# =============================================================================
# 0. PRE-FLIGHT DEPENDENCY CHECKER
# =============================================================================
def check_dependencies() -> bool:
    """Verifies all necessary computer vision and AI dependencies before running."""
    required_packages = {
        "numpy": "numpy",
        "cv2": "opencv-python",
        "torch": "torch",
        "ultralytics": "ultralytics",
        "scipy": "scipy",
        "imageio": "imageio",
        "imageio_ffmpeg": "imageio-ffmpeg",
    }
    
    missing = []
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
            
    if missing:
        print("\n" + "=" * 65)
        print(" [!] MISSING REQUIRED PACKAGES")
        print("=" * 65)
        print(f"The following required packages are not installed: {', '.join(missing)}")
        print("\nPlease install them using:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 65 + "\n")
        return False
    return True

if not check_dependencies():
    sys.exit(1)

import numpy as np
import cv2
import torch
import imageio
from ultralytics import YOLO
from scipy.ndimage import uniform_filter1d

from remontada_engine import ContinuousEVEngine, EngineOutput
from remontada_overlay import RemontadaOverlay
from pipeline_integration import RemontadaPipelineAdapter


# =============================================================================
# 1. HOMOGRAPHY & VIEW TRANSFORMER
# =============================================================================
class PitchViewTransformer:
    """
    Transforms pixel coordinates (u, v) from camera broadcast perspective
    to standard tactical pitch coordinates [X: 0..104m, Y: 0..68m] and vice versa.
    """

    def __init__(
        self,
        source_pixel_points: Optional[np.ndarray] = None,
        target_pitch_points: Optional[np.ndarray] = None,
        pitch_length: float = 104.0,
        pitch_width: float = 68.0,
        frame_shape: Tuple[int, int] = (720, 1280),
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.frame_shape = frame_shape
        h, w = frame_shape[:2]

        if source_pixel_points is None:
            source_pixel_points = np.array([
                [w * 0.12, h * 0.25],  # Top-left corner of visible pitch
                [w * 0.88, h * 0.25],  # Top-right corner of visible pitch
                [w * 0.98, h * 0.92],  # Bottom-right corner of visible pitch
                [w * 0.02, h * 0.92],  # Bottom-left corner of visible pitch
            ], dtype=np.float32)

        if target_pitch_points is None:
            target_pitch_points = np.array([
                [0.0, 0.0],
                [pitch_length, 0.0],
                [pitch_length, pitch_width],
                [0.0, pitch_width],
            ], dtype=np.float32)

        self.src_pts = source_pixel_points
        self.dst_pts = target_pitch_points

        self.homography, _ = cv2.findHomography(self.src_pts, self.dst_pts)
        if self.homography is not None:
            self.inverse_homography = np.linalg.inv(self.homography)
        else:
            self.inverse_homography = None

    def transform_pixel_to_pitch(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """Maps an image pixel position (e.g. player feet) to pitch meters (X, Y)."""
        if self.homography is not None:
            pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.homography)
            px = float(np.clip(transformed[0, 0, 0], 0.0, self.pitch_length))
            py = float(np.clip(transformed[0, 0, 1], 0.0, self.pitch_width))
            return px, py
        
        h, w = self.frame_shape[:2]
        px = (pixel_x / w) * self.pitch_length
        py = (pixel_y / h) * self.pitch_width
        return float(np.clip(px, 0.0, self.pitch_length)), float(np.clip(py, 0.0, self.pitch_width))

    def transform_pitch_to_pixel(self, pitch_x: float, pitch_y: float) -> Optional[Tuple[int, int]]:
        """Maps tactical pitch coordinates (meters) to screen pixel coordinates (u, v)."""
        if self.inverse_homography is not None:
            pt = np.array([[[pitch_x, pitch_y]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, self.inverse_homography)
            return int(transformed[0, 0, 0]), int(transformed[0, 0, 1])

        h, w = self.frame_shape[:2]
        return int((pitch_x / self.pitch_length) * w), int((pitch_y / self.pitch_width) * h)


# =============================================================================
# 2. JERSEY COLOR TEAM ASSIGNMENT
# =============================================================================
class TeamColorAssigner:
    """
    Extracts jersey color features from player bounding box upper torsos
    and clusters players into Team 0 and Team 1 using KMeans.
    """

    def __init__(self):
        self.team_color_centers: Optional[np.ndarray] = None
        self.player_team_history: Dict[int, List[int]] = {}

    def extract_jersey_color(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Extracts dominant color vector from player's upper torso region."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_h = y2 - y1
        box_w = x2 - x1
        if box_h <= 10 or box_w <= 10:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        torso_y1 = y1 + int(box_h * 0.15)
        torso_y2 = y1 + int(box_h * 0.45)
        torso_x1 = x1 + int(box_w * 0.20)
        torso_x2 = x1 + int(box_w * 0.80)

        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if torso_crop.size == 0:
            return np.array([128.0, 128.0, 128.0], dtype=np.float32)

        hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
        median_hsv = np.median(hsv_crop.reshape(-1, 3), axis=0)
        return median_hsv.astype(np.float32)

    def fit_teams(self, sample_colors: List[np.ndarray]) -> None:
        """Clusters initial set of collected jersey colors into two team prototypes."""
        if len(sample_colors) < 4:
            self.team_color_centers = np.array([[0, 200, 200], [120, 200, 200]], dtype=np.float32)
            return

        colors_arr = np.array(sample_colors, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        _, _, centers = cv2.kmeans(colors_arr, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        self.team_color_centers = centers

    def assign_team(self, player_id: int, color_feat: np.ndarray) -> int:
        """Assigns team 0 or 1 based on nearest cluster center and temporal voting."""
        if self.team_color_centers is None:
            return 1 if (player_id % 2 == 1) else 0

        d0 = np.linalg.norm(color_feat - self.team_color_centers[0])
        d1 = np.linalg.norm(color_feat - self.team_color_centers[1])
        assigned = 0 if d0 < d1 else 1

        if player_id not in self.player_team_history:
            self.player_team_history[player_id] = []
        self.player_team_history[player_id].append(assigned)

        history = self.player_team_history[player_id][-30:]
        return 1 if (sum(history) > len(history) / 2.0) else 0


# =============================================================================
# 3. YOLOV8 + BYTETRACK OBJECT TRACKING PIPELINE
# =============================================================================
class FootballTracker:
    """
    Runs YOLOv8 object detection with ByteTrack tracking on match video,
    extracting player bounding boxes, velocities, team IDs, and ball coordinates.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence: float = 0.35,
        view_transformer: Optional[PitchViewTransformer] = None,
    ):
        print(f"[*] Initializing YOLOv8 Model [{model_path}] with ByteTrack...")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.transformer = view_transformer or PitchViewTransformer()
        self.team_assigner = TeamColorAssigner()

    def track_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Processes video and produces the complete structured `tracks` dictionary.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video source: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.transformer.frame_shape = (height, width)

        frames_to_process = min(total_frames, max_frames) if max_frames else total_frames
        print(f"[*] Running ByteTracker on {frames_to_process} frames ({width}x{height} @ {fps:.1f} FPS)...")

        raw_tracks: Dict[str, Dict[int, Any]] = {"players": {}, "ball": {}}
        jersey_samples: List[np.ndarray] = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            raw_tracks["players"][frame_idx] = {}

            # Execute YOLOv8 detection with ByteTrack (Class 0: person, Class 32: sports ball)
            results = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=self.confidence,
                classes=[0, 32],
                verbose=False,
            )

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].item())
                        box_xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                        conf = float(boxes.conf[i].item())

                        if cls_id == 0:  # Player / Referee / GK
                            track_id = int(boxes.id[i].item()) if boxes.id is not None and boxes.id[i] is not None else (i + 1)
                            
                            feet_x = (box_xyxy[0] + box_xyxy[2]) / 2.0
                            feet_y = box_xyxy[3]
                            pitch_x, pitch_y = self.transformer.transform_pixel_to_pitch(feet_x, feet_y)

                            color_feat = self.team_assigner.extract_jersey_color(frame, box_xyxy)
                            if frame_idx < 30:
                                jersey_samples.append(color_feat)

                            raw_tracks["players"][frame_idx][track_id] = {
                                "bbox": box_xyxy,
                                "position": [pitch_x, pitch_y],
                                "transformed_position": [pitch_x, pitch_y],
                                "color_feat": color_feat,
                                "conf": conf,
                            }

                        elif cls_id == 32:  # Sports Ball
                            ball_cx = (box_xyxy[0] + box_xyxy[2]) / 2.0
                            ball_cy = (box_xyxy[1] + box_xyxy[3]) / 2.0
                            ball_pitch_x, ball_pitch_y = self.transformer.transform_pixel_to_pitch(ball_cx, ball_cy)
                            raw_tracks["ball"][frame_idx] = {
                                "bbox": box_xyxy,
                                "position": [ball_pitch_x, ball_pitch_y],
                                "transformed_position": [ball_pitch_x, ball_pitch_y],
                                "conf": conf,
                            }

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"    [+] Tracked {frame_idx}/{frames_to_process} frames...")

        cap.release()

        # Step 2: Fit Team Color Clusters
        self.team_assigner.fit_teams(jersey_samples)

        # Step 3: Interpolate missing ball positions
        self._interpolate_ball_trajectory(raw_tracks["ball"], frame_idx)

        # Step 4: Calculate smoothed velocities and assign teams & GK roles
        processed_tracks = self._post_process_tracks(raw_tracks, fps, frame_idx)
        print("[*] Tracking extraction and kinematics post-processing complete!")
        return processed_tracks

    def _interpolate_ball_trajectory(self, ball_dict: Dict[int, Any], total_frames: int) -> None:
        """Fills missing ball detection frames with linear coordinate interpolation."""
        detected_frames = sorted(ball_dict.keys())
        if not detected_frames:
            for f in range(total_frames):
                ball_dict[f] = {
                    "position": [52.0, 34.0],
                    "transformed_position": [52.0, 34.0],
                }
            return

        all_frames = list(range(total_frames))
        xs = [ball_dict[f]["position"][0] for f in detected_frames]
        ys = [ball_dict[f]["position"][1] for f in detected_frames]

        interp_x = np.interp(all_frames, detected_frames, xs)
        interp_y = np.interp(all_frames, detected_frames, ys)

        for f in range(total_frames):
            if f not in ball_dict:
                ball_dict[f] = {
                    "position": [float(interp_x[f]), float(interp_y[f])],
                    "transformed_position": [float(interp_x[f]), float(interp_y[f])],
                    "interpolated": True,
                }

    def _post_process_tracks(
        self,
        raw_tracks: Dict[str, Any],
        fps: float,
        total_frames: int,
    ) -> Dict[str, Any]:
        """Calculates smoothed (Vx, Vy) velocities and assigns teams & GK masks."""
        player_trajectories: Dict[int, Dict[int, Tuple[float, float]]] = {}
        for f in range(total_frames):
            for pid, info in raw_tracks["players"].get(f, {}).items():
                if pid not in player_trajectories:
                    player_trajectories[pid] = {}
                player_trajectories[pid][f] = (info["position"][0], info["position"][1])

        player_velocities: Dict[int, Dict[int, Tuple[float, float]]] = {}
        dt = 1.0 / max(fps, 1.0)

        for pid, frame_pos in player_trajectories.items():
            f_list = sorted(frame_pos.keys())
            player_velocities[pid] = {}
            for i, f in enumerate(f_list):
                if len(f_list) == 1:
                    vx, vy = 0.0, 0.0
                elif i == 0:
                    next_f = f_list[1]
                    vx = (frame_pos[next_f][0] - frame_pos[f][0]) / ((next_f - f) * dt)
                    vy = (frame_pos[next_f][1] - frame_pos[f][1]) / ((next_f - f) * dt)
                elif i == len(f_list) - 1:
                    prev_f = f_list[-2]
                    vx = (frame_pos[f][0] - frame_pos[prev_f][0]) / ((f - prev_f) * dt)
                    vy = (frame_pos[f][1] - frame_pos[prev_f][1]) / ((f - prev_f) * dt)
                else:
                    prev_f = f_list[i - 1]
                    next_f = f_list[i + 1]
                    vx = (frame_pos[next_f][0] - frame_pos[prev_f][0]) / ((next_f - prev_f) * dt)
                    vy = (frame_pos[next_f][1] - frame_pos[prev_f][1]) / ((next_f - prev_f) * dt)

                speed = math.hypot(vx, vy)
                if speed > 10.0:
                    vx = (vx / speed) * 10.0
                    vy = (vy / speed) * 10.0
                player_velocities[pid][f] = (float(vx), float(vy))

        final_tracks: Dict[str, Any] = {"players": {}, "ball": raw_tracks["ball"]}

        for f in range(total_frames):
            final_tracks["players"][f] = {}
            frame_players = raw_tracks["players"].get(f, {})
            
            for pid, info in frame_players.items():
                pos = info["position"]
                vx, vy = player_velocities.get(pid, {}).get(f, (0.0, 0.0))
                team_id = self.team_assigner.assign_team(pid, info.get("color_feat", np.zeros(3)))
                is_gk = (pos[0] < 8.0) or (pos[0] > (self.transformer.pitch_length - 8.0))

                final_tracks["players"][f][pid] = {
                    "bbox": info.get("bbox", [0, 0, 0, 0]),
                    "position": pos,
                    "transformed_position": pos,
                    "speed_x": vx,
                    "speed_y": vy,
                    "team": team_id,
                    "is_goalkeeper": is_gk,
                }

        return final_tracks


# =============================================================================
# 4. SYNTHETIC MATCH VIDEO GENERATOR (FALLBACK DEMO)
# =============================================================================
def generate_synthetic_match_video(output_path: str, num_frames: int = 350) -> None:
    """
    Creates a realistic synthetic football broadcast clip for testing when
    test_clip_1.mp4 is not present on disk.
    """
    print(f"[*] Generating synthetic match video demo -> {output_path} ({num_frames} frames)...")
    w, h = 1280, 720
    fps = 25.0

    # imageio's ffmpeg plugin ships its own ffmpeg binary and encodes proper
    # H.264 (yuv420p) MP4s, so we don't depend on Windows having openh264.dll
    # or a working system ffmpeg install.
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart"],
    )

    for f in range(num_frames):
        t = f * 0.04
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (38, 92, 45)

        # Pitch markings
        cv2.rectangle(frame, (60, 50), (w - 60, h - 50), (230, 230, 230), 2)
        cv2.line(frame, (w // 2, 50), (w // 2, h - 50), (230, 230, 230), 2)
        cv2.circle(frame, (w // 2, h // 2), 70, (230, 230, 230), 2)

        # Penalty boxes
        cv2.rectangle(frame, (60, h // 2 - 130), (230, h // 2 + 130), (230, 230, 230), 2)
        cv2.rectangle(frame, (w - 230, h // 2 - 130), (w - 60, h // 2 + 130), (230, 230, 230), 2)

        ball_x = int(600 + 180 * math.sin(t * 0.8))
        ball_y = int(360 + 90 * math.cos(t * 0.6))
        
        # Draw Attackers
        for i in range(10):
            px = int(450 + 400 * (i / 10.0) + 30 * math.sin(t + i))
            py = int(120 + 480 * ((i % 5) / 5.0) + 20 * math.cos(t * 1.2 + i))
            cv2.circle(frame, (px, py), 12, (255, 190, 30), -1)
            cv2.circle(frame, (px, py), 12, (20, 20, 20), 2)

        # Draw Defenders
        for i in range(10):
            dx = int(550 + 420 * (i / 10.0) - 25 * math.cos(t + i))
            dy = int(140 + 460 * ((i % 5) / 5.0) + 25 * math.sin(t * 0.9 + i))
            cv2.circle(frame, (dx, dy), 12, (30, 40, 240), -1)
            cv2.circle(frame, (dx, dy), 12, (20, 20, 20), 2)

        cv2.circle(frame, (ball_x, ball_y), 7, (255, 255, 255), -1)
        cv2.circle(frame, (ball_x, ball_y), 7, (20, 20, 20), 2)

        # OpenCV drawing produces BGR; imageio/ffmpeg expects RGB, so convert
        # before appending or the output colors will look inverted.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)

    writer.close()
    print(f"[*] Synthetic demo video successfully saved to {output_path}!")


# =============================================================================
# 5. MAIN INTEGRATION & TACTICAL FREEZE-FRAME PIPELINE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Remontada Tactical EV Engine - YOLO/ByteTrack Pipeline & Live Visualizer"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="test_clip_1.mp4",
        help="Path to input match video file (default: test_clip_1.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="remontada_annotated_output.mp4",
        help="Path to save annotated output video (default: remontada_annotated_output.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="YOLO model path or weight file (default: yolov8m.pt)",
    )
    parser.add_argument(
        "--attacking-team",
        type=int,
        default=1,
        help="Team ID currently attacking towards X=104 (default: 1)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max number of frames to process (e.g. for quick preview)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow pop-up window (for headless server runs)",
    )
    args = parser.parse_args()

    # Step A: Validate or automatically generate input video
    input_video_path = args.video
    if not os.path.exists(input_video_path):
        print(f"[!] Input video '{input_video_path}' was not found on disk.")
        print("[*] Generating a synthetic 350-frame match video demo so you can test immediately...")
        generate_synthetic_match_video(input_video_path, num_frames=350)

    # Step B: Initialize ViewTransformer, Engine, Overlay, and Adapter
    view_transformer = PitchViewTransformer()
    engine = ContinuousEVEngine(attack_direction=1)
    overlay = RemontadaOverlay(view_transformer=view_transformer)
    adapter = RemontadaPipelineAdapter(
        engine=engine,
        overlay=overlay,
        view_transformer=view_transformer,
        attacking_team=args.attacking_team,
    )

    # Step C: Run YOLOv8 + ByteTrack Tracker Integration
    tracker = FootballTracker(
        model_path=args.model,
        confidence=0.30,
        view_transformer=view_transformer,
    )
    tracks = tracker.track_video(
        video_path=input_video_path,
        max_frames=args.max_frames,
    )

    # Step D: Open Video Capture and Setup Video Writer
    cap = cv2.VideoCapture(input_video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_run = min(total_frames, args.max_frames) if args.max_frames else total_frames

    # NOTE: cv2.VideoWriter is intentionally not used here. On Windows its
    # built-in fourcc codecs are unreliable for MP4: 'mp4v' writes a raw
    # stream that many players (including Windows' own) can't decode and
    # just freeze on, while 'avc1' needs an openh264.dll that OpenCV doesn't
    # ship and most systems don't have installed. imageio's ffmpeg plugin
    # bundles its own ffmpeg binary and writes a properly muxed, widely
    # compatible H.264 MP4 instead.
    writer = imageio.get_writer(
        args.output,
        fps=fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart"],
    )

    print("\n" + "=" * 70)
    print(" REMONTADA TACTICAL ENGINE - LIVE EXECUTION & ADAPTER HANDOFF")
    print(f" Output Video:     {args.output}")
    print(f" Live UI Window:   {'Enabled' if not args.no_display else 'Disabled'}")
    print("=" * 70 + "\n")

    frame_num = 0
    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (args.max_frames and frame_num >= args.max_frames):
            break

        # -------------------------------------------------------------
        # ADAPTER HANDOFF: Pass tracks dictionary into Remontada Engine
        # -------------------------------------------------------------
        annotated_frame, output = adapter.process_frame(
            frame_img=frame,
            tracks=tracks,
            frame_num=frame_num,
            pass_release=False,
            fps=fps,
        )

        # annotated_frame is BGR (OpenCV's native order); imageio/ffmpeg
        # expects RGB, so convert before appending or colors invert on export.
        rgb_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb_annotated_frame)

        # -------------------------------------------------------------
        # LIVE POP-UP WINDOW: cv2.imshow('Remontada Live')
        # -------------------------------------------------------------
        if not args.no_display:
            cv2.imshow("Remontada Live", annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\n[!] Execution stopped by user via keyboard interrupt.")
                break

        frame_num += 1
        if frame_num % 25 == 0 or frame_num == frames_to_run:
            elapsed = time.perf_counter() - start_time
            curr_fps = frame_num / max(elapsed, 1e-3)
            print(
                f"Frame {frame_num:4d}/{frames_to_run:4d} | "
                f"EV: +{output.max_ev_value:.2f} | "
                f"Target: ({output.max_ev_coord[0]:.1f}m, {output.max_ev_coord[1]:.1f}m) | "
                f"Pressure: {output.pressure_metric:4.1f}% | "
                f"Feasibility: {output.pass_feasibility*100:3.0f}% | "
                f"Speed: {curr_fps:4.1f} FPS"
            )

    cap.release()
    writer.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_time
    avg_fps = frame_num / max(total_time, 1e-3)
    print("\n" + "=" * 70)
    print(" [✓] REMONTADA PIPELINE PROCESSING COMPLETE!")
    print(f" Total Frames Processed: {frame_num}")
    print(f" Total Processing Time:   {total_time:.2f} seconds ({avg_fps:.1f} FPS)")
    print(f" Final Video Output:      {os.path.abspath(args.output)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
    