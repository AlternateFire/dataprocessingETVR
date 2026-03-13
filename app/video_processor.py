"""Video processing with gaze overlay for export."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .gaze_processor import get_gaze_timeline


def get_video_info(video_path: str) -> Dict:
    """Extract video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = (frame_count / fps) * 1000 if fps > 0 else 0
    cap.release()
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'frame_count': frame_count,
        'duration_ms': duration_ms,
    }


def draw_gaze_overlay(frame: np.ndarray, x: int, y: int, size: int = 20) -> np.ndarray:
    """Draw gaze indicator (dot with glow) on frame."""
    h, w = frame.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    
    overlay = frame.copy()
    
    # Outer glow (semi-transparent)
    cv2.circle(overlay, (x, y), size, (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Inner bright dot
    cv2.circle(frame, (x, y), 6, (0, 200, 255), -1)
    cv2.circle(frame, (x, y), 6, (255, 255, 255), 2)
    
    return frame


def export_video_with_gaze(
    video_path: str,
    csv_path: str,
    output_path: str,
    progress_callback: Optional[callable] = None,
    scale: float = 1.0,
    flip_x: bool = False,
    flip_y: bool = False,
    offset_x: int = 0,
    offset_y: int = 0,
) -> str:
    """
    Create output video with gaze overlay.
    Returns path to output file.
    """
    info = get_video_info(video_path)
    timeline = get_gaze_timeline(
        csv_path,
        video_fps=info['fps'],
        video_width=info['width'],
        video_height=info['height'],
        video_frame_count=info['frame_count'],
        flip_x=flip_x,
        flip_y=flip_y,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    
    out_w = int(info['width'] * scale)
    out_h = int(info['height'] * scale)
    # Scale gaze coords to output resolution
    for g in timeline:
        g['x'] = int(g['x'] * scale)
        g['y'] = int(g['y'] * scale)

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, info['fps'], (out_w, out_h))

    frame_idx = 0
    total = info['frame_count']

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (out_w, out_h))
        if frame_idx < len(timeline):
            g = timeline[frame_idx]
            frame = draw_gaze_overlay(frame, g['x'], g['y'])
        out.write(frame)
        frame_idx += 1
        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx / total * 100)
    
    cap.release()
    out.release()
    if progress_callback:
        progress_callback(100.0)
    return output_path
