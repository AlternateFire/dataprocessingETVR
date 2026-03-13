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


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color (#RRGGBB) to BGR tuple for OpenCV."""
    h = hex_color.lstrip('#')
    if len(h) != 6:
        return (0, 200, 255)  # default cyan
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_gaze_overlay(
    frame: np.ndarray, x: int, y: int, size: int = 20,
    color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Draw gaze indicator (dot with glow) on frame."""
    h, w = frame.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    color = color or (0, 200, 255)

    overlay = frame.copy()
    cv2.circle(overlay, (x, y), size, color, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.circle(frame, (x, y), 7, color, -1)
    cv2.circle(frame, (x, y), 7, (255, 255, 255), 2)
    return frame


def _compute_saccades(timeline: List[Dict], threshold: float = 35.0) -> List[Dict]:
    """Detect rapid eye movements (saccades) from frame-to-frame velocity."""
    segments = []
    for i in range(1, len(timeline)):
        a, b = timeline[i - 1], timeline[i]
        vel = np.hypot(b['x'] - a['x'], b['y'] - a['y'])
        if vel >= threshold:
            segments.append({
                'x1': int(a['x']), 'y1': int(a['y']),
                'x2': int(b['x']), 'y2': int(b['y']),
                'end_frame': i,
            })
    return segments


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
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    time_offset_ms: float = 0,
    smooth_window: int = 2,
    mapping_mode: str = "adaptive",
    gaze_color: Optional[str] = None,
    show_saccades: bool = True,
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
        scale_x=scale_x,
        scale_y=scale_y,
        time_offset_ms=time_offset_ms,
        smooth_window=smooth_window,
        mapping_mode=mapping_mode,
    )
    
    out_w = int(info['width'] * scale)
    out_h = int(info['height'] * scale)
    for g in timeline:
        g['x'] = int(g['x'] * scale)
        g['y'] = int(g['y'] * scale)

    saccades = _compute_saccades(timeline, threshold=35.0) if show_saccades else []
    saccade_persist_frames = 6
    gaze_bgr = hex_to_bgr(gaze_color) if gaze_color else None

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
            frame = draw_gaze_overlay(frame, g['x'], g['y'], color=gaze_bgr)
        for seg in saccades:
            if seg['end_frame'] >= frame_idx - saccade_persist_frames and seg['end_frame'] <= frame_idx + 2:
                pt1 = (int(np.clip(seg['x1'], 0, out_w - 1)), int(np.clip(seg['y1'], 0, out_h - 1)))
                pt2 = (int(np.clip(seg['x2'], 0, out_w - 1)), int(np.clip(seg['y2'], 0, out_h - 1)))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
        out.write(frame)
        frame_idx += 1
        if progress_callback and frame_idx % 10 == 0:
            progress_callback(frame_idx / total * 100)
    
    cap.release()
    out.release()
    if progress_callback:
        progress_callback(100.0)
    return output_path
