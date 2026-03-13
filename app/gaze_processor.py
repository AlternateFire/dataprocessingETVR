"""Gaze data processing and coordinate mapping for EyeTrackVR data."""

import pandas as pd
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from typing import Tuple, List, Optional


def load_gaze_data(csv_path: str, merge_eyes: bool = True) -> pd.DataFrame:
    """Load gaze CSV and return processed dataframe.
    If eye_id exists and merge_eyes=True, averages LEFT and RIGHT eyes per timestamp for better accuracy.
    """
    df = pd.read_csv(csv_path)
    required = ['timestamp_ms', 'x', 'y']
    if not all(c in df.columns for c in required):
        raise ValueError(f"CSV must contain columns: {required}")

    if merge_eyes and 'eye_id' in df.columns:
        # Binocular: average both eyes per timestamp (EyeTrackVR logs LEFT/RIGHT separately)
        numeric_cols = ['x', 'y']
        if 'pupil_dilation' in df.columns:
            numeric_cols.append('pupil_dilation')
        if 'eye_blink' in df.columns:
            numeric_cols.append('eye_blink')
        agg_dict = {c: 'mean' for c in numeric_cols}
        for c in df.columns:
            if c not in numeric_cols and c != 'timestamp_ms':
                agg_dict[c] = 'first'
        df = df.groupby('timestamp_ms', as_index=False).agg(agg_dict)
    return df.sort_values('timestamp_ms').reset_index(drop=True)


def smooth_gaze(df: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Light moving average to reduce jitter without adding lag (window=2 ~50ms at 40Hz)."""
    df = df.copy()
    if 'eye_blink' in df.columns:
        # Filter out blink samples (high eye_blink = eyes closed)
        blink_mask = df['eye_blink'] > 0.9
        df.loc[blink_mask, ['x', 'y']] = np.nan

    # Interpolate NaNs
    df['x'] = df['x'].interpolate(method='linear', limit_direction='both')
    df['y'] = df['y'].interpolate(method='linear', limit_direction='both')

    # Light smoothing (window=2 minimizes lag while reducing noise)
    if window > 1:
        df['x_smooth'] = uniform_filter1d(df['x'].values, size=window, mode='nearest')
        df['y_smooth'] = uniform_filter1d(df['y'].values, size=window, mode='nearest')
    else:
        df['x_smooth'] = df['x'].values
        df['y_smooth'] = df['y'].values
    return df


def map_gaze_to_video(
    x: float, y: float,
    video_width: int, video_height: int,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[int, int]:
    """
    Map normalized gaze coordinates to video pixel coordinates.
    EyeTrackVR data often uses normalized coords - we map to video bounds.
    """
    # Linear mapping with clamping
    x_norm = np.clip((x - x_range[0]) / (x_range[1] - x_range[0]), 0, 1)
    y_norm = np.clip((y - y_range[0]) / (y_range[1] - y_range[0]), 0, 1)
    px = int(x_norm * (video_width - 1))
    py = int(y_norm * (video_height - 1))
    return (px, py)


def auto_suggest_calibration(csv_path: str) -> dict:
    """
    Analyze gaze CSV and suggest calibration parameters.
    Returns dict with time_offset_ms, mapping_mode, flip_x, flip_y, scale_x, scale_y, smooth_window,
    plus a 'changes' list describing what was auto-detected.
    """
    df = load_gaze_data(csv_path)
    changes = []
    suggestions = {
        "time_offset_ms": 0,
        "mapping_mode": "adaptive",
        "flip_x": False,
        "flip_y": False,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "smooth_window": 2,
        "changes": [],
    }

    t = df["timestamp_ms"].values
    x_min, x_max = float(df["x"].min()), float(df["x"].max())
    y_min, y_max = float(df["y"].min()), float(df["y"].max())

    # Time offset: align video frame 0 with first gaze sample
    first_ts = float(t[0])
    if first_ts > 5:
        suggestions["time_offset_ms"] = -round(first_ts)
        suggestions["changes"].append(f"Time offset: {suggestions['time_offset_ms']}ms (align to first gaze sample)")

    # Mapping: use adaptive if y has negative values (EyeTrackVR variants use y in [-1,1] or similar)
    if y_min < -0.05 or y_max > 1.05:
        suggestions["mapping_mode"] = "adaptive"
        suggestions["changes"].append(
            f"Mapping: adaptive (y range [{y_min:.2f}, {y_max:.2f}] — Fixed would clip)"
        )
    elif -0.1 <= x_min and x_max <= 1.1 and 0 <= y_min and y_max <= 1.1:
        suggestions["mapping_mode"] = "fixed_eyetrackvr"
        suggestions["changes"].append("Mapping: Fixed EyeTrackVR (x∈[-1,1], y∈[0,1])")

    if not suggestions["changes"]:
        suggestions["changes"].append("No automatic adjustments needed")

    return suggestions


def compute_gaze_ranges(df: pd.DataFrame) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute x,y ranges from data for adaptive mapping. Uses percentiles for robustness."""
    x_min, x_max = df['x'].quantile(0.02), df['x'].quantile(0.98)
    y_min, y_max = df['y'].quantile(0.02), df['y'].quantile(0.98)
    # Add padding
    x_pad = max(0.1, (x_max - x_min) * 0.1)
    y_pad = max(0.1, (y_max - y_min) * 0.1)
    return ((x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad))


def build_gaze_interpolator(
    df: pd.DataFrame,
    video_width: int,
    video_height: int,
    smooth_window: int = 2,
    time_offset_ms: float = 0,
    mapping_mode: str = "adaptive",
) -> Tuple[callable, callable, float, float]:
    """
    Build interpolation functions for gaze at any timestamp.
    Returns (px_interp, py_interp, t_start_ms, t_end_ms).
    mapping_mode: "adaptive" = fit to data percentiles; "fixed_eyetrackvr" = x in [-1,1], y in [0,1].
    time_offset_ms: added to query time (positive = advance gaze when it lags).
    """
    df = smooth_gaze(df, window=smooth_window)
    if mapping_mode == "fixed_eyetrackvr":
        x_range, y_range = (-1.0, 1.0), (0.0, 1.0)
    else:
        x_range, y_range = compute_gaze_ranges(df)

    t = df['timestamp_ms'].values
    x = df['x_smooth'].values
    y = df['y_smooth'].values

    interp_kind = 'cubic' if len(t) >= 4 else 'linear'
    x_raw = interp1d(t, x, kind=interp_kind, bounds_error=False, fill_value=(x[0], x[-1]))
    y_raw = interp1d(t, y, kind=interp_kind, bounds_error=False, fill_value=(y[0], y[-1]))

    def px_interp(ts):
        ts_adj = ts + time_offset_ms
        xv = float(x_raw(ts_adj))
        yv = float(y_raw(ts_adj))
        return map_gaze_to_video(xv, yv, video_width, video_height, x_range, y_range)[0]

    def py_interp(ts):
        ts_adj = ts + time_offset_ms
        xv = float(x_raw(ts_adj))
        yv = float(y_raw(ts_adj))
        return map_gaze_to_video(xv, yv, video_width, video_height, x_range, y_range)[1]

    return (px_interp, py_interp, float(t[0]), float(t[-1]))


def _apply_gaze_transform(
    px: int, py: int,
    video_width: int, video_height: int,
    flip_x: bool = False, flip_y: bool = False,
    offset_x: int = 0, offset_y: int = 0,
    scale_x: float = 1.0, scale_y: float = 1.0,
) -> Tuple[int, int]:
    """Apply flip, scale (from center), and offset to gaze pixel coordinates."""
    cx, cy = (video_width - 1) / 2, (video_height - 1) / 2
    px = (px - cx) * scale_x + cx
    py = (py - cy) * scale_y + cy
    if flip_x:
        px = video_width - 1 - px
    if flip_y:
        py = video_height - 1 - py
    px = int(np.clip(px + offset_x, 0, video_width - 1))
    py = int(np.clip(py + offset_y, 0, video_height - 1))
    return (px, py)


def get_gaze_timeline(
    csv_path: str,
    video_fps: float,
    video_width: int,
    video_height: int,
    video_duration_ms: Optional[float] = None,
    video_frame_count: Optional[int] = None,
    time_offset_ms: float = 0,
    smooth_window: int = 2,
    mapping_mode: str = "adaptive",
    flip_x: bool = False,
    flip_y: bool = False,
    offset_x: int = 0,
    offset_y: int = 0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[dict]:
    """
    Generate gaze coordinates for each video frame.
    Returns list of {frame_idx, timestamp_ms, x, y} for frontend.
    flip_x/flip_y: mirror gaze on axis (for orientation checks).
    offset_x/offset_y: pixel offset (e.g. camera mounted left of headset).
    """
    df = load_gaze_data(csv_path)
    x_interp, y_interp, t_start, t_end = build_gaze_interpolator(
        df, video_width, video_height,
        smooth_window=smooth_window,
        time_offset_ms=time_offset_ms,
        mapping_mode=mapping_mode,
    )

    ms_per_frame = 1000.0 / video_fps if video_fps > 0 else 33.33
    if video_frame_count is not None:
        num_frames = video_frame_count
    elif video_duration_ms is not None:
        num_frames = int(video_duration_ms / ms_per_frame)
    else:
        num_frames = max(1, int((t_end - t_start) / ms_per_frame))

    timeline = []
    for i in range(num_frames):
        ts_ms = t_start + i * ms_per_frame
        try:
            px = int(x_interp(ts_ms))
            py = int(y_interp(ts_ms))
        except (ValueError, TypeError):
            px, py = video_width // 2, video_height // 2
        px, py = _apply_gaze_transform(
            px, py, video_width, video_height,
            flip_x=flip_x, flip_y=flip_y,
            offset_x=offset_x, offset_y=offset_y,
            scale_x=scale_x, scale_y=scale_y,
        )
        timeline.append({
            'frame_idx': i,
            'timestamp_ms': ts_ms,
            'x': px,
            'y': py,
        })
    return timeline
