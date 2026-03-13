# EyeTrackVR Gaze Data Processor

Process gaze CSV data + world camera video to approximately visualize and export gaze-overlay footage.
This system does not behave like properly interpolated gaze data with pupil data like Pupil Labs systems. 
Do not expect the highest amount of accuracy!

## Features

- **Upload & process**: Drop or select MP4 video + gaze CSV (with `timestamp_ms`, `x`, `y` columns)
- **Interactive playback**: Video player with real-time gaze overlay (dot indicator)
- **Export**: Generate MP4 with gaze overlay baked in
- **Sessions**: Save exports the gaze-overlay video and stores it; load plays the stored video directly. Optional "Compact" mode uses 75% resolution to reduce file size (~50% smaller for 4–6 min recordings)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

## Data Format

- **CSV**: `timestamp_ms`, `eye_id`, `x`, `y`, (optional: `pupil_dilation`, `eye_blink`)
- **Video**: World camera MP4; timestamps should align with gaze recording

Gaze coordinates are mapped to video pixels. **Accuracy calibration** (below player):
- **Auto-detect**: Analyzes gaze data and suggests time offset, mapping mode, etc. Runs automatically on process; manual adjustments remain available
- **Use auto** (toggle): When on, uses auto-detected settings; when off, uses manual form values
- **Time offset** (ms): Fix sync if gaze lags/leads video (+ advances gaze)
- **Mapping**: Adaptive (fit to data) or Fixed EyeTrackVR (x∈[-1,1], y∈[0,1])
- **Scale X/Y**: Correct FOV mismatch between world camera and display

## Purpose
- For a research project @ University of Maryland, College Park
- Exploring Cursor and agentic development tools to implement systems and feautres
- Exploring 'prompt engineering,' curious on the limits and extents to the system
