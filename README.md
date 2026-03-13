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

Gaze coordinates are mapped to video pixels using adaptive range fitting and smoothing.

## Purpose
- For a research project @ University of Maryland, College Park
- Exploring Cursor and AI assisted development tools to implement systems and feautres
- Exploring 'prompt engineering,' curious on the limits and extents to the system
