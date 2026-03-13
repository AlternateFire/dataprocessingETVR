#!/usr/bin/env python3
"""Run the EyeTrackVR Gaze Processor server.

Usage: python run.py
       (Activate venv first: source .venv/bin/activate)
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
