"""FastAPI application for gaze data processor."""

import os
import shutil
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .video_processor import get_video_info, export_video_with_gaze
from .gaze_processor import get_gaze_timeline, auto_suggest_calibration
from .database import init_db, save_session, load_session, list_sessions, delete_session, update_session_output_sync

BASE_DIR = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
EXPORT_DIR = BASE_DIR / "exports"
SESSIONS_DIR = BASE_DIR / "sessions_data"
UPLOAD_DIR.mkdir(exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

# Track export progress
export_progress = {}

# API router - mount before static so /api/* is handled correctly (avoids 405 on POST)
api = APIRouter(prefix="/api", tags=["api"])


@api.post("/upload")
async def upload_files(
    video: UploadFile = File(...),
    csv: UploadFile = File(...),
):
    """Upload video and CSV, process and return gaze timeline + video info."""
    session_id = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{session_id}_video{Path(video.filename).suffix}"
    csv_path = UPLOAD_DIR / f"{session_id}_gaze.csv"

    try:
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        with open(csv_path, "wb") as f:
            shutil.copyfileobj(csv.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    info = get_video_info(str(video_path))
    suggestions = auto_suggest_calibration(str(csv_path))
    timeline = get_gaze_timeline(
        str(csv_path),
        video_fps=info["fps"],
        video_width=info["width"],
        video_height=info["height"],
        video_frame_count=info["frame_count"],
        time_offset_ms=suggestions["time_offset_ms"],
        smooth_window=suggestions["smooth_window"],
        mapping_mode=suggestions["mapping_mode"],
        scale_x=suggestions["scale_x"],
        scale_y=suggestions["scale_y"],
    )

    return {
        "session_id": session_id,
        "video_path": str(video_path),
        "csv_path": str(csv_path),
        "video_info": info,
        "gaze_timeline": timeline,
        "video_url": f"/api/video/{session_id}",
        "auto_calibration": suggestions,
    }


@api.post("/auto-calibrate")
async def get_auto_calibration(csv_path: str = Form(...)):
    """Analyze gaze CSV and return suggested calibration parameters."""
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV not found")
    return auto_suggest_calibration(csv_path)


@api.post("/reprocess")
async def reprocess_gaze(
    csv_path: str = Form(...),
    video_info: str = Form("{}"),
    time_offset_ms: float = Form(0),
    smooth_window: int = Form(2),
    mapping_mode: str = Form("adaptive"),
    scale_x: float = Form(1.0),
    scale_y: float = Form(1.0),
):
    """Reprocess gaze with new calibration. Returns updated gaze_timeline.
    Flip/offset are applied by frontend for display; only mapping params are reprocessed here.
    """
    import json
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="CSV not found")
    info = json.loads(video_info)
    timeline = get_gaze_timeline(
        csv_path,
        video_fps=info.get("fps", 30),
        video_width=info.get("width", 1920),
        video_height=info.get("height", 1080),
        video_frame_count=info.get("frame_count"),
        time_offset_ms=float(time_offset_ms),
        smooth_window=max(1, min(9, int(smooth_window))),
        mapping_mode=mapping_mode if mapping_mode in ("adaptive", "fixed_eyetrackvr") else "adaptive",
        flip_x=False,
        flip_y=False,
        offset_x=0,
        offset_y=0,
        scale_x=float(scale_x),
        scale_y=float(scale_y),
    )
    return {"gaze_timeline": timeline}


@api.get("/video/{session_id}")
async def serve_video(session_id: str):
    """Stream uploaded video for playback."""
    matches = list(UPLOAD_DIR.glob(f"{session_id}_video*"))
    if not matches:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(matches[0], media_type="video/mp4")


@api.post("/export")
async def export_video(
    background_tasks: BackgroundTasks,
    video_path: str = Form(...),
    csv_path: str = Form(...),
    flip_x: str = Form("false"),
    flip_y: str = Form("false"),
    offset_x: int = Form(0),
    offset_y: int = Form(0),
    scale_x: float = Form(1.0),
    scale_y: float = Form(1.0),
    time_offset_ms: float = Form(0),
    smooth_window: int = Form(2),
    mapping_mode: str = Form("adaptive"),
    gaze_color: str = Form(""),
    show_saccades: str = Form("true"),
):
    """Export video with gaze overlay. Returns export_id for polling progress."""
    if not os.path.exists(video_path) or not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="Video or CSV file not found")

    export_id = str(uuid.uuid4())[:8]
    output_path = EXPORT_DIR / f"gaze_overlay_{export_id}.mp4"
    export_progress[export_id] = {"progress": 0, "done": False, "path": str(output_path)}

    def update_progress(pct):
        export_progress[export_id]["progress"] = pct
        if pct >= 100:
            export_progress[export_id]["done"] = True

    def run_export():
        try:
            export_video_with_gaze(
                video_path, csv_path, str(output_path), update_progress,
                flip_x=flip_x.lower() == "true",
                flip_y=flip_y.lower() == "true",
                offset_x=int(offset_x),
                offset_y=int(offset_y),
                scale_x=float(scale_x),
                scale_y=float(scale_y),
                time_offset_ms=float(time_offset_ms),
                smooth_window=max(1, min(9, int(smooth_window))),
                mapping_mode=mapping_mode if mapping_mode in ("adaptive", "fixed_eyetrackvr") else "adaptive",
                gaze_color=gaze_color or None,
                show_saccades=show_saccades.lower() == "true",
            )
        except Exception as e:
            export_progress[export_id]["error"] = str(e)
            export_progress[export_id]["done"] = True

    background_tasks.add_task(run_export)
    return {"export_id": export_id, "status": "processing"}


@api.get("/export/{export_id}/status")
async def export_status(export_id: str):
    """Poll export progress."""
    if export_id not in export_progress:
        raise HTTPException(status_code=404, detail="Export not found")
    data = export_progress[export_id]
    result = {"progress": data["progress"], "done": data["done"]}
    if data.get("error"):
        result["error"] = data["error"]
    if data["done"] and not data.get("error"):
        result["download_url"] = f"/api/download/{export_id}"
    return result


@api.get("/download/{export_id}")
async def download_export(export_id: str):
    """Download exported video."""
    if export_id not in export_progress:
        raise HTTPException(status_code=404, detail="Export not found")
    path = export_progress[export_id].get("path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(path, filename=f"gaze_overlay_{export_id}.mp4")


save_progress = {}  # session_id -> {progress, done, error}


@api.post("/sessions")
async def create_session(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    video_path: str = Form(""),
    csv_path: str = Form(""),
    video_info: str = Form("{}"),
    gaze_timeline: str = Form("[]"),
    session_upload_id: str = Form(""),
    compact: str = Form("false"),
    flip_x: str = Form("false"),
    flip_y: str = Form("false"),
    offset_x: int = Form(0),
    offset_y: int = Form(0),
    scale_x: float = Form(1.0),
    scale_y: float = Form(1.0),
    time_offset_ms: float = Form(0),
    smooth_window: int = Form(2),
    mapping_mode: str = Form("adaptive"),
    gaze_color: str = Form(""),
    show_saccades: str = Form("true"),
):
    """Save session: exports video with gaze overlay and stores it."""
    import json
    vid_info = json.loads(video_info)
    timeline = json.loads(gaze_timeline)
    if not video_path or not os.path.exists(video_path) or not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail="Video and CSV files required")
    sid = await save_session(
        name, csv_path or None, video_path or None, vid_info, timeline,
        session_upload_id=session_upload_id or None,
        status="processing",
    )
    session_dir = SESSIONS_DIR / str(sid)
    session_dir.mkdir(exist_ok=True)
    output_path = session_dir / "output.mp4"
    save_progress[sid] = {"progress": 0, "done": False, "error": None}
    scale = 0.75 if compact.lower() == "true" else 1.0

    def run_save():
        try:
            export_video_with_gaze(
                video_path, csv_path, str(output_path),
                progress_callback=lambda p: save_progress.__setitem__(sid, {**save_progress.get(sid, {}), "progress": p}),
                scale=scale,
                flip_x=flip_x.lower() == "true",
                flip_y=flip_y.lower() == "true",
                offset_x=int(offset_x),
                offset_y=int(offset_y),
                scale_x=float(scale_x),
                scale_y=float(scale_y),
                time_offset_ms=float(time_offset_ms),
                smooth_window=max(1, min(9, int(smooth_window))),
                mapping_mode=mapping_mode if mapping_mode in ("adaptive", "fixed_eyetrackvr") else "adaptive",
                gaze_color=gaze_color or None,
                show_saccades=show_saccades.lower() == "true",
            )
            save_progress[sid] = {"progress": 100, "done": True, "error": None}
            update_session_output_sync(sid, str(output_path))
        except Exception as e:
            save_progress[sid] = {"progress": 0, "done": True, "error": str(e)}

    background_tasks.add_task(run_save)
    return {"session_id": sid, "status": "processing", "message": "Exporting and saving..."}


@api.get("/sessions")
async def get_sessions():
    """List all saved sessions."""
    return await list_sessions()


@api.get("/sessions/{session_id}/save-status")
async def get_save_status(session_id: int):
    """Poll save/export progress."""
    if session_id not in save_progress:
        s = await load_session(session_id)
        if s and s.get("status") == "ready":
            return {"progress": 100, "done": True, "error": None}
        return {"progress": 0, "done": False, "error": None}
    d = save_progress[session_id]
    return {"progress": d["progress"], "done": d["done"], "error": d.get("error")}


@api.get("/sessions/{session_id}")
async def get_session(session_id: int):
    """Load a saved session."""
    s = await load_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    output_path = SESSIONS_DIR / str(session_id) / "output.mp4"
    stored_path = s.get("output_path") or ""
    if output_path.exists():
        s["video_url"] = f"/api/session-video/{session_id}"
        s["status"] = "ready"
    elif stored_path and os.path.exists(stored_path):
        s["video_url"] = f"/api/session-video/{session_id}"
        s["status"] = "ready"
    else:
        s["video_url"] = None
        s["status"] = s.get("status") or "processing"
    return s


@api.get("/session-video/{session_id}")
async def serve_session_video(session_id: int):
    """Serve saved session video (gaze overlay baked in)."""
    output_path = SESSIONS_DIR / str(session_id) / "output.mp4"
    if output_path.exists():
        return FileResponse(str(output_path), media_type="video/mp4")
    s = await load_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    path = s.get("output_path") or ""
    if path and os.path.exists(path):
        return FileResponse(path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")


@api.delete("/sessions/{session_id}")
async def remove_session(session_id: int):
    """Delete a session and its stored files."""
    session_dir = SESSIONS_DIR / str(session_id)
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
    if session_id in save_progress:
        del save_progress[session_id]
    ok = await delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Deleted"}


# Mount API first so /api/* routes take precedence over static (fixes 405 on POST)
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="EyeTrackVR Gaze Processor", lifespan=lifespan)
app.include_router(api)

# Serve static frontend last
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
