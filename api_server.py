"""
SmartPlay AI — REST API over the padel analysis pipeline.

Court keypoints: either send JSON with the job (form field or small file), or rely on
cache/fixed_keypoints_detection.json from config.

Run: uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import config
from run_analysis import normalize_court_keypoints_xy, run_analysis

BASE_DIR = Path(config.BASE_DIR)
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results_json"

app = FastAPI(title="SmartPlay AI API", description="Padel video analysis API")

_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_origins_env = os.getenv("FRONTEND_ORIGINS", _default_origins)
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]

# ngrok / tunnel frontends (https://xxxx.ngrok-free.dev)
NGROK_ORIGIN_REGEX = r"https://.*\.ngrok-free\.dev|https://.*\.ngrok\.io|http://.*\.ngrok-free\.dev"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=NGROK_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job registry (single-process MVP)
JOBS: Dict[str, Dict[str, Any]] = {}


def _job_dir(job_id: str) -> Path:
    return RESULTS_DIR / job_id


def _persist_job_meta(job_id: str) -> None:
    meta_path = _job_dir(job_id) / "job_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(JOBS.get(job_id, {}), f, indent=2, default=str)


def _execute_job(job_id: str, video_filename: str) -> None:
    job_upload = UPLOAD_DIR / job_id / video_filename
    out_dir = _job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / "output.mp4"
    csv_path = out_dir / "data.csv"
    summary_path = out_dir / "summary.json"
    shots_path = out_dir / "shot_events.json"

    kp_load = JOBS[job_id].get("keypoints_load_path")
    if not kp_load or not os.path.isfile(kp_load):
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = "Missing court keypoints path for job"
        _persist_job_meta(job_id)
        return

    JOBS[job_id]["status"] = "processing"
    JOBS[job_id]["error"] = None
    _persist_job_meta(job_id)

    try:
        meta = run_analysis(
            str(job_upload.resolve()),
            str(output_video.resolve()),
            collect_data=True,
            collect_data_path=str(csv_path.resolve()),
            summary_json_path=str(summary_path.resolve()),
            shot_events_json_path=str(shots_path.resolve()),
            allow_interactive_keypoints=False,
            keypoints_load_path=kp_load,
            persist_keypoints=False,
        )
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["meta"] = meta
        JOBS[job_id]["error"] = None
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["meta"] = None
    _persist_job_meta(job_id)


def _resolve_job_keypoints_path(
    job_upload_dir: Path,
    court_keypoints_json: Optional[str],
    keypoints_file_body: Optional[bytes],
) -> str:
    """Return absolute path to a JSON file with 12 court keypoints."""
    raw_points = None

    if keypoints_file_body:
        try:
            raw_points = json.loads(keypoints_file_body.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid keypoints_file JSON: {e}") from e

    if raw_points is None and court_keypoints_json and court_keypoints_json.strip():
        try:
            raw_points = json.loads(court_keypoints_json.strip())
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid court_keypoints_json: {e}") from e

    kp_job_path = job_upload_dir / "court_keypoints.json"

    if raw_points is not None:
        try:
            pairs = normalize_court_keypoints_xy(raw_points)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        job_upload_dir.mkdir(parents=True, exist_ok=True)
        with open(kp_job_path, "w", encoding="utf-8") as f:
            json.dump([[x, y] for x, y in pairs], f)
        return str(kp_job_path.resolve())

    g = config.FIXED_COURT_KEYPOINTS_LOAD_PATH
    if g and os.path.isfile(g):
        return str(Path(g).resolve())

    raise HTTPException(
        status_code=400,
        detail=(
            "Court keypoints required: send `court_keypoints_json` (multipart form) or "
            "`keypoints_file` (.json with 12 [[x,y], ...] points in video pixel coords), "
            "or create cache/fixed_keypoints_detection.json on the server."
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tools/first-frame")
async def first_frame(file: UploadFile = File(...)) -> dict[str, Any]:
    """Return the first decoded video frame as a JPEG (base64) plus native width/height for UI calibration."""
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    finally:
        await file.close()

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise HTTPException(status_code=400, detail="Cannot read first frame")
        h, w = frame.shape[:2]
        enc_ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not enc_ok:
            raise HTTPException(status_code=500, detail="Failed to encode JPEG")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return {
            "width": int(w),
            "height": int(h),
            "mime": "image/jpeg",
            "image_base64": b64,
        }
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    court_keypoints_json: Optional[str] = Form(None),
    keypoints_file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    job_id = str(uuid.uuid4())
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    kp_body: Optional[bytes] = None
    if keypoints_file is not None and keypoints_file.filename:
        kp_body = await keypoints_file.read()

    keypoints_load_path = _resolve_job_keypoints_path(
        job_upload_dir,
        court_keypoints_json,
        kp_body,
    )

    ext = Path(file.filename or "video.mp4").suffix or ".mp4"
    safe_name = f"input{ext}"
    dest = job_upload_dir / safe_name

    try:
        with open(dest, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
    finally:
        await file.close()

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "video_file": safe_name,
        "keypoints_load_path": keypoints_load_path,
        "error": None,
        "meta": None,
    }
    _persist_job_meta(job_id)

    background_tasks.add_task(_execute_job, job_id, safe_name)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued"},
    )


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        meta_file = _job_dir(job_id) / "job_meta.json"
        if meta_file.is_file():
            with open(meta_file, encoding="utf-8") as f:
                data = json.load(f)
                JOBS[job_id] = data
        else:
            raise HTTPException(status_code=404, detail="Unknown job_id")

    row = JOBS[job_id].copy()
    row.pop("meta", None)
    return row


@app.get("/jobs/{job_id}/results")
def get_results(job_id: str) -> dict[str, Any]:
    get_job(job_id)
    if JOBS[job_id].get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed (status={JOBS[job_id].get('status')})",
        )

    out: dict[str, Any] = {
        "job_id": job_id,
        "meta": JOBS[job_id].get("meta"),
        "summary": None,
        "shot_events": None,
    }
    d = _job_dir(job_id)
    sp = d / "summary.json"
    if sp.is_file():
        with open(sp, encoding="utf-8") as f:
            out["summary"] = json.load(f)
    ep = d / "shot_events.json"
    if ep.is_file():
        with open(ep, encoding="utf-8") as f:
            out["shot_events"] = json.load(f)

    return out


@app.get("/jobs/{job_id}/download/video")
def download_video(job_id: str) -> FileResponse:
    get_job(job_id)
    path = _job_dir(job_id) / "output.mp4"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Output video not found")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}_output.mp4")


@app.get("/jobs/{job_id}/download/first_frame")
def download_first_frame(job_id: str) -> FileResponse:
    get_job(job_id)
    out_dir = _job_dir(job_id)
    video_path = out_dir / "output.mp4"
    frame_path = out_dir / "first_frame.jpg"
    
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Output video not found")
        
    if not frame_path.is_file():
        cap = cv2.VideoCapture(str(video_path.resolve()))
        success, frame = cap.read()
        if success and frame is not None:
            cv2.imwrite(str(frame_path.resolve()), frame)
        cap.release()
        
    if not frame_path.is_file():
        raise HTTPException(status_code=500, detail="Could not extract first frame from video")
        
    return FileResponse(frame_path, media_type="image/jpeg", filename=f"{job_id}_preview.jpg")


@app.get("/jobs/{job_id}/download/csv")
def download_csv(job_id: str) -> FileResponse:
    get_job(job_id)
    path = _job_dir(job_id) / "data.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(path, media_type="text/csv", filename=f"{job_id}_data.csv")


def _json_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


@app.get("/jobs/{job_id}/analytics")
def get_job_analytics(job_id: str) -> dict[str, Any]:
    """
    Aggregated analytics aligned with dashboard.py: player cards, speed series,
    sampled positions for tactical plot, shot breakdown from shot_events.json.
    """
    get_job(job_id)
    if JOBS[job_id].get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Job not completed (status={JOBS[job_id].get('status')})",
        )

    csv_path = _job_dir(job_id) / "data.csv"
    summary_path = _job_dir(job_id) / "summary.json"
    shots_path = _job_dir(job_id) / "shot_events.json"

    summary_raw: dict[str, Any] = {}
    if summary_path.is_file():
        with open(summary_path, encoding="utf-8") as f:
            summary_raw = json.load(f)
    real_shoots = summary_raw.get("player_shoots", {})
    score_top = int(summary_raw.get("score_top", 4))
    score_bottom = int(summary_raw.get("score_bottom", 5))

    shot_report: Optional[dict[str, Any]] = None
    if shots_path.is_file():
        with open(shots_path, encoding="utf-8") as f:
            shot_report = json.load(f)

    shot_breakdown: dict[str, dict[str, int]] = {}
    if shot_report:
        for ev in shot_report.get("shot_events", []) or []:
            pid = str(ev.get("player_id", "?"))
            stype = str(ev.get("shot_type", "?"))
            if pid not in shot_breakdown:
                shot_breakdown[pid] = {}
            shot_breakdown[pid][stype] = shot_breakdown[pid].get(stype, 0) + 1

    ratings_path = _job_dir(job_id) / "ratings.json"
    ratings = DEFAULT_RATINGS
    if ratings_path.is_file():
        try:
            with open(ratings_path, "r", encoding="utf-8") as f:
                ratings = json.load(f)
        except Exception:
            pass

    out: dict[str, Any] = {
        "job_id": job_id,
        "player_cards": [],
        "speed_series": {"time": [], "players": {}},
        "heatmap": {},
        "summary_player_shoots": real_shoots,
        "shots_breakdown_by_player": shot_breakdown,
        "shots_meta": shot_report.get("metadata") if shot_report else None,
        "ratings": ratings,
        "score_top": score_top,
        "score_bottom": score_bottom,
    }

    if not csv_path.is_file():
        out["warning"] = "No data.csv for this job; run pipeline with collect_data enabled."
        return out

    df = pd.read_csv(csv_path)
    player_cards: list[dict[str, Any]] = []
    for pid in range(1, 5):
        vcol = f"player{pid}_Vnorm4"
        dcol = f"player{pid}_distance"
        xcol = f"player{pid}_x"
        ycol = f"player{pid}_y"
        if vcol not in df.columns or dcol not in df.columns:
            continue
        
        # 1. Total strikes
        shoots = int(real_shoots.get(str(pid), 0))
        
        # 2. Net Presence Pct: Time spent within 3.0m of the net (y = 0)
        net_presence_pct = 0.0
        if ycol in df.columns:
            valid_y = df[ycol].dropna()
            if not valid_y.empty:
                net_presence_pct = round(float((valid_y.abs() < 3.0).sum() / len(valid_y)) * 100, 1)

        # 3. Explosive Speed Standard Deviation (Speed Consistency)
        speed_std = 0.0
        valid_v = df[vcol].dropna()
        if not valid_v.empty:
            speed_std = round(float(valid_v.std()) * 3.6, 1)

        # 4. Accurate Calorie Burn based on distance covered
        total_dist = float(df[dcol].fillna(0).sum())
        calorie_burn = round(total_dist * 0.14, 1)

        # 5. Strike Analysis (Overhead preference ratio)
        player_shots = shot_breakdown.get(str(pid), {})
        total_player_shots = sum(player_shots.values())
        overhead_count = player_shots.get("SMASH", 0) + player_shots.get("VIBORA", 0) + player_shots.get("BANDEJA", 0)
        overhead_pct = round((overhead_count / total_player_shots * 100), 1) if total_player_shots > 0 else 0.0

        # Calculate a weighted live match performance score (out of 100)
        # to decide the MATCH MVP based on real high-intensity game stats
        # Clean velocity by removing physical impossibilities (cap at 25 km/h ≈ 6.94 m/s)
        clean_v = df[vcol].dropna().abs()
        clean_v = clean_v[clean_v < 6.94]
        
        avg_speed = round(float(clean_v.mean()) * 3.6, 1) if not clean_v.empty else 0.0
        max_speed = round(float(clean_v.quantile(0.98)) * 3.6, 1) if not clean_v.empty else 0.0

        live_score = round(
            (shoots * 3.0) + 
            (net_presence_pct * 0.35) + 
            (avg_speed * 1.2) + 
            (calorie_burn * 0.08)
        )
        
        # Collective match score for respective team
        team_score = score_top if pid in (1, 2) else score_bottom

        player_cards.append(
            {
                "player_id": pid,
                "label": f"Player {pid}",
                "total_distance_m": round(total_dist, 1),
                "avg_speed_kmh": avg_speed,
                "max_speed_kmh": max_speed,
                "shoots": shoots,
                "score": live_score,
                "team_score": team_score,
                "net_presence_pct": net_presence_pct,
                "speed_consistency_kmh": speed_std,
                "calorie_burn_kcal": calorie_burn,
                "overhead_pct": overhead_pct,
            }
        )
    out["player_cards"] = player_cards

    max_pts = 800
    if "time" in df.columns:
        n = len(df)
        if n <= max_pts:
            sel = df
        else:
            ix = [int(round(i * (n - 1) / (max_pts - 1))) for i in range(max_pts)]
            sel = df.iloc[ix]

        out["speed_series"]["time"] = [
            _json_float(x) for x in sel["time"].tolist()
        ]
        for pid in range(1, 5):
            vcol = f"player{pid}_Vnorm4"
            if vcol in sel.columns:
                smoothed_v = sel[vcol].abs().rolling(window=3, min_periods=1).mean()
                out["speed_series"]["players"][str(pid)] = [
                    min(25.0, round((_json_float(v) or 0.0) * 3.6, 3))
                    for v in smoothed_v.tolist()
                ]

    rng = 42
    for pid in range(1, 5):
        xc, yc = f"player{pid}_x", f"player{pid}_y"
        if xc not in df.columns or yc not in df.columns:
            continue
        pdf = df[[xc, yc]].dropna()
        if pdf.empty:
            continue
        if len(pdf) > 2000:
            pdf = pdf.sample(2000, random_state=rng)
            rng += 1
        out["heatmap"][str(pid)] = {
            "x": [_json_float(x) for x in pdf[xc].tolist()],
            "y": [_json_float(y) for y in pdf[yc].tolist()],
        }

    return out


DEFAULT_RATINGS = {
    "1": { "attack": 85, "defense": 78, "precision": 80, "speed": 82, "stamina": 88, "notes": "Dominant backcourt tactical controller. Focuses on defensive transitions." },
    "2": { "attack": 92, "defense": 85, "precision": 90, "speed": 89, "stamina": 91, "notes": "Aggressive net player, highly efficient overhead smashes." },
    "3": { "attack": 76, "defense": 88, "precision": 75, "speed": 80, "stamina": 85, "notes": "Excellent court coverage, defensive anchor and lob specialist." },
    "4": { "attack": 84, "defense": 80, "precision": 85, "speed": 86, "stamina": 82, "notes": "Well-rounded player, fast transition game and volley control." }
}


@app.get("/jobs/{job_id}/ratings")
def get_job_ratings(job_id: str) -> dict[str, Any]:
    if job_id == "demo" or not job_id:
        ratings_path = RESULTS_DIR / "demo" / "ratings.json"
        if ratings_path.is_file():
            try:
                with open(ratings_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_RATINGS

    get_job(job_id)
    ratings_path = _job_dir(job_id) / "ratings.json"
    if ratings_path.is_file():
        try:
            with open(ratings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_RATINGS


@app.post("/jobs/{job_id}/ratings")
def save_job_ratings(job_id: str, ratings: dict = Body(...)) -> dict[str, str]:
    if job_id == "demo" or not job_id:
        ratings_path = RESULTS_DIR / "demo" / "ratings.json"
        ratings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ratings_path, "w", encoding="utf-8") as f:
            json.dump(ratings, f, indent=2)
        return {"status": "saved"}

    get_job(job_id)
    ratings_path = _job_dir(job_id) / "ratings.json"
    ratings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ratings_path, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)
    return {"status": "saved"}

