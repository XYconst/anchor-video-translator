import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import JobStatus
from pipeline import jobs, run_pipeline, update_job, cleanup_all_sessions, enqueue_job
from utils import LANGUAGES
from services.script_parser import parse_script_docx, parse_script_text

app = FastAPI(title="Video Translator")


@app.on_event("startup")
async def _wipe_stale_sessions():
    """On boot, delete any leftover upload dirs from a previous (crashed) run.

    Voices from a previous run can't be enumerated locally, but the upload dir
    is wiped so no orphan media files remain on disk.
    """
    import shutil
    upload_dir = settings.upload_dir
    if os.path.isdir(upload_dir):
        for entry in os.listdir(upload_dir):
            full = os.path.join(upload_dir, entry)
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)


@app.on_event("shutdown")
async def _cleanup_on_shutdown():
    """On graceful shutdown, kill voices + wipe work dirs for any in-flight jobs."""
    cleanup_all_sessions()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth ---

async def verify_auth(request: Request):
    pass  # Auth disabled


# --- Routes ---

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/languages")
async def get_languages():
    return LANGUAGES


@app.post("/api/translate", dependencies=[Depends(verify_auth)])
async def translate_video(
    video: UploadFile = File(...),
    languages: str = Form(""),  # comma-separated language codes (ignored when script is uploaded)
    script: Optional[UploadFile] = File(None),
    script_text: Optional[str] = Form(None),  # pasted translation text (alternative to .docx)
    caption_box: Optional[str] = Form(None),  # JSON: {x,y,scale,rotation}
    background_urls: Optional[str] = Form(None),  # JSON: [{"url":"...","lang":"en"}, ...]
    caption_style: Optional[str] = Form(None),  # JSON: {font_name, primary_color, ...}
    voiceovers: list[UploadFile] = File(default=[]),  # filename = lang code
):
    import json as _json
    parsed_box = None
    if caption_box:
        try:
            parsed_box = _json.loads(caption_box)
        except Exception:
            parsed_box = None
    parsed_style = None
    if caption_style:
        try:
            parsed_style = _json.loads(caption_style)
        except Exception:
            parsed_style = None
    job_id = str(uuid.uuid4())
    work_dir = os.path.join(settings.upload_dir, job_id)
    os.makedirs(work_dir, exist_ok=True)

    # Save the uploaded video.
    video_name = Path(video.filename or "video").stem
    video_path = os.path.join(work_dir, "original.mp4")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # If a script .docx is provided, parse it (it can dictate language sections).
    scripts: dict[str, str] = {}
    if script is not None and script.filename:
        if not script.filename.lower().endswith(".docx"):
            raise HTTPException(400, "Script must be a .docx file")
        script_path = os.path.join(work_dir, "script.docx")
        with open(script_path, "wb") as f:
            f.write(await script.read())
        try:
            scripts = parse_script_docx(script_path)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse script: {e}")
        if not scripts:
            raise HTTPException(
                400,
                "No language sections detected in the script. Use bold headers like **EN**, **DE**, **GR**.",
            )
        unknown = [l for l in scripts.keys() if l not in LANGUAGES]
        if unknown:
            raise HTTPException(400, f"Script contains unsupported languages: {unknown}")

    # If pasted translation text is provided (and no .docx), parse it.
    if not scripts and script_text and script_text.strip():
        try:
            scripts = parse_script_text(script_text)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse pasted script: {e}")
        if not scripts:
            raise HTTPException(
                400,
                "No language sections detected in the pasted text. Use headers like EN, DE, GR.",
            )
        unknown = [l for l in scripts.keys() if l not in LANGUAGES]
        if unknown:
            raise HTTPException(400, f"Pasted script contains unsupported languages: {unknown}")

    # Save uploaded voiceover audio files. Filename → language code via the
    # same fuzzy lookup the .docx parser uses. Files that don't resolve to a
    # known language are rejected with a clear error.
    from services.voiceover import language_from_filename
    voiceover_audio: dict[str, str] = {}
    if voiceovers:
        for f in voiceovers:
            if not f.filename:
                continue
            lang = language_from_filename(f.filename)
            if not lang or lang not in LANGUAGES:
                raise HTTPException(
                    400,
                    f"Could not infer language from voiceover filename '{f.filename}'. "
                    "Name files like en.mp3 / de.wav / english.m4a.",
                )
            ext = Path(f.filename).suffix or ".wav"
            vo_path = os.path.join(work_dir, f"voiceover_{lang}{ext}")
            with open(vo_path, "wb") as out:
                out.write(await f.read())
            voiceover_audio[lang] = vo_path

    # Decide the final language list:
    #  - If voiceover files are present, ONLY their languages are processed
    #    (the user explicitly drove the lang context via uploaded files).
    #  - Else if a .docx script is present, langs come from the script sections.
    #  - Else fall back to the form's `languages` CSV.
    if voiceover_audio:
        lang_list = sorted(voiceover_audio.keys())
    elif scripts:
        lang_list = [l for l in scripts.keys() if l in LANGUAGES]
    else:
        lang_list = [l.strip() for l in languages.split(",") if l.strip()]
        invalid = [l for l in lang_list if l not in LANGUAGES]
        if invalid:
            raise HTTPException(400, f"Invalid languages: {invalid}")

    if not lang_list:
        raise HTTPException(400, "No languages selected")

    # Optional per-language background URLs.
    bg_urls: dict[str, str] = {}
    if background_urls:
        try:
            raw = _json.loads(background_urls)
            if isinstance(raw, list):
                for entry in raw:
                    if not isinstance(entry, dict):
                        continue
                    u = (entry.get("url") or "").strip()
                    l = (entry.get("lang") or "").strip()
                    if u and l and l in LANGUAGES:
                        bg_urls[l] = u
        except Exception as e:
            raise HTTPException(400, f"Invalid background_urls JSON: {e}")

    # Init job status — starts as "queued"; the worker sets it to "processing".
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="queued",
        current_step="Waiting in queue...",
        languages_total=lang_list,
    )

    # Add to the serial queue (one job runs at a time).
    enqueue_job(
        job_id,
        video_path=video_path,
        languages=lang_list,
        video_name=video_name,
        scripts=scripts,
        caption_box=parsed_box,
        background_urls=bg_urls,
        caption_style=parsed_style,
        voiceover_audio=voiceover_audio,
    )

    return {"job_id": job_id, "languages": lang_list, "script_used": bool(scripts)}


@app.post("/api/cancel/{job_id}", dependencies=[Depends(verify_auth)])
async def cancel_job(job_id: str):
    from pipeline import running_tasks, cancelled_jobs
    # Cooperative flag — checked by sync sections of the pipeline.
    cancelled_jobs.add(job_id)
    task = running_tasks.get(job_id)
    if task is None:
        if job_id in jobs and jobs[job_id].status == "processing":
            jobs[job_id].status = "error"
            jobs[job_id].error = "Cancelled"
        return {"ok": True, "already_finished": True}
    task.cancel()
    return {"ok": True}


@app.get("/api/status/{job_id}", dependencies=[Depends(verify_auth)])
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/download/{job_id}/{language}", dependencies=[Depends(verify_auth)])
async def download_video(job_id: str, language: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    final_path = os.path.join(settings.upload_dir, job_id, language, "final.mp4")
    if not os.path.exists(final_path):
        raise HTTPException(404, "Video not ready yet")

    return FileResponse(
        final_path,
        media_type="video/mp4",
        filename=f"translated_{language}.mp4",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
