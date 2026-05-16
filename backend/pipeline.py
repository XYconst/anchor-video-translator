from __future__ import annotations
import asyncio
import os
import shutil
import time
from typing import Callable, Dict, List, Optional
from pydub import AudioSegment
from models import JobStatus, Segment, StepTiming, WordTiming
from services.ffmpeg import (
    extract_audio, get_video_info, get_audio_duration,
    stretch_audio_segment, replace_audio, burn_captions,
    extract_speaker_audio, retime_video_by_segments,
    mix_voice_with_accompaniment,
)
from services.audio_separation import separate_vocals

# Fit priority: video deforms first, audio only as a fallback.
#
#   MAX_VIDEO_STRETCH = 1.10  → shot may be slowed down by at most 10 %
#                                (orig_slot grown to ≤1.10× original).
#   MAX_VIDEO_COMPRESS = inf  → shot may be sped up without limit
#                                (orig_slot may shrink to any positive value).
#   MAX_AUDIO_SPEEDUP = 1.20  → audio may speed up up to +20 % to fit.
#   MIN_AUDIO_SPEEDUP = 1.00  → audio is NEVER slowed down (was 0.80).
#                                Operator feedback: slowing the audio to
#                                fill the slot left dead air at the end of
#                                the dub ("the video is so slowed that it's
#                                like 2 seconds at the end of nothing
#                                happening"). When the dub is shorter than
#                                the original slot we now compress the
#                                video slot to match the audio instead.
#
# Asymmetric on purpose: a slowed shot looks awkward, but a slightly sped-up
# shot is invisible, so the speed-up takes any load. Audio stretch capped
# tight on the up side because anything past 20 % chipmunks; bottom is
# locked at 1.0 so we never produce trailing silence.
MAX_VIDEO_STRETCH = 1.10        # max video slowdown factor
MAX_VIDEO_COMPRESS = float("inf")  # video can speed up without limit
MAX_AUDIO_SPEEDUP = 1.20
MIN_AUDIO_SPEEDUP = 1.00
from services.gemini import (
    transcribe_and_translate,
    align_script_to_segments,
    transcribe_with_word_timestamps,
    align_words_in_slice,
)
from services.elevenlabs import clone_voice, generate_tts_segment, delete_voice
from services.captions import generate_ass_captions
from config import settings


# In-memory job store
jobs: dict[str, JobStatus] = {}

# Tracks live cleanup state for in-flight jobs so a shutdown / cancellation can
# wipe voices and working files even if the pipeline never reaches its `finally`.
# { job_id: {"voice_ids": set[str], "work_dir": str} }
active_sessions: dict[str, dict] = {}

# Maps job_id -> the asyncio.Task running its pipeline, so /api/cancel can stop it.
running_tasks: dict[str, "object"] = {}

# Cooperative-cancellation flag set. /api/cancel adds the job_id here AND
# calls task.cancel(); the pipeline polls this set at every coarse step so
# synchronous sections (BiRefNet matting, ffmpeg, demucs, time.sleep retries)
# can bail out without waiting for an `await` point.
cancelled_jobs: set[str] = set()

# In-process concurrent worker pool. N async tasks each pull from the same
# queue, so up to N pipelines run at once on one container.
#
# Tunable via MAX_CONCURRENT_JOBS env var. Default 3 — that's the realistic
# sweet spot for Railway Pro: demucs + KIE/Gemini + ElevenLabs + ffmpeg per
# job, and 3 in flight comfortably fits ~8 GB RAM and 2-4 vCPU without
# fighting itself. Above ~5 the container starts to OOM on demucs.
#
# True horizontal scaling (5+ replicas) requires the Redis + Supabase
# refactor planned as a separate session — multi-replica with the current
# in-memory `jobs` dict would 404 status polls that hit the wrong replica.
import os as _os
MAX_CONCURRENT_JOBS = int(_os.environ.get("MAX_CONCURRENT_JOBS", "3"))
_job_queue: asyncio.Queue | None = None
_worker_tasks: list[asyncio.Task] = []


def _get_queue() -> asyncio.Queue:
    global _job_queue
    if _job_queue is None:
        _job_queue = asyncio.Queue()
    return _job_queue


async def _queue_worker(worker_id: int):
    """Pull jobs off the shared queue and run them. Multiple workers share
    the queue, so each worker dequeues independently and the queue itself
    enforces fair FIFO ordering."""
    q = _get_queue()
    while True:
        item = await q.get()
        job_id = item["job_id"]
        try:
            # Skip if already cancelled while queued.
            if job_id in cancelled_jobs:
                if job_id in jobs:
                    jobs[job_id].status = "error"
                    jobs[job_id].error = "Cancelled"
                continue

            # Transition from queued → processing.
            if job_id in jobs:
                jobs[job_id].status = "processing"
                jobs[job_id].current_step = "Starting..."

            await run_pipeline(**item["kwargs"])
        except Exception:
            pass  # run_pipeline handles its own errors
        finally:
            running_tasks.pop(job_id, None)
            q.task_done()


def ensure_worker():
    """Spin up the worker pool if it isn't running yet. Idempotent — safe to
    call on every enqueue. Replaces dead workers so a one-off crash inside a
    worker (rare; run_pipeline has its own try/except) doesn't permanently
    shrink the pool."""
    global _worker_tasks
    # Drop any workers that died since last call.
    _worker_tasks = [t for t in _worker_tasks if not t.done()]
    while len(_worker_tasks) < MAX_CONCURRENT_JOBS:
        idx = len(_worker_tasks)
        _worker_tasks.append(asyncio.create_task(_queue_worker(idx)))


def enqueue_job(job_id: str, **pipeline_kwargs):
    """Add a job to the serial queue."""
    ensure_worker()
    pipeline_kwargs["job_id"] = job_id
    _get_queue().put_nowait({"job_id": job_id, "kwargs": pipeline_kwargs})


async def _run_sync(fn, *args, **kwargs):
    """Run a blocking function in a thread so the event loop stays responsive."""
    loop = asyncio.get_event_loop()
    import functools
    call = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(None, call)


class JobCancelled(Exception):
    """Raised inside the pipeline when /api/cancel was called for this job."""


def check_cancelled(job_id: str):
    if job_id in cancelled_jobs:
        raise JobCancelled()


def cleanup_session(job_id: str, *, remove_files: bool = True):
    """Delete cloned voices and (optionally) the working directory for a job."""
    sess = active_sessions.pop(job_id, None)
    if not sess:
        return
    for voice_id in sess.get("voice_ids", set()):
        try:
            delete_voice(voice_id)
        except Exception:
            pass
    if remove_files:
        work_dir = sess.get("work_dir")
        if work_dir and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


def cleanup_all_sessions():
    """Best-effort cleanup of every tracked session. Called on shutdown."""
    for job_id in list(active_sessions.keys()):
        cleanup_session(job_id)


def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        for k, v in kwargs.items():
            setattr(jobs[job_id], k, v)


# Progress weights — must sum to 1.0.
# Prep covers everything before per-language work; per_lang is split per language.
PROGRESS_PREP_SHARE = 0.10
PROGRESS_LANG_SHARE = 0.90
# Sub-step weights inside a single language (must sum to 1.0).
LANG_SUBSTEPS = [
    ("translate", 0.30),  # script align OR transcribe+translate
    ("tts",       0.35),  # generate per-segment TTS
    ("sync",      0.10),  # fit to timeline
    ("replace",   0.10),  # mux audio
    ("captions",  0.05),  # build .ass
    ("burn",      0.10),  # ffmpeg burn-in
]


# Per-job step timing state, kept out-of-band so JobStatus stays serializable.
# { job_id: {"key": str, "label": str, "started_at": float} }
_step_state: dict[str, dict] = {}


def _set_progress(
    job_id: str,
    step_title: str,
    pct: float,
    *,
    step_key: Optional[str] = None,
    **extra,
):
    """Update job progress.

    `step_key` is a stable identifier for the *step*, while `step_title` is the
    human label (which may change every tick to show a counter like "(3/120)").
    When `step_key` changes, the previous step's elapsed time is appended to
    `step_history` and a new timer starts.
    """
    pct = max(0.0, min(100.0, pct))
    job = jobs.get(job_id)
    if job is None:
        return

    key = step_key or step_title
    now = time.time()
    state = _step_state.get(job_id)

    if state is None or state["key"] != key:
        # Close out the previous step.
        if state is not None:
            elapsed = max(0.0, now - state["started_at"])
            job.step_history = list(job.step_history) + [
                StepTiming(name=state["label"], seconds=round(elapsed, 2))
            ]
        _step_state[job_id] = {"key": key, "label": step_title, "started_at": now}
        job.current_step_started_at = now
    else:
        # Same step — refresh the displayed label (e.g. updated counter).
        state["label"] = step_title

    update_job(job_id, current_step=step_title, progress=pct, **extra)


def _finalize_steps(job_id: str):
    """Close the currently-active step (if any). Call once at end of pipeline."""
    state = _step_state.pop(job_id, None)
    job = jobs.get(job_id)
    if state is None or job is None:
        return
    elapsed = max(0.0, time.time() - state["started_at"])
    job.step_history = list(job.step_history) + [
        StepTiming(name=state["label"], seconds=round(elapsed, 2))
    ]


def _lang_progress(lang_idx: int, total_langs: int, substep_key: str, frac_within: float = 1.0) -> float:
    """Return overall percent after `substep_key` of language `lang_idx` is `frac_within` complete."""
    prep = PROGRESS_PREP_SHARE * 100
    per_lang = (PROGRESS_LANG_SHARE * 100) / max(total_langs, 1)
    finished_subs = 0.0
    for key, weight in LANG_SUBSTEPS:
        if key == substep_key:
            finished_subs += weight * frac_within
            break
        finished_subs += weight
    return prep + per_lang * (lang_idx + finished_subs)


def get_unique_speakers(segments: list[Segment]) -> list[str]:
    """Get list of unique speakers in order of first appearance."""
    seen = set()
    speakers = []
    for seg in segments:
        if seg.speaker not in seen:
            seen.add(seg.speaker)
            speakers.append(seg.speaker)
    return speakers


def merge_minor_speakers(
    segments: list[Segment],
    min_share: float = 0.08,
    min_chars: int = 80,
) -> list[Segment]:
    """Mutate `segments` so any speaker contributing less than `min_share` of total
    speaking time AND fewer than `min_chars` of dialogue is reassigned to the
    speaker that talks immediately around them.

    This kills Gemini's stray "Speaker N" hallucinations on a single short segment
    without forcing a hard cap (so a video with 5 real speakers still works).
    """
    if not segments:
        return segments

    # Tally per-speaker totals.
    durations: dict[str, float] = {}
    char_counts: dict[str, int] = {}
    for s in segments:
        durations[s.speaker] = durations.get(s.speaker, 0.0) + max(0.0, s.end - s.start)
        char_counts[s.speaker] = char_counts.get(s.speaker, 0) + len(s.original or "")
    total_dur = sum(durations.values()) or 1.0

    minors = {
        spk for spk in durations
        if (durations[spk] / total_dur) < min_share and char_counts.get(spk, 0) < min_chars
    }
    if not minors or len(durations) - len(minors) < 1:
        return segments

    # Reassign each minor segment to its nearest non-minor neighbour.
    for i, seg in enumerate(segments):
        if seg.speaker not in minors:
            continue
        prev_spk = next(
            (segments[j].speaker for j in range(i - 1, -1, -1)
             if segments[j].speaker not in minors),
            None,
        )
        next_spk = next(
            (segments[j].speaker for j in range(i + 1, len(segments))
             if segments[j].speaker not in minors),
            None,
        )
        seg.speaker = prev_spk or next_spk or seg.speaker
    return segments


# Characters that should never appear in dubbed dialogue or captions.
_BANNED_PUNCT_RE = __import__("re").compile(r"[!¡‼⁉]+")


def strip_banned_punct(text: str) -> str:
    if not text:
        return text
    return _BANNED_PUNCT_RE.sub(".", text).replace("..", ".").strip()


def _retime_caption_words_per_segment(
    fitted_segments: list[Segment],
    synced_audio_path: str,
    target_language: str,
    work_dir: str,
    progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """For each fitted segment, slice the synced audio to that segment's time
    window and ask Gemini to forced-align the segment text against the slice.
    Returned timestamps are relative to the slice; we add the segment offset
    and write them back into `seg.words`.

    Falls back to whatever proportional spacing fit_segments_to_timeline set
    if any single segment fails — never crashes the job.
    """
    from services.voiceover import slice_audio
    import os as _os

    n = len(fitted_segments)
    for i, seg in enumerate(fitted_segments):
        if progress:
            try: progress(i, n)
            except Exception: pass
        words = seg.words
        if not words:
            continue
        text = seg.translated or seg.original
        if not text or not text.strip():
            continue
        slice_path = _os.path.join(work_dir, f"capalign_{i:04d}.wav")
        try:
            # IMPORTANT: don't run silenceremove on the alignment slice — we
            # need the timestamps to be relative to the original segment
            # window, so any leading/trailing silence has to be preserved.
            slice_audio(
                synced_audio_path,
                float(seg.start),
                float(seg.end),
                slice_path,
                clean_silences=False,
            )
            aligned = align_words_in_slice(slice_path, text, target_language)
        except Exception as e:
            print(f"[caption_align seg {i}] {e}")
            aligned = []
        finally:
            try: _os.remove(slice_path)
            except OSError: pass

        if not aligned:
            continue

        # Map per-word aligned timestamps onto the segment's existing
        # `words` list. We keep the segment's word TEXT (which is what the
        # caption renderer uses) and overwrite only the timings.
        if len(aligned) >= len(words):
            chosen = aligned[: len(words)]
        else:
            # Stretch fewer aligned timings across more words proportionally.
            if len(aligned) == 0:
                continue
            span_s = aligned[0]["start"]
            span_e = aligned[-1]["end"]
            if span_e <= span_s:
                continue
            step = (span_e - span_s) / len(words)
            chosen = [
                {"start": span_s + j * step, "end": span_s + (j + 1) * step}
                for j in range(len(words))
            ]
        for w, t in zip(words, chosen):
            abs_start = float(seg.start) + float(t["start"])
            abs_end = float(seg.start) + float(t["end"])
            abs_end = min(abs_end, float(seg.end))
            abs_start = max(abs_start, float(seg.start))
            if abs_end <= abs_start:
                abs_end = abs_start + 0.05
            w.start = round(abs_start, 3)
            w.end = round(abs_end, 3)


def _retime_caption_words(
    fitted_segments: list[Segment],
    word_timings: list[dict],
) -> None:
    """Re-time `fitted_segments[*].words` to match the spoken cadence in the
    synced audio.

    Strategy: for each segment, find every Gemini word whose midpoint falls
    inside the segment's `[start, end]` window. Take the **timestamps** from
    those Gemini words but keep the **text** from the original segment, then
    distribute the original word texts across the captured time-spans.

    This decouples caption text from Gemini's tokenisation (which never
    matches the source text exactly), and uses the segment time window as
    the only anchor — so a missing word here or an extra one there can't
    cause drift across the whole video like the previous count-based
    cursor did.
    """
    if not word_timings or not fitted_segments:
        return

    # Pre-extract midpoints for fast bisect.
    starts = [float(w["start"]) for w in word_timings]
    ends = [float(w["end"]) for w in word_timings]
    mids = [(s + e) / 2.0 for s, e in zip(starts, ends)]

    import bisect

    for seg in fitted_segments:
        n = len(seg.words)
        if n == 0:
            continue
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        # Find the first word whose midpoint >= seg_start, and the first
        # whose midpoint > seg_end. Slice between.
        lo = bisect.bisect_left(mids, seg_start)
        hi = bisect.bisect_right(mids, seg_end)
        captured = list(zip(starts[lo:hi], ends[lo:hi]))

        # Determine the time span for this segment from the captured words.
        if captured:
            span_start = captured[0][0]
            span_end = captured[-1][1]
        else:
            # No Gemini words in this window — leave as-is (proportional
            # spacing inside the segment slot, which fit_segments_to_timeline
            # already set up).
            continue

        # Clamp to the segment's window so caption events never spill into
        # the next segment.
        span_start = max(span_start, seg_start)
        span_end = min(span_end, seg_end)
        if span_end - span_start < 0.05:
            continue

        if len(captured) >= n:
            # Distribute the n original-text words across the first n
            # captured timing slots, then stretch the last word to the end
            # of the captured span. This preserves the spoken pacing.
            chosen = captured[:n]
            for j, (w, (cs, ce)) in enumerate(zip(seg.words, chosen)):
                cs = max(cs, seg_start)
                ce = min(ce, seg_end)
                if ce <= cs:
                    ce = cs + 0.05
                w.start = round(cs, 3)
                w.end = round(ce, 3)
            # Snap the final word's end to the captured span end.
            if seg.words:
                seg.words[-1].end = round(min(span_end, seg_end), 3)
        else:
            # More words in our text than Gemini found — proportionally
            # distribute the original words across the captured span.
            step = (span_end - span_start) / n
            for j, w in enumerate(seg.words):
                w.start = round(span_start + j * step, 3)
                w.end = round(span_start + (j + 1) * step, 3)


def fit_segments_to_timeline(
    tts_paths: list[str],
    segments: list[Segment],
    video_duration: float,
    work_dir: str,
    output_path: str,
) -> tuple[list[Segment], list[tuple[float, float, float]], float]:
    """Speed-adjust each TTS segment to fit its time slot, then concatenate.

    When TTS would need to be sped up beyond `MAX_AUDIO_SPEEDUP`, the segment's
    slot is grown to keep audio at the cap; the caller stretches the underlying
    video scene by the same amount so picture and voice stay in sync.

    Returns:
        fitted_segments: segments with timings on the NEW (possibly stretched) timeline
        retimes: list of (orig_start, orig_end, new_duration) for video stretching
        new_video_duration: total duration of the new audio/video timeline
    """
    # Keep the mix at studio quality (48 kHz mono). pydub's default silent()
    # is 11025 Hz, which crushes everything that gets appended down to that.
    TARGET_SR = 48000
    combined = AudioSegment.silent(duration=0, frame_rate=TARGET_SR)
    fitted_segments: list[Segment] = []
    retimes: list[tuple[float, float, float]] = []
    time_offset = 0.0  # cumulative shift introduced by stretched scenes

    for i, (seg, tts_path) in enumerate(zip(segments, tts_paths)):
        orig_target = seg.end - seg.start
        tts_duration = get_audio_duration(tts_path)

        # Fit-priority decision (per segment):
        #
        # Audio LONGER than slot (ratio > 1):
        #   • Slow the video to absorb up to +10 %.
        #   • If still longer, speed the audio up by ≤5 %.
        #   • Anything beyond ratio == 1.10 × 1.05 ≈ 1.155 will get clipped
        #     to the slot at the end of this loop.
        #
        # Audio SHORTER than slot (ratio < 1):
        #   • Speed the video up to match the audio duration EXACTLY.
        #     Video has no compression cap, so audio always plays at native.
        if orig_target <= 0 or tts_duration <= 0:
            new_target = orig_target
        else:
            ratio = tts_duration / orig_target
            if ratio > 1.0:
                # Audio is longer. First grow the slot using the video stretch
                # budget, capped at MAX_VIDEO_STRETCH.
                grown = min(tts_duration, orig_target * MAX_VIDEO_STRETCH)
                new_target = grown
                retimes.append((seg.start, seg.end, new_target))
                # If even max stretch isn't enough, audio speeds up the rest
                # (capped by MAX_AUDIO_SPEEDUP at the factor calc below).
            elif ratio < 1.0:
                # Audio is shorter. Speed the video up to match — no cap.
                new_target = tts_duration
                retimes.append((seg.start, seg.end, new_target))
            else:
                new_target = orig_target

        # Gap before this segment (preserved on the original timeline; the
        # cumulative offset is applied via time_offset below).
        if i == 0:
            gap = seg.start
        else:
            gap = seg.start - segments[i - 1].end
        if gap > 0:
            combined += AudioSegment.silent(duration=int(gap * 1000), frame_rate=TARGET_SR)

        # Speed-adjust to fit the (possibly grown/shrunk) slot, then clamp
        # the audio time-stretch to ±5 %. Anything beyond that is left for
        # the trim/pad pass below.
        if tts_duration > 0 and new_target > 0:
            factor = tts_duration / new_target
            factor = max(MIN_AUDIO_SPEEDUP, min(MAX_AUDIO_SPEEDUP, factor))
        else:
            factor = 1.0

        stretched_path = os.path.join(work_dir, f"fitted_{i:04d}.wav")
        stretch_audio_segment(tts_path, stretched_path, factor)
        stretched_audio = AudioSegment.from_wav(stretched_path)
        # Force every appended slice to the target rate so concat doesn't
        # silently downmix everything to the lower of the two sample rates.
        if stretched_audio.frame_rate != TARGET_SR:
            stretched_audio = stretched_audio.set_frame_rate(TARGET_SR)
        if stretched_audio.channels != 1:
            stretched_audio = stretched_audio.set_channels(1)

        target_ms = int(new_target * 1000)
        # NEVER hard-trim the audio — that cuts words off the end. If we
        # still overshoot after the speed-fit + 5 % cap, grow the slot to
        # contain the whole utterance and let the video stretch absorb it.
        if len(stretched_audio) > target_ms:
            extra_ms = len(stretched_audio) - target_ms
            new_target = len(stretched_audio) / 1000.0
            target_ms = len(stretched_audio)
            # Mark this segment as needing extra video stretch beyond the
            # original 10 % budget — we'd rather have a slightly slowed
            # shot than missing words.
            if retimes and retimes[-1][0] == seg.start:
                retimes[-1] = (seg.start, seg.end, new_target)
            else:
                retimes.append((seg.start, seg.end, new_target))
        elif len(stretched_audio) < target_ms:
            # Audio came out shorter than the original slot. Old behaviour
            # was to pad with silence so the video kept playing — but that
            # produced dead air at the end of every undershot segment, and
            # at the end of the dub overall. Instead, shrink the slot to
            # the actual audio length: the video clip gets compressed to
            # match the dub, no padding. This is the symmetric counterpart
            # to the "audio too long → grow slot" branch above.
            new_target = len(stretched_audio) / 1000.0
            target_ms = len(stretched_audio)
            if retimes and retimes[-1][0] == seg.start:
                retimes[-1] = (seg.start, seg.end, new_target)
            else:
                retimes.append((seg.start, seg.end, new_target))

        combined += stretched_audio

        try:
            os.remove(stretched_path)
        except OSError:
            pass

        # Segment timings on the NEW timeline.
        new_start = seg.start + time_offset
        new_end = new_start + new_target
        time_offset += (new_target - orig_target)

        translated_words = seg.translated.split()
        new_words = []
        if translated_words:
            word_dur = new_target / len(translated_words)
            for j, w in enumerate(translated_words):
                new_words.append(WordTiming(
                    word=w,
                    start=round(new_start + j * word_dur, 3),
                    end=round(new_start + (j + 1) * word_dur, 3),
                ))

        fitted_segments.append(Segment(
            start=round(new_start, 3),
            end=round(new_end, 3),
            original=seg.original,
            translated=seg.translated,
            words=new_words,
            speaker=seg.speaker,
        ))

    new_video_duration = video_duration + time_offset

    current_ms = len(combined)
    target_ms = int(new_video_duration * 1000)
    if current_ms < target_ms:
        combined += AudioSegment.silent(duration=target_ms - current_ms, frame_rate=TARGET_SR)
    elif current_ms > target_ms:
        combined = combined[:target_ms]

    # Export at 48 kHz / 16-bit. pydub honours frame_rate from the segment;
    # set bytes-per-sample explicitly so the file is studio-grade WAV.
    combined = combined.set_frame_rate(TARGET_SR).set_sample_width(2)
    combined.export(output_path, format="wav")
    return fitted_segments, retimes, new_video_duration


async def run_pipeline(
    job_id: str,
    video_path: str,
    languages: list[str],
    video_name: str,
    scripts: dict[str, str] | None = None,
    caption_box: dict | None = None,
    background_urls: dict[str, str] | None = None,
    caption_style: dict | None = None,
    voiceover_audio: dict[str, str] | None = None,
):
    """Run the full translation pipeline for all requested languages.

    `scripts` is an optional `{lang_code: pre_translated_text}` mapping. When a
    language is present in `scripts`, the pre-translated script is aligned to
    the video segments instead of running machine translation, and captions
    are sourced from the script as well.
    """
    scripts = scripts or {}
    voiceover_audio = voiceover_audio or {}
    work_dir = os.path.join(settings.upload_dir, job_id)
    os.makedirs(work_dir, exist_ok=True)

    voice_ids: dict[str, str] = {}  # speaker -> voice_id
    active_sessions[job_id] = {"voice_ids": set(), "work_dir": work_dir}
    try:
        total_langs = len(languages)

        background_urls = background_urls or {}

        check_cancelled(job_id)
        # Step 1: Get video info
        _set_progress(job_id, "Analyzing video...", 1, step_key="analyze")
        info = await _run_sync(get_video_info, video_path)
        video_duration = info["duration"]

        # Step 2: Extract audio
        _set_progress(job_id, "Extracting audio...", 3, step_key="extract_audio")
        audio_path = os.path.join(work_dir, "original_audio.wav")
        await _run_sync(extract_audio, video_path, audio_path)

        # Step 2b: Separate vocals from music/SFX so we can preserve the
        # original soundtrack under the new dub. Best-effort: if Demucs fails
        # (e.g. very short clip), we just continue without an accompaniment.
        accompaniment_path: Optional[str] = None
        try:
            _set_progress(
                job_id, "Separating music from vocals...", 4, step_key="separate_audio"
            )
            stems_dir = os.path.join(work_dir, "stems")
            _vocals_path, accompaniment_path = await _run_sync(separate_vocals, audio_path, stems_dir)
        except Exception as e:
            print(f"[pipeline] demucs separation failed: {e} — proceeding without music bed")
            accompaniment_path = None

        # Step 3: Transcribe first to identify speakers
        _set_progress(job_id, "Identifying speakers...", 5, step_key="transcribe")
        initial_segments = await transcribe_and_translate(audio_path, languages[0])
        # Strip "!" everywhere — both translated text and word-level captions.
        for s in initial_segments:
            s.translated = strip_banned_punct(s.translated)
            for w in s.words:
                w.word = strip_banned_punct(w.word)
        # Collapse hallucinated minor speakers into their neighbours.
        merge_minor_speakers(initial_segments)
        speakers = get_unique_speakers(initial_segments)

        # Skip voice cloning entirely if every requested language has a
        # pre-recorded voiceover — we won't be calling ElevenLabs TTS at all.
        any_needs_tts = any(l not in voiceover_audio for l in languages)
        if any_needs_tts:
            _set_progress(job_id, f"Found {len(speakers)} speakers, cloning voices...", 7, step_key="clone_voices")
            # Step 4: Clone each speaker's voice
            for speaker in speakers:
                check_cancelled(job_id)
                speaker_audio_path = os.path.join(work_dir, f"speaker_{speaker.replace(' ', '_')}.wav")
                await _run_sync(extract_speaker_audio, audio_path, initial_segments, speaker, speaker_audio_path)
                voice_id = await _run_sync(clone_voice, speaker_audio_path, f"{video_name}_{speaker.replace(' ', '_')}")
                voice_ids[speaker] = voice_id
                active_sessions[job_id]["voice_ids"].add(voice_id)
                try:
                    os.remove(speaker_audio_path)
                except OSError:
                    pass

        # --- Analysis pass: precompute the speaker safe-zone trajectory ---
        # We do this ONCE up front, before any per-language work, so the
        # matting in replace_background uses a stable, smoothed bbox per
        # frame instead of re-detecting (and occasionally flickering) each
        # time. Only worth running if at least one language asks for a
        # background swap.
        precomputed_safe_zones = None
        if background_urls:
            try:
                from services.background import precompute_safe_zones
                check_cancelled(job_id)
                _set_progress(job_id, "Locking on speaker...", 9, step_key="lock_speaker")
                precomputed_safe_zones = await _run_sync(
                    precompute_safe_zones,
                    video_path,
                    progress_cb=lambda f, label: _set_progress(
                        job_id, label, 9 + f * 0.5, step_key="lock_speaker"
                    ),
                    cancel_check=lambda: check_cancelled(job_id),
                )
            except Exception as e:
                print(f"[pipeline] safe-zone precompute failed: {e} — falling back to per-frame")
                precomputed_safe_zones = None

        # Process each language
        for lang_idx, lang in enumerate(languages):
            check_cancelled(job_id)
            lang_dir = os.path.join(work_dir, lang)
            os.makedirs(lang_dir, exist_ok=True)

            # Decide the source of segments for this language:
            #  - script provided                      -> align script to initial_segments (no MT)
            #  - voiceover provided without script    -> transcribe VO, use as script
            #  - first language without either        -> reuse initial transcription
            #  - otherwise                            -> re-transcribe & translate
            lang_name = lang.upper()
            # Earlier we also transcribed uploaded VOs here and injected the
            # transcript into `scripts` so the script-alignment path would
            # handle them. That was wrong: align_script_to_segments forces
            # the text to fit `initial_segments` (first-language timings),
            # which produced garbled chunks for non-first languages whose
            # sentence structure didn't match. Downstream VO alignment then
            # fell back to 0.5 s per segment and the output collapsed to a
            # few seconds of disjointed audio. VO-without-script now falls
            # through to the plain `transcribe_and_translate` path below,
            # which produces properly-structured per-language segments that
            # align_voiceover_to_segments can actually match against the VO.
            if lang in scripts and scripts[lang]:
                _set_progress(
                    job_id,
                    f"Aligning script for {lang_name}...",
                    _lang_progress(lang_idx, total_langs, "translate", 0.0),
                    step_key=f"translate:{lang}",
                    current_language=lang,
                )
                segments, warns = await align_script_to_segments(
                    scripts[lang], initial_segments, lang,
                )
                for w in warns:
                    print(f"[script:{lang}] WARN {w}")
            elif lang_idx == 0:
                _set_progress(
                    job_id,
                    f"Preparing {lang_name}...",
                    _lang_progress(lang_idx, total_langs, "translate", 0.0),
                    step_key=f"translate:{lang}",
                    current_language=lang,
                )
                segments = initial_segments
            else:
                _set_progress(
                    job_id,
                    f"Translating to {lang_name}...",
                    _lang_progress(lang_idx, total_langs, "translate", 0.0),
                    step_key=f"translate:{lang}",
                    current_language=lang,
                )
                segments = await transcribe_and_translate(audio_path, lang)

            # Strip "!" from this language's text + words before TTS/captions.
            for s in segments:
                s.translated = strip_banned_punct(s.translated)
                for w in s.words:
                    w.word = strip_banned_punct(w.word)

            # Step 5: Either slice the user's pre-recorded voiceover OR
            # generate per-segment TTS via ElevenLabs.
            n_segs = max(len(segments), 1)
            if lang in voiceover_audio:
                from services.voiceover import (
                    align_voiceover_to_segments,
                    preprocess_voiceover,
                    slice_audio,
                )
                _set_progress(
                    job_id,
                    f"Preparing voiceover for {lang_name}...",
                    _lang_progress(lang_idx, total_langs, "tts", 0.0),
                    step_key=f"tts:{lang}",
                )
                vo_path = voiceover_audio[lang]
                # Normalise channel layout to dual-mono stereo and match
                # loudness to the original video's audio bed. Falls back to
                # the raw file if anything in the preprocess chain errors.
                try:
                    pre_path = os.path.join(lang_dir, f"vo_preprocessed.wav")
                    vo_path = preprocess_voiceover(vo_path, audio_path, pre_path)
                except Exception as e:
                    print(
                        f"[pipeline] VO preprocess failed for {lang}: {e} "
                        f"— using uploaded file as-is"
                    )
                _set_progress(
                    job_id,
                    f"Aligning voiceover for {lang_name}...",
                    _lang_progress(lang_idx, total_langs, "tts", 0.0),
                    step_key=f"tts:{lang}",
                )
                vo_ranges = align_voiceover_to_segments(vo_path, segments, lang)
                # Gemini's alignment is only accurate to ~100–200 ms; without
                # a pad on each side the raw cut chops word attacks/releases.
                # We widen up to WIDEN_MAX per side but cap each pad at half
                # the gap to the neighbouring segment so widening can't steal
                # their word. A zero-length gap (back-to-back segments)
                # yields a zero pad on that side — the Gemini boundary is
                # taken as authoritative there.
                WIDEN_MAX = 0.25
                tts_paths = []
                for i, (vs, ve) in enumerate(vo_ranges):
                    check_cancelled(job_id)
                    prev_end = vo_ranges[i - 1][1] if i > 0 else 0.0
                    next_start = (
                        vo_ranges[i + 1][0]
                        if i + 1 < len(vo_ranges)
                        else float("inf")
                    )
                    lead_pad = min(WIDEN_MAX, max(0.0, (vs - prev_end) / 2))
                    tail_pad = min(WIDEN_MAX, max(0.0, (next_start - ve) / 2))
                    out_path = os.path.join(lang_dir, f"vo_{i:04d}.wav")
                    slice_audio(
                        vo_path, vs, ve, out_path,
                        lead_pad=lead_pad, tail_pad=tail_pad,
                    )
                    tts_paths.append(out_path)
                    _set_progress(
                        job_id,
                        f"Slicing voiceover for {lang_name} ({i + 1}/{n_segs})...",
                        _lang_progress(lang_idx, total_langs, "tts", (i + 1) / n_segs),
                        step_key=f"tts:{lang}",
                    )
            else:
                _set_progress(
                    job_id,
                    f"Generating voices for {lang_name}...",
                    _lang_progress(lang_idx, total_langs, "tts", 0.0),
                    step_key=f"tts:{lang}",
                )
                tts_paths = []
                for i, seg in enumerate(segments):
                    check_cancelled(job_id)
                    out_path = os.path.join(lang_dir, f"tts_{i:04d}.wav")
                    speaker_voice = voice_ids.get(seg.speaker) or list(voice_ids.values())[0]
                    await _run_sync(generate_tts_segment, seg.translated, speaker_voice, out_path)
                    tts_paths.append(out_path)
                    _set_progress(
                        job_id,
                        f"Generating voices for {lang_name} ({i + 1}/{n_segs})...",
                        _lang_progress(lang_idx, total_langs, "tts", (i + 1) / n_segs),
                        step_key=f"tts:{lang}",
                    )

            # Step 6: Fit each segment to its exact time slot
            _set_progress(
                job_id,
                f"Syncing to scenes for {lang_name}...",
                _lang_progress(lang_idx, total_langs, "sync", 0.0),
                step_key=f"sync:{lang}",
            )
            synced_audio = os.path.join(lang_dir, "synced_audio.wav")
            synced_segments, video_retimes, new_video_duration = await _run_sync(
                fit_segments_to_timeline,
                tts_paths, segments, video_duration, lang_dir, synced_audio,
            )

            # Optional: replace the visual background with a per-language blog URL
            # before the audio mux. Pure pre-step for THIS language only — does
            # not touch the original video_path used by other languages.
            lang_video_source = video_path
            if lang in background_urls:
                from services.background import replace_background

                def _bg_progress(frac: float, label: str, _li=lang_idx, _lang=lang):
                    _set_progress(
                        job_id,
                        f"{label} ({lang_name})",
                        _lang_progress(_li, total_langs, "replace", frac * 0.5),
                        step_key=f"background:{_lang}",
                    )

                bg_out = os.path.join(lang_dir, "with_new_bg.mp4")
                await _run_sync(
                    replace_background,
                    video_path=video_path,
                    blog_url=background_urls[lang],
                    out_path=bg_out,
                    work_dir=lang_dir,
                    progress_cb=_bg_progress,
                    segments=segments,
                    cancel_check=lambda _jid=job_id: check_cancelled(_jid),
                    safe_zones=precomputed_safe_zones,
                )
                lang_video_source = bg_out

            # Step 6b: Stretch any video scenes whose voiceover wouldn't fit at
            # the audio speed cap. Only runs when fit_segments_to_timeline
            # actually grew at least one slot.
            if video_retimes:
                retimed_video = os.path.join(lang_dir, "video_retimed.mp4")
                await _run_sync(
                    retime_video_by_segments,
                    lang_video_source,
                    video_retimes,
                    video_duration,
                    retimed_video,
                    lang_dir,
                )
                lang_video_source = retimed_video

            # Step 7: Mix dubbed voice over the original music/SFX bed (with
            # sidechain ducking) and then replace the video's audio track.
            _set_progress(
                job_id,
                f"Mixing music for {lang_name}..." if accompaniment_path else f"Replacing audio for {lang_name}...",
                _lang_progress(lang_idx, total_langs, "replace", 0.5 if lang in background_urls else 0.0),
                step_key=f"replace_audio:{lang}",
            )
            final_audio_path = synced_audio
            if accompaniment_path:
                try:
                    mixed_path = os.path.join(lang_dir, "synced_with_music.wav")
                    await _run_sync(
                        mix_voice_with_accompaniment,
                        synced_audio, accompaniment_path, mixed_path,
                    )
                    final_audio_path = mixed_path
                except Exception as e:
                    print(f"[pipeline] mix with accompaniment failed: {e} — using voice only")
                    final_audio_path = synced_audio
            video_no_captions = os.path.join(lang_dir, "video_no_captions.mp4")
            await _run_sync(replace_audio, lang_video_source, final_audio_path, video_no_captions)

            # Step 8 + 9: Generate + burn captions, unless the user disabled them.
            captions_enabled = True
            if isinstance(caption_style, dict) and caption_style.get("enabled") is False:
                captions_enabled = False

            # Re-time caption words against the *actual fitted audio*. We
            # do this PER SEGMENT — each segment's audio is sliced and
            # forced-aligned with its known text, which is far more accurate
            # than asking Gemini to transcribe the whole synced audio at once.
            if captions_enabled:
                try:
                    _set_progress(
                        job_id,
                        f"Timing captions to audio for {lang_name}...",
                        _lang_progress(lang_idx, total_langs, "captions", 0.0),
                        step_key=f"caption_align:{lang}",
                    )

                    def _cap_progress(idx: int, total: int, _li=lang_idx, _ln=lang_name):
                        if total <= 0:
                            return
                        _set_progress(
                            job_id,
                            f"Aligning captions for {_ln} ({idx + 1}/{total})...",
                            _lang_progress(_li, total_langs, "captions", (idx + 1) / total),
                            step_key=f"caption_align:{lang}",
                        )

                    await _run_sync(
                        _retime_caption_words_per_segment,
                        synced_segments,
                        final_audio_path,
                        lang,
                        lang_dir,
                        progress=_cap_progress,
                    )
                except Exception as e:
                    print(f"[pipeline] caption word retiming failed: {e} — using even spacing")

            final_path = os.path.join(lang_dir, "final.mp4")
            if not captions_enabled:
                _set_progress(
                    job_id,
                    f"Finalising {lang_name} (no captions)...",
                    _lang_progress(lang_idx, total_langs, "burn", 0.0),
                    step_key=f"burn:{lang}",
                )
                # Just promote video_no_captions to final.mp4 (no re-encode).
                shutil.copyfile(video_no_captions, final_path)
            else:
                _set_progress(
                    job_id,
                    f"Generating captions for {lang_name}...",
                    _lang_progress(lang_idx, total_langs, "captions", 0.0),
                    step_key=f"captions:{lang}",
                )
                ass_path = os.path.join(lang_dir, "captions.ass")
                generate_ass_captions(
                    synced_segments, ass_path,
                    video_width=info["width"],
                    video_height=info["height"],
                    caption_box=caption_box,
                    caption_style=caption_style,
                )

                _set_progress(
                    job_id,
                    f"Burning captions for {lang_name}...",
                    _lang_progress(lang_idx, total_langs, "burn", 0.0),
                    step_key=f"burn:{lang}",
                )
                fonts_dir = os.path.dirname(settings.font_path)
                await _run_sync(burn_captions, video_no_captions, ass_path, final_path, fonts_dir)

            # Mark language as done
            done = jobs[job_id].languages_done + [lang]
            update_job(job_id, languages_done=done)

            # Cleanup
            for f in tts_paths + [synced_audio, video_no_captions]:
                try:
                    os.remove(f)
                except OSError:
                    pass

        _finalize_steps(job_id)
        _set_progress(job_id, "Done!", 100, step_key="done")
        _finalize_steps(job_id)
        update_job(job_id, status="completed")
        # Success: drop voices but keep the work_dir so /api/download still serves final.mp4.
        cleanup_session(job_id, remove_files=False)

    except BaseException as e:
        # Covers normal exceptions, asyncio.CancelledError, KeyboardInterrupt,
        # and our own cooperative JobCancelled — any interruption wipes voices
        # AND the entire session directory.
        import asyncio
        is_cancel = isinstance(e, (asyncio.CancelledError, JobCancelled))
        _finalize_steps(job_id)
        update_job(
            job_id,
            status="error",
            error="Cancelled" if is_cancel else str(e),
        )
        cleanup_session(job_id, remove_files=True)
        cancelled_jobs.discard(job_id)
        if isinstance(e, JobCancelled):
            return  # don't re-raise our internal sentinel
        raise
