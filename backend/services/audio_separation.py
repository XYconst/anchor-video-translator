"""Audio source separation for the dubbing pipeline.

Splits an input audio file into a vocals stem and an accompaniment stem so
the pipeline can replace only the voice while preserving the original
music and sound effects.

Two backends, in preference order:

  1. ElevenLabs Audio Isolation — cloud API, much cleaner separation on
     speech-dominant content. Primary path.
  2. Demucs (htdemucs_ft)       — local, no network / API cost, used as a
     fallback when ElevenLabs is disabled or fails.

Which one runs is controlled by the ``USE_ELEVENLABS_ISOLATION`` env var
(``1``/``true`` → try ElevenLabs first, default; ``0``/``false`` → go
straight to Demucs). A missing/invalid ElevenLabs key also triggers the
fallback automatically.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional

from config import settings


# Two-stem mode: only "vocals" vs "no_vocals". Faster than the full 4-stem
# split and that's all the dubbing pipeline actually needs.
#
# htdemucs_ft is the fine-tuned ensemble variant of htdemucs — it runs four
# fine-tuned models and averages them. ~4x slower than plain htdemucs but
# produces dramatically cleaner stems with far less vocal bleed / music
# smearing, which matters a lot when you replace the voice and keep the
# music bed: artifacts in the accompaniment become obvious without the
# original voice masking them.
_MODEL = "htdemucs_ft"


def _eleven_enabled() -> bool:
    # Uses ElevenLabs' /v1/music/stem-separation endpoint, which returns a
    # ZIP of real separated stems (vocals + drums/bass/other). Unlike the
    # audio_isolation endpoint (which only gives the voice), this gives us
    # a proper accompaniment stem directly — no broken subtraction trick.
    flag = os.environ.get("USE_ELEVENLABS_ISOLATION", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(getattr(settings, "elevenlabs_api_key", "") or "")


def separate_vocals(
    audio_path: str,
    out_dir: str,
    python_executable: Optional[str] = None,
) -> tuple[str, str]:
    """Return ``(vocals_wav, accompaniment_wav)`` for ``audio_path``.

    Tries ElevenLabs voice isolation first (when enabled + key available);
    falls back to Demucs (``htdemucs_ft``) if that errors. Both stems are
    written into ``out_dir``. Raises ``RuntimeError`` only when BOTH paths
    fail.
    """
    if _eleven_enabled():
        try:
            from services.audio_isolation_eleven import isolate_voice_elevenlabs
            print("[separate_vocals] using ElevenLabs audio isolation")
            return isolate_voice_elevenlabs(audio_path, out_dir)
        except Exception as e:
            print(
                f"[separate_vocals] ElevenLabs isolation failed: {e} "
                f"— falling back to Demucs"
            )

    return _separate_vocals_demucs(audio_path, out_dir, python_executable)


def _separate_vocals_demucs(
    audio_path: str,
    out_dir: str,
    python_executable: Optional[str] = None,
) -> tuple[str, str]:
    """Demucs fallback. Same return shape as :func:`separate_vocals`."""
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, "_demucs")
    if os.path.isdir(work):
        shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)

    py = python_executable or sys.executable
    proc = subprocess.run(
        [
            py, "-m", "demucs",
            "-n", _MODEL,
            "--two-stems=vocals",
            "-o", work,
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"demucs failed (rc={proc.returncode}):\n{proc.stderr or proc.stdout}"
        )

    base = os.path.splitext(os.path.basename(audio_path))[0]
    src_dir = os.path.join(work, _MODEL, base)
    src_vocals = os.path.join(src_dir, "vocals.wav")
    src_accomp = os.path.join(src_dir, "no_vocals.wav")
    if not (os.path.isfile(src_vocals) and os.path.isfile(src_accomp)):
        raise RuntimeError(
            f"demucs output missing: expected {src_vocals} and {src_accomp}"
        )

    vocals = os.path.join(out_dir, "vocals.wav")
    accomp = os.path.join(out_dir, "accompaniment.wav")
    shutil.move(src_vocals, vocals)
    shutil.move(src_accomp, accomp)
    shutil.rmtree(work, ignore_errors=True)
    return vocals, accomp
