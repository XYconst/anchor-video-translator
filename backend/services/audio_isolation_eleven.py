"""ElevenLabs stem separation → (vocals, accompaniment).

Uses ``POST /v1/music/stem-separation`` (SDK: ``client.music.separate_stems``),
which returns a ZIP archive containing every separated stem as an individual
audio file (vocals, drums, bass, other, …). We pull the vocals out, mix the
remaining stems into a single accompaniment track, and return both paths so
the rest of the pipeline can consume them exactly like Demucs output.

Why this over the ``audio_isolation`` endpoint:
  • ``audio_isolation`` only returns the isolated VOICE. Deriving the
    accompaniment by subtraction failed because the re-processed voice is
    not phase-aligned to the source, so the "subtraction" actually ADDS
    energy and the original voice ends up louder, not quieter, in the mix.
  • ``separate_stems`` returns both sides of the split natively, so no
    subtraction trick is needed and the accompaniment is a true stem.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import zipfile
from typing import Iterable

from elevenlabs import ElevenLabs

from config import settings


# MP3 first: the endpoint returns raw PCM (no header) for ``pcm_*`` formats,
# which means we'd have to guess channel count at decode time. mp3 is a
# self-describing container so ffmpeg handles it directly, and 128 kbps mp3
# is inaudibly different from lossless in the final ducked mix. PCM formats
# are kept as fallbacks for tiers / account states where mp3 might fail.
_OUTPUT_FORMATS = ["mp3_44100_128", "pcm_44100", "pcm_24000", "pcm_16000"]


def _raw_pcm_flags(fmt: str) -> list[str] | None:
    """Return ffmpeg input flags for a raw-PCM format string, or None when
    the format is self-describing (mp3, etc.).
    """
    m = re.match(r"pcm_(\d+)", fmt)
    if not m:
        return None
    sr = m.group(1)
    # ElevenLabs' raw PCM output is stereo 16-bit signed little-endian for
    # stem files. If that ever changes, ffmpeg will error loudly here and
    # we fall through to the next format.
    return ["-f", "s16le", "-ar", sr, "-ac", "2"]


def _collect_zip_bytes(stream: Iterable[bytes]) -> bytes:
    buf = io.BytesIO()
    for chunk in stream:
        if chunk:
            buf.write(chunk)
    return buf.getvalue()


def _is_voice_stem(name: str) -> bool:
    """True when a zip entry's filename looks like the voice stem."""
    stem = os.path.splitext(os.path.basename(name))[0].lower()
    return any(tag in stem for tag in ("vocal", "voice", "lead"))


def isolate_voice_elevenlabs(audio_path: str, out_dir: str) -> tuple[str, str]:
    """Return ``(vocals_wav, accompaniment_wav)`` via ElevenLabs stem separation.

    Both files are 48 kHz mono WAV written into ``out_dir``. Raises if the API
    call fails, the ZIP is malformed, or the expected stems are missing.
    """
    os.makedirs(out_dir, exist_ok=True)
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)

    last_err: Exception | None = None
    zip_bytes: bytes | None = None
    chosen_fmt: str | None = None
    for fmt in _OUTPUT_FORMATS:
        try:
            with open(audio_path, "rb") as f:
                stream = client.music.separate_stems(file=f, output_format=fmt)
                data = _collect_zip_bytes(stream)
            if len(data) < 1024:
                raise RuntimeError(f"empty response (got {len(data)} bytes)")
            zip_bytes = data
            chosen_fmt = fmt
            print(f"[audio_isolation_eleven] stems returned in format={fmt}")
            break
        except Exception as e:
            last_err = e
            print(
                f"[audio_isolation_eleven] format={fmt} failed ({e!r}); "
                f"trying next"
            )

    if zip_bytes is None or chosen_fmt is None:
        raise RuntimeError(
            f"ElevenLabs stem separation failed for every output format; "
            f"last error: {last_err!r}"
        )

    raw_decode_flags = _raw_pcm_flags(chosen_fmt) or []

    stems_raw_dir = os.path.join(out_dir, "_eleven_stems")
    os.makedirs(stems_raw_dir, exist_ok=True)
    voice_src: str | None = None
    other_srcs: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            dest = os.path.join(stems_raw_dir, os.path.basename(info.filename))
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            if _is_voice_stem(info.filename):
                voice_src = dest
            else:
                other_srcs.append(dest)

    if voice_src is None:
        raise RuntimeError(
            f"ElevenLabs zip did not contain a vocals stem; got files: "
            f"{os.listdir(stems_raw_dir)}"
        )
    if not other_srcs:
        raise RuntimeError(
            "ElevenLabs zip only contained the vocals stem — nothing to "
            "mix into an accompaniment"
        )

    # Re-encode the vocals stem to canonical 48 kHz mono WAV.
    vocals_wav = os.path.join(out_dir, "vocals.wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            *raw_decode_flags, "-i", voice_src,
            "-ac", "1", "-ar", "48000", "-acodec", "pcm_s16le",
            vocals_wav,
        ],
        check=True,
    )

    # Accompaniment = every non-vocal stem mixed together.
    # amix with normalize=0 preserves absolute levels so the music bed
    # comes out at its natural loudness (roughly original - vocals).
    accomp_wav = os.path.join(out_dir, "accompaniment.wav")
    ff_cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for src in other_srcs:
        ff_cmd += [*raw_decode_flags, "-i", src]
    if len(other_srcs) == 1:
        ff_cmd += [
            "-ac", "1", "-ar", "48000", "-acodec", "pcm_s16le",
            accomp_wav,
        ]
    else:
        n = len(other_srcs)
        filter_inputs = "".join(f"[{i}:a]" for i in range(n))
        ff_cmd += [
            "-filter_complex",
            (
                f"{filter_inputs}amix=inputs={n}:duration=longest:"
                f"normalize=0[mix];"
                f"[mix]aformat=channel_layouts=mono,aresample=48000[out]"
            ),
            "-map", "[out]",
            "-acodec", "pcm_s16le",
            accomp_wav,
        ]
    subprocess.run(ff_cmd, check=True)

    # Tidy up raw stems — we only hand back the two canonicalised WAVs.
    for src in [voice_src, *other_srcs]:
        try:
            os.remove(src)
        except OSError:
            pass
    try:
        os.rmdir(stems_raw_dir)
    except OSError:
        pass

    return vocals_wav, accomp_wav
