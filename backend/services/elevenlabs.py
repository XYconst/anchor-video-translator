from __future__ import annotations
import os
import wave
from typing import List
from elevenlabs import ElevenLabs
from config import settings

client = ElevenLabs(api_key=settings.elevenlabs_api_key)

# Tuned voice settings for consistency across segments
VOICE_SETTINGS = {
    "stability": 0.75,
    "similarity_boost": 0.9,
    "style": 0.0,
    "use_speaker_boost": True,
}


def clone_voice(audio_path: str, name: str) -> str:
    """Clone a voice from audio file. Returns voice_id."""
    with open(audio_path, "rb") as f:
        voice = client.voices.ivc.create(
            name=name,
            files=[f],
            remove_background_noise=True,
            description="Auto-cloned voice for video translation",
        )
    return voice.voice_id


def generate_tts_segment(text: str, voice_id: str, output_path: str) -> str:
    """Generate TTS for a single segment with consistent voice settings."""
    audio_generator = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="pcm_22050",
        voice_settings=VOICE_SETTINGS,
    )

    audio_bytes = b""
    for chunk in audio_generator:
        audio_bytes += chunk

    with wave.open(output_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(22050)
        wav.writeframes(audio_bytes)

    return output_path


def generate_all_tts(segments, voice_id: str, work_dir: str) -> list:
    """Generate TTS for each segment individually. Returns list of audio paths."""
    paths = []
    for i, seg in enumerate(segments):
        out_path = os.path.join(work_dir, f"tts_{i:04d}.wav")
        generate_tts_segment(seg.translated, voice_id, out_path)
        paths.append(out_path)
    return paths


def delete_voice(voice_id: str):
    """Delete a cloned voice to free up quota."""
    try:
        client.voices.delete(voice_id=voice_id)
    except Exception:
        pass
