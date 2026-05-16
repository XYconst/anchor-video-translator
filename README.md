# Anchor Video Translator

Open-source pipeline that takes one MP4 in and returns the same clip
dubbed + captioned into 30+ languages, with the original speaker's voice
cloned for each market. Cuts stay on-tempo: the video deforms to match
the dub instead of the audio being padded with dead air.

Built and used in production by [Anchor Media](https://anchormedia.io)
on every ad we ship.

---

## How it works (one paragraph)

The backend is a FastAPI pipeline. Input video → ffmpeg pulls the audio
→ ElevenLabs isolates the vocals from the music bed → KIE / Gemini-3-Flash
transcribes + speaker-diarizes + translates → ElevenLabs clones every
speaker's voice and re-renders the translated lines → audio is fitted
back to the cut (audio sped up to +20% if too long, video compressed if
too short, never silence-padded) → captions are burned in with libass
using whatever position + style the operator picked → final MP4 dropped
on disk under `uploads/{job_id}/{lang}/final.mp4`.

The frontend is a Next.js studio for picking the source clip, languages,
caption position/style, and watching the per-step progress.

---

## What you need before you start

Two API keys:

1. **ElevenLabs** — `https://elevenlabs.io/app/settings/api-keys`
   - Paid plan strongly recommended. One full job (1 clip × 5 languages)
     uses ~5-15 minutes of dubbing credits + a clone slot per speaker.
2. **KIE.ai** — `https://kie.ai`
   - Sign up, copy the API key from the dashboard.
   - This is a proxy that gives you cheaper access to Gemini-3-Flash than
     calling Google Gemini directly.

System tools:

- Python 3.12+ (3.13 works, the setup script installs `audioop-lts` for it)
- ffmpeg (`brew install ffmpeg` / `apt install ffmpeg`)
- Node 18+ (for the Next.js frontend)

---

## Quick start (manual)

```bash
git clone https://github.com/XYconst/anchor-video-translator.git
cd anchor-video-translator

# 1. Backend
./setup.sh                        # creates backend/venv, installs deps
cp .env.example .env              # then fill in ELEVENLABS_API_KEY + KIE_API_KEY
cd backend
./venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev                       # opens at http://localhost:3000
```

Drop an MP4, pick the languages, click Translate. Rendered cuts appear
under `Delivered` once the pipeline finishes.

---

## Quick start (Claude Code does it for you)

If you have [Claude Code](https://claude.com/claude-code) installed:

```bash
git clone https://github.com/XYconst/anchor-video-translator.git
cd anchor-video-translator
claude
```

Then paste this into Claude:

> Set this project up. Read the README, then walk me through getting an
> ElevenLabs API key and a KIE.ai API key. Once I paste them, write them
> into a `.env` file at the project root, run `./setup.sh`, and start
> both the backend (port 8000) and frontend (port 3000) for me. Open the
> frontend in my browser when ready.

That's the whole onboarding.

---

## Configuration

Every knob lives in `.env`. See `.env.example` for the full list with
comments. The only two required values are `ELEVENLABS_API_KEY` and
`KIE_API_KEY` — everything else has a sane default.

---

## Deploying to your own infrastructure

A working Railway config ships with the repo:

- `Dockerfile` — multi-stage CPU-only Python 3.12, ~3 GB image. Pulls
  torch from the CPU wheel index so the CUDA wheels (4-5 GB each)
  don't bloat the container.
- `railway.toml` — points at the Dockerfile, healthcheck `/api/health`,
  300-second startup window so torch + chromium imports finish before
  Railway kills the container.

Suggested Railway setup:

1. Create a new service from this repo.
2. Set the env vars from `.env.example`.
3. Mount a volume at `/data/uploads` so renders survive deploys.
4. Set `UPLOAD_DIR=/data/uploads` to match.

---

## Architecture notes

- **Caption fit-priority**: when the dub is longer than the original
  cut, audio speeds up to +20% before any video stretching happens.
  When the dub is shorter, the video compresses to fit instead of
  padding silence. Bias is toward never producing dead air.
- **Caption preview is WYSIWYG**: the studio's preview uses container-
  query CSS units (`cqw`) bound to the same `box_width × 0.32` ratio
  the burn-in renderer uses, so placement in the preview matches the
  output exactly.
- **Demucs vocal isolation is best-effort**: if it fails (very short
  clips, malformed audio), the pipeline drops the music bed silently
  and uses voice-only.
- **Serial-queue with worker pool**: one FastAPI process can run
  `MAX_CONCURRENT_JOBS` translations in parallel (default 3). For true
  horizontal scaling beyond ~5 you need Redis-backed job state instead
  of the current in-memory dict — open an issue if you want to PR that.

---

## Why we open-sourced this

Translating one ad into 5 markets used to take a video editor an
afternoon. Now it takes ~6 minutes of compute and zero human time. We
got tired of the workflow being a black box, so here it is — fork it,
ship it, send your videos out faster.

If you ship something cool with it, tell us at hello@anchormedia.io.

---

## License

MIT.
