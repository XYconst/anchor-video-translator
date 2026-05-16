#!/usr/bin/env bash
# One-time setup for the video-translator-vb backend.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/backend"
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
# pydub (transitive via pipeline.py) imports stdlib `audioop`, which was removed
# in Python 3.13. The backport package provides `audioop` and `pyaudioop`.
PY_MINOR="$(./venv/bin/python -c 'import sys; print(sys.version_info[1])')"
if [ "$PY_MINOR" -ge 13 ]; then
  ./venv/bin/pip install audioop-lts
fi
echo "video-translator-vb backend installed. Create apps/video-translator-vb/.env (copy ELEVENLABS_API_KEY and GEMINI_API_KEY from apps/api/.env) before starting."
