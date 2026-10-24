#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# VoxCPM2 Wyoming TTS bridge — setup & run
#
# Unlike the CosyVoice stack, VoxCPM2 is pip-installable: no model repo
# clone, no requirements gymnastics. Needs:
#   Python >= 3.10, < 3.13   |   PyTorch >= 2.5   |   CUDA >= 12.0
# Model weights (~few GB) auto-download from HuggingFace on first run,
# or below via the optional pre-download step.
# ---------------------------------------------------------------------------

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Load .env if present (allows overriding the vars below)
source .env 2>/dev/null || true

export VOXCPM_BRIDGE_PORT="${VOXCPM_BRIDGE_PORT:-10212}"
export PROMPT_WAV="${PROMPT_WAV:-$DIR/voices/nicoleheart.wav}"
export VOICE_NAME="${VOICE_NAME:-nicoleheart}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BRIDGE_SCRIPT="${BRIDGE_SCRIPT:-$DIR/bridge_voxcpm2.py}"

if [ ! -f "$BRIDGE_SCRIPT" ]; then
    echo "ERROR: bridge script not found: $BRIDGE_SCRIPT" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# uv bootstrap + venv
# ---------------------------------------------------------------------------

if ! command -v uv &> /dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
    echo ">>> Creating venv (Python 3.11)..."
    # voxcpm requires >= 3.10, < 3.13 — 3.11 is the sweet spot.
    # uv downloads the interpreter if the system doesn't have it.
    uv venv -p 3.11
fi

source .venv/bin/activate

# ---------------------------------------------------------------------------
# Dependencies (idempotent; uv skips what's satisfied)
# ---------------------------------------------------------------------------

echo ">>> Installing/updating dependencies..."
if [ ! -f ".deps-stamp" ]; then
    # PyTorch with CUDA — install first so voxcpm doesn't drag in a CPU
    # build via an unpinned dependency. Adjust the cuXXX suffix to your
    # driver (cu121/cu124/cu126); the default matches most current setups.
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu132}"
    uv pip install torch --index-url "$TORCH_INDEX"

    uv pip install voxcpm wyoming soundfile

    touch .deps-stamp
else
    echo ">>> Dependencies already installed (rm .deps-stamp to force)"
fi

# ---------------------------------------------------------------------------
# Optional: pre-download model weights (first synthesis would otherwise
# pay the download cost). Skipped if HF cache already has them.
# ---------------------------------------------------------------------------

if [ "${PREDOWNLOAD_MODEL:-1}" = "1" ] && [ ! -f ".model-stamp" ]; then
    echo ">>> Pre-downloading model weights..."
    python - <<'EOF'
import logging
logging.basicConfig(level=logging.INFO)
from huggingface_hub import snapshot_download
snapshot_download("openbmb/VoxCPM2",
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("model weights cached")
EOF
    touch .model-stamp
else
    echo ">>> Skipping model pre-download (already cached or disabled)"
fi

# ---------------------------------------------------------------------------
# Sanity checks before launching
# ---------------------------------------------------------------------------

if [ ! -f "$PROMPT_WAV" ]; then
    echo "WARNING: prompt WAV not found at $PROMPT_WAV" >&2
    echo "         cloning will fall back to default voice unless --prompt-wav is fixed" >&2
fi

echo ">>> Starting Wyoming bridge on port $VOXCPM_BRIDGE_PORT"
echo "    voice:   $VOICE_NAME ($PROMPT_WAV)"
echo "    GPU:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec python "$BRIDGE_SCRIPT" \
    --uri "tcp://0.0.0.0:${VOXCPM_BRIDGE_PORT}" \
    --model-id "${MODEL_ID:-openbmb/VoxCPM2}" \
    --prompt-wav "$PROMPT_WAV" \
    --voice "$VOICE_NAME" \
    "${EXTRA_ARGS[@]}"
