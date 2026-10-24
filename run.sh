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
export PROMPT_TEXT="${PROMPT_TEXT:-}"
export VOICE_SEED="${VOICE_SEED:--1}"
export VOICE_NAME="${VOICE_NAME:-nicoleheart}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CLONING_MODE="${CLONING_MODE:-}"

# Initialize before sourcing so .env can extend it, and so the
# unconditional expansion at the bottom is safe under `set -e`.
if ! declare -p EXTRA_ARGS &>/dev/null; then
    EXTRA_ARGS=()
fi

if [ -n "$CLONING_MODE" ]; then
    EXTRA_ARGS+=("--cloning-mode" "$CLONING_MODE")
    echo "    clone:   ${CLONING_MODE:-auto} (ultimate needs PROMPT_TEXT)"
fi

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
    # build via an unpinned dependency. IMPORTANT: cu132 wheels have no
    # matching torchaudio build; use cu130 (works with 13.x drivers).
    TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
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
snapshot_download("${MODEL_ID:-openbmb/VoxCPM2}",
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

if [ -z "$PROMPT_TEXT" ]; then
    echo "WARNING: PROMPT_TEXT is not set — ultimate cloning disabled." >&2
    echo "         Expect audible voice drift between sentence blocks." >&2
    echo "         Set PROMPT_TEXT to the exact transcript of the reference WAV." >&2
fi

if [ "$VOICE_SEED" -lt 0 ]; then
    echo "WARNING: VOICE_SEED not set (< 0) — each block gets a random voice" >&2
    echo "         bias, causing inconsistent timbre between sentences." >&2
    echo "         Set VOICE_SEED (e.g. 42) in .env for a pinned voice." >&2
fi

# NEW: contradictory combo check (cosmetic — the bridge downgrades anyway)
if [ "${CLONING_MODE:-}" = "ultimate" ] && [ -z "$PROMPT_TEXT" ]; then
    echo "WARNING: CLONING_MODE=ultimate needs PROMPT_TEXT; the bridge will" >&2
    echo "         fall back to controllable mode." >&2
fi

# NEW: only pass the flag when set, so unset keeps auto-resolution
if [ -n "${CLONING_MODE:-}" ]; then
    EXTRA_ARGS+=("--cloning-mode" "$CLONING_MODE")
fi

echo ">>> Starting Wyoming bridge on port $VOXCPM_BRIDGE_PORT"
echo "    voice:   $VOICE_NAME ($PROMPT_WAV)"
echo "    clone:   ${CLONING_MODE:-auto} (ultimate needs PROMPT_TEXT)"   # NEW
echo "    seed:    $VOICE_SEED"
echo "    GPU:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec python "$BRIDGE_SCRIPT" \
    --uri "tcp://0.0.0.0:${VOXCPM_BRIDGE_PORT}" \
    --model-id "${MODEL_ID:-openbmb/VoxCPM2}" \
    --prompt-wav "$PROMPT_WAV" \
    --prompt-text "$PROMPT_TEXT" \
    --seed "$VOICE_SEED" \
    --voice "$VOICE_NAME" \
    "${EXTRA_ARGS[@]}"
