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

# Load .env if present (strip Windows CRs so values stay clean). EXTRA_ARGS
# may be set there either as a plain string or as a bash array.
if [ -f .env ]; then
    source <(tr -d '\r' < .env) 2>/dev/null || true
fi

# Return $1's value if it is a valid number, else the default (with a warning).
# usage: safe_num NAME DEFAULT [int|float]
safe_num() {
    local name="$1" default="$2" kind="${3:-int}" val re
    val="${!name:-}"
    val="${val//$'\r'/}"
    if [ "$kind" = "int" ]; then re='^-?[0-9]+$'; else re='^[0-9]+([.][0-9]+)?$'; fi
    if [[ "$val" =~ $re ]]; then
        echo "$val"
    else
        if [ -n "$val" ]; then
            echo "WARNING: ignoring invalid $name='$val' (using $default)" >&2
        fi
        echo "$default"
    fi
}

export VOXCPM_BRIDGE_PORT="$(safe_num VOXCPM_BRIDGE_PORT 10212 int)"
export PROMPT_WAV="${PROMPT_WAV:-$DIR/voices/nicoleheart.wav}"
export PROMPT_TEXT="${PROMPT_TEXT:-}"
export VOICE_SEED="$(safe_num VOICE_SEED 42 int)"
export VOICE_NAME="${VOICE_NAME:-nicoleheart}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

# Needed by the pre-download step below as well as the launch command.
MODEL_ID="${MODEL_ID:-openbmb/VoxCPM2}"

# Normalise EXTRA_ARGS to an array whatever .env made of it. Appending to a
# plain string with += would otherwise glue it into one broken argument.
if declare -p EXTRA_ARGS 2>/dev/null | grep -q '^declare -[a-zA-Z-]*a'; then
    :   # already an array
elif [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS"
else
    EXTRA_ARGS=()
fi

# CLONING_MODE: "ultimate" (needs PROMPT_TEXT, best timbre, no style/tags)
#               "controllable" (reference only, style/tags work)
# Default: auto (ultimate if PROMPT_TEXT set, else controllable)
export CLONING_MODE="${CLONING_MODE:-}"

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
# Dependencies (idempotent; uv skips what's satisfied). The stamp records
# the settings, so changing the torch index or the voxcpm commit re-runs it.
# ---------------------------------------------------------------------------

# IMPORTANT: cu132 wheels have no matching torchaudio build; use cu130
# (works with 13.x drivers).
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
DEPS_KEY="v2 | $TORCH_INDEX | ${VOXCPM_COMMIT_HASH:-main}"

if [ "$(cat .deps-stamp 2>/dev/null)" != "$DEPS_KEY" ]; then
    echo ">>> Installing/updating dependencies..."
    # PyTorch with CUDA — install first so voxcpm doesn't drag in a CPU
    # build via an unpinned dependency.
    uv pip install torch --index-url "$TORCH_INDEX"

    # CRITICAL: Install from GitHub main, NOT pypi.
    # The seed parameter (required for voice consistency) was merged
    # June 2026 (PR #327, 30118a33...), after the last pypi release.
    # Pin VOXCPM_COMMIT_HASH to a FULL commit hash for reproducible installs.
    install_voxcpm() {
        if [ -n "${VOXCPM_COMMIT_HASH:-}" ]; then
            echo ">>> Installing voxcpm from git @ ${VOXCPM_COMMIT_HASH:0:8}..."
            uv pip install --upgrade "voxcpm @ git+https://github.com/OpenBMB/VoxCPM.git@$VOXCPM_COMMIT_HASH"
        else
            echo ">>> WARNING: VOXCPM_COMMIT_HASH not set; installing from main HEAD."
            echo "         Set VOXCPM_COMMIT_HASH in .env to pin a specific commit."
            uv pip install --upgrade "voxcpm @ git+https://github.com/OpenBMB/VoxCPM.git"
        fi
    }

    DEPS_OK=1
    if ! install_voxcpm; then
        # A bad pin must not crash-loop the service: keep what is installed.
        if python -c "import voxcpm" 2>/dev/null; then
            echo ">>> WARNING: voxcpm install failed (is VOXCPM_COMMIT_HASH a valid" >&2
            echo "    commit?). Continuing with the version already installed." >&2
            DEPS_OK=0
        else
            echo "ERROR: voxcpm install failed and none is installed." >&2
            exit 1
        fi
    fi

    # Bridge dependencies (voxcpm does not pull these in)
    uv pip install wyoming soundfile numpy scipy

    # Only record success, so a failed install is retried on the next start.
    if [ "$DEPS_OK" = "1" ]; then
        echo "$DEPS_KEY" > .deps-stamp
    fi
else
    echo ">>> Dependencies already installed (rm .deps-stamp to force)"
fi

# ---------------------------------------------------------------------------
# Optional: pre-download model weights (first synthesis would otherwise
# pay the download cost). The stamp records the model id.
# ---------------------------------------------------------------------------

if [ "${PREDOWNLOAD_MODEL:-1}" = "1" ] && [ "$(cat .model-stamp 2>/dev/null)" != "$MODEL_ID" ]; then
    echo ">>> Pre-downloading model weights ($MODEL_ID)..."
    MODEL_ID="$MODEL_ID" python - <<'PY'
import logging, os
logging.basicConfig(level=logging.INFO)
from huggingface_hub import snapshot_download
snapshot_download(os.environ["MODEL_ID"],
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("model weights cached")
PY
    echo "$MODEL_ID" > .model-stamp
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
    echo "WARNING: VOICE_SEED is negative — each block gets a random voice" >&2
    echo "         bias, causing inconsistent timbre between sentences." >&2
    echo "         Set VOICE_SEED (e.g. 42) in .env for a pinned voice." >&2
fi

# Contradictory combo check (cosmetic — the bridge downgrades anyway)
if [ "${CLONING_MODE:-}" = "ultimate" ] && [ -z "$PROMPT_TEXT" ]; then
    echo "WARNING: CLONING_MODE=ultimate needs PROMPT_TEXT; the bridge will" >&2
    echo "         fall back to controllable mode." >&2
fi

# Only pass the flag when set, so unset keeps auto-resolution
if [ -n "${CLONING_MODE:-}" ]; then
    EXTRA_ARGS+=("--cloning-mode" "$CLONING_MODE")
fi

ARGS=(
    --uri "tcp://0.0.0.0:${VOXCPM_BRIDGE_PORT}"
    --model-id "$MODEL_ID"
    --prompt-wav "$PROMPT_WAV"
    --prompt-text "$PROMPT_TEXT"
    --seed "$VOICE_SEED"
    --voice "$VOICE_NAME"
)

# MAX_STYLE_TAGS: cap on style tags combined per block (0 = no limit).
MAX_STYLE_TAGS="$(safe_num MAX_STYLE_TAGS 0 int)"
if [ "$MAX_STYLE_TAGS" -gt 0 ]; then
    ARGS+=(--max-style-tags "$MAX_STYLE_TAGS")
fi

# VOICE_DESCRIPTOR: identity clause put in front of every style instruction.
# Set it (even to empty) to override the bridge default; empty = delivery only.
if [ "${VOICE_DESCRIPTOR+set}" = "set" ]; then
    ARGS+=(--voice-descriptor "$VOICE_DESCRIPTOR")
fi

echo ">>> Starting Wyoming bridge on port $VOXCPM_BRIDGE_PORT"
echo "    voice:   $VOICE_NAME ($PROMPT_WAV)"
echo "    clone:   ${CLONING_MODE:-auto} (ultimate needs PROMPT_TEXT)"
echo "    seed:    $VOICE_SEED"
echo "    GPU:     CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "    deps:    voxcpm @ ${VOXCPM_COMMIT_HASH:-git/main HEAD}"

# Show the exact command so argument problems are visible
printf '>>> cmd: python %q' "$BRIDGE_SCRIPT"
printf ' %q' "${ARGS[@]}" "${EXTRA_ARGS[@]}"
echo

exec python "$BRIDGE_SCRIPT" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
