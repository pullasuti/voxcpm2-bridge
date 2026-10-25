#!/usr/bin/env python3
"""VoxCPM2 Wyoming TTS bridge — direct variant (no HTTP), streaming.

Adapted from the CosyVoice direct bridge. Engine-agnostic machinery
carried over unchanged: sentence merging, prebuffering (jitter buffer),
uniform re-chunking, Error events, graceful shutdown, bind-first /
load-later startup. VoxCPM2-specific behavior:

  * SOFT-VOICE STYLE VOCABULARY — every style instruction carries a
    byte-identical identity clause (--voice-descriptor, default
    "a soft, delicate, gentle voice") followed by a delivery clause.
    Because the identity half of the conditioning never changes,
    cross-block timbre drift shrinks AND the delivery vocabulary is
    deliberately low-arousal (smiling, pleased, tender) instead of
    high-arousal (excited, energetic), which fights the reference
    timbre. Expressive range is preserved; loudness is not.
  * CLONING MODES — upstream, ultimate cloning (reference supplied via
    BOTH pathways: isolated reference + continuation prefix with its
    transcript) and style control are MUTUALLY EXCLUSIVE. In ultimate
    ("Hi-Fi") mode, the control prefix in parentheses is ignored and
    may be READ ALOUD as text, and inline non-verbal tags are
    suppressed along with it. See ronger-x/voxcpm2-service PR #3:
    "When Hi-Fi mode is enabled, the control instruction is ignored."

      --cloning-mode ultimate      best timbre fidelity, NO style/tags
                                   (requires --prompt-text)
      --cloning-mode controllable  style parentheticals + inline tags
                                   work; slightly less faithful clone

    Default: ultimate when --prompt-text is given, controllable
    otherwise. In ultimate mode the bridge SUPPRESSES style/tag
    instructions (with a logged warning) instead of letting the
    model read them aloud.
  * VOICE CONSISTENCY — three measures against timbre drift between
    blocks (each merged block is a separate generation call):
      1. FIXED SEED — requires the git build of voxcpm (pypi 2.0.3
         predates the June 2026 seed merge, PR #327). The bridge
         verifies via signature introspection; falls back to global
         RNG reseeding (random, numpy, torch, CUDA) on older builds.
      2. CONSTANT IDENTITY CLAUSE — the descriptor in every style
         instruction is identical, so the identity half of the
         conditioning is stable across blocks.
      3. ULTIMATE CLONING — dual-pathway reference (highest-SIM
         cloning recipe per the technical report, at the cost of
         style tags).
  * TRUE STREAMING — each model yield is forwarded to the play queue
    the moment it is produced (run_coroutine_threadsafe from the GPU
    thread), so audio flows to HA progressively within a block.
  * STARTUP WARM-UP — a short dummy synthesis runs after model load
    so torch.compile's one-time cost is paid during startup rather
    than on HA's first real request. Non-fatal if it fails (the
    compile cost then lands on the first real request instead).
  * STYLE CONTROL (controllable mode only) — parenthetical
    instruction prefix on the text; sentence tags from the LLM map
    into this prefix. Inline tags likewise controllable-mode only.
  * INLINE NON-VERBAL TAGS (controllable mode only) — VoxCPM2's own
    vocabulary, inserted in square brackets per the usage docs:
        [laughing] [sigh] [Uhm] [Shh]
        [Question-ah/-ei/-en/-oh] [Surprise-wa/-yo] [Dissatisfaction-hnn]
    (exact case matters; there is no [cough]). Other tags are aliased where
    an equivalent exists;
    tags with no equivalent are dropped with a warning rather than
    fed to the model.
  * 48 kHz OUTPUT, 16 kHz REFERENCE INPUT — reference resampled once
    at startup (polyphase filter if scipy is available, linear
    interpolation otherwise).

Known upstream limitations (worth remembering): the docs themselves
note that cloning/style results vary between runs; a fixed seed and
constant identity clause mitigate but do not fully eliminate drift.
For production-grade consistency upstream recommends LoRA fine-tuning
on 5-10 minutes of clean audio.

Must run inside a venv with `pip install voxcpm wyoming soundfile`
(Python >= 3.10, < 3.13; PyTorch >= 2.5; voxcpm from git for seed).
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import random
import re
import signal
import sys
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

_LOGGER = logging.getLogger(__name__)

# Rough speech-rate estimate for merging decisions.
WORDS_PER_SECOND = 2.8

# A one-shot synthesize arriving this soon after synthesize-stop, with
# matching text, is a late duplicate of the stream just played, not a request.
LATE_DUPLICATE_WINDOW_S = 3.0

# ---------------------------------------------------------------------------
# Soft-voice style vocabulary
#
# Every instruction = BASE_VOICE (identity, never varies) + delivery
# clause (varies by tag). Two reasons:
#   1. Delivery vocabulary is deliberately LOW-AROUSAL: high-arousal
#      words (excited, energetic, loud) fight a soft, delicate
#      reference and cause timbre drift; low-arousal words (smiling,
#      pleased, tender) preserve the voice while coloring emotion.
#   2. The identity clause is byte-identical across all tags, so the
#      identity half of the conditioning is stable across blocks —
#      less re-negotiation of "who is speaking" per block.
#
# Structurally incompatible tags (loud, angry-at-volume) are remapped
# to low-arousal equivalents; there is no wording that makes a
# delicate voice genuinely loud without abandoning the clone.
# ---------------------------------------------------------------------------

DEFAULT_BASE_VOICE = "a soft, delicate, gentle voice"

# Static, reference-ready compositions using the default descriptor.
# The engine rebuilds this table at load time when --voice-descriptor
# overrides the default, so .env tuning needs no code changes.
_DELIVERY_CLAUSES = {
    "warm":      "speaking warmly and kindly",
    "sad":       "speaking sadly and wistfully",
    "serious":   "speaking calmly and evenly",
    "confident": "speaking quietly assured",
    "happy":     "smiling warmly as she speaks",
    "excited":   "pleased and delighted, lightly animated",
    "playful":   "speaking playfully and amused",
    "tender":    "speaking tenderly and affectionately",
    "whisper":   "whispering barely above breath",
    "thoughtful": "speaking thoughtfully and unhurried",
    "slow":      "speaking slowly and unhurried",
    "fast":      "speaking a little quicker, still relaxed",
}

# Legacy high-arousal tags from the LLM remapped to low-arousal
# delivery clauses so old prompts keep working (and stop drifting).
_TAG_FALLBACKS = {
    "loud": "raising her voice slightly, still gentle",
    "angry": "quietly annoyed and firm",
}

STYLE_TAG_ALIASES = {
    "friendly": "warm",
    "gentle": "tender",
    "cheerful": "happy",
    "calm": "serious",
    "excitedly": "excited",
    "happily": "happy",
    "sadly": "sad",
    "angrily": "angry",
    "softly": "whisper",
    "quieter": "whisper",
}

STYLE_ORDER = [
    "warm", "happy", "sad", "serious", "confident", "excited",
    "playful", "tender", "whisper", "thoughtful", "slow", "fast",
]


def build_style_tags(base_voice: str) -> dict[str, str]:
    """Compose the full tag table: identity clause + delivery clause."""
    tags = {
        key: f"{base_voice}, {clause}"
        for key, clause in _DELIVERY_CLAUSES.items()
    }
    for key, clause in _TAG_FALLBACKS.items():
        tags[key] = f"{base_voice}, {clause}"
    return tags


# ---------------------------------------------------------------------------
# Inline tag handling (controllable mode only)
# ---------------------------------------------------------------------------

# VoxCPM2's documented non-verbal tags, in the EXACT case the model expects
# (cookbook: voxcpm.readthedocs.io). Keys are lowercase lookups; values are
# what is sent to the model. There is no [cough] (nor a breath tag): those
# are dropped rather than sent, because unknown tags are spoken as words.
CANON_INLINE = {
    "laughing": "[laughing]",
    "sigh": "[sigh]",
    "uhm": "[Uhm]",
    "shh": "[Shh]",
    "question-ah": "[Question-ah]",
    "question-ei": "[Question-ei]",
    "question-en": "[Question-en]",
    "question-oh": "[Question-oh]",
    "surprise-wa": "[Surprise-wa]",
    "surprise-yo": "[Surprise-yo]",
    "dissatisfaction-hnn": "[Dissatisfaction-hnn]",
}

TAG_ALIASES = {
    "laughter": "laughing",
    "laughs": "laughing",
    "laugh": "laughing",
    "sighs": "sigh",
    "um": "uhm",
    "hm": "uhm",
    "shush": "shh",
    "hush": "shh",
}

# Known to have no VoxCPM2 equivalent: stripped quietly (one warning each).
UNSUPPORTED_INLINE = {"cough", "coughs", "breath", "breaths", "breathing"}
_warned_unsupported: set[str] = set()

TAG_RE = re.compile(r"\s*[\[\{]([^\]\}]+)[\]\}]")

_TERM_RE = re.compile(r"[.!?…][\"'”’)\]]*\s+")
_WORD_RE = re.compile(r"[A-Za-z.]+$")

_ABBREVIATIONS = {
    "e.g", "i.e", "eg", "ie", "etc", "mr", "mrs", "ms", "dr", "prof",
    "sr", "jr", "vs", "st",
}


def parse_tags(
    text: str,
    style_tags: dict[str, str],
    allow_inline: bool = True,
) -> tuple[str, Optional[str]]:
    """Split text into (spoken text, style instruction or None).

    Square-bracket tags whose canonical form is a style keyword become
    the (sticky) style instruction; those with an inline equivalent are
    passed through in VoxCPM2's vocabulary; anything else is dropped
    with a warning.

    When allow_inline is False (ultimate cloning mode, where upstream
    suppresses tags), inline tags are stripped from the text entirely.
    """
    tags: list[str] = []
    out: list[str] = []
    pos = 0

    for m in TAG_RE.finditer(text):
        tag = m.group(1).strip()
        if m.start() > pos:
            out.append(text[pos:m.start()])

        key = tag.lower()
        folded_style = STYLE_TAG_ALIASES.get(key, key)
        folded_inline = TAG_ALIASES.get(key, key)

        if folded_inline in CANON_INLINE:
            if allow_inline:
                # keep a space before the tag: the regex ate the one in front
                out.append(f" {CANON_INLINE[folded_inline]} ")
            else:
                _LOGGER.info("stripping inline tag %r (ultimate mode)", tag)
        elif key in UNSUPPORTED_INLINE:
            if key not in _warned_unsupported:
                _warned_unsupported.add(key)
                _LOGGER.warning(
                    "tag %r has no VoxCPM2 equivalent; removing it", tag)
        elif folded_style in style_tags:
            tags.append(style_tags[folded_style])
        else:
            _LOGGER.warning("dropping tag %r (no equivalent)", tag)

        pos = m.end()

    if pos < len(text):
        out.append(text[pos:])

    clean = " ".join("".join(out).split())
    clean = re.sub(r"\s+([,.!?;:])", r"\1", clean)  # "[sigh] ," -> "[sigh],"
    return clean, ("; ".join(tags) if tags else None)


def speed_hint(speed: float) -> Optional[str]:
    if speed == 1.0:
        return None
    if speed < 0.7:
        return "speaking as slowly as possible"
    if speed < 1.0:
        return "speaking a bit slower than normal"
    if speed > 1.5:
        return "speaking as fast as possible"
    return "speaking a bit faster than normal"


_NORM_TAG_RE = re.compile(r"[\[\{][^\]\}]*[\]\}]")


def norm_text(text: str) -> str:
    """Comparison key: tags removed, lowercase, punctuation/space-insensitive."""
    text = _NORM_TAG_RE.sub(" ", text)
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def estimate_seconds(text: str) -> float:
    return len(text.split()) / WORDS_PER_SECOND


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def to_pcm16(data: Any) -> bytes:
    """numpy array or torch tensor, float [-1, 1] -> PCM16 LE bytes."""
    if not isinstance(data, np.ndarray):
        data = data.detach().cpu().numpy()
    arr = np.asarray(data, dtype=np.float64).squeeze()
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def _resample(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resampling when scipy is available (best quality),
    linear-interpolation fallback otherwise."""
    try:
        from scipy.signal import resample_poly  # noqa: PLC0415
        from math import gcd
        g = gcd(sr, target_sr)
        return resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    except ImportError:
        duration = len(data) / sr
        idx = np.linspace(0.0, duration, int(duration * target_sr), endpoint=False)
        src_idx = np.arange(len(data)) / sr
        return np.interp(idx, src_idx, data).astype(np.float32)


def ensure_reference_16k(wav_path: str) -> str:
    """VoxCPM2 expects 16 kHz reference audio. Resample once if needed."""
    try:
        import soundfile as sf  # noqa: PLC0415
    except ImportError:
        _LOGGER.warning(
            "soundfile not installed; passing %s through unresampled — "
            "verify the reference is 16 kHz", wav_path,
        )
        return wav_path

    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if sr == 16000:
        return wav_path

    _LOGGER.info("resampling reference %s: %d Hz -> 16000 Hz", wav_path, sr)
    if data.ndim > 1:
        data = data.mean(axis=1)

    resampled = _resample(data, sr, 16000)

    out = Path(tempfile.gettempdir()) / (
        Path(wav_path).stem + "_16k_reference.wav"
    )
    sf.write(str(out), resampled, 16000)
    return str(out)


# ---------------------------------------------------------------------------
# Model wrapper — the ONLY place that talks to VoxCPM2
# ---------------------------------------------------------------------------

class SynthEngine:
    def __init__(
        self,
        model_id: str,
        prompt_wav: str,
        prompt_text: str,
        cloning_mode: str,
        voice_descriptor: str,
        default_style: Optional[str],
        speed: float,
        cfg_value: float,
        inference_timesteps: int,
        seed: int,
        constant_identity: bool = True,
    ) -> None:
        self.model_id = model_id
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.cloning_mode = cloning_mode  # "ultimate" | "controllable"
        self.default_style = default_style
        self.speed = speed
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.seed = seed

        self.voice_descriptor = voice_descriptor
        self.constant_identity = constant_identity

        # Soft-voice style table, composed from the descriptor.
        self.style_tags = build_style_tags(voice_descriptor)

        self.model: Any = None
        self.reference_wav: Optional[str] = None
        self.sample_rate: int = 48000
        self._lock = threading.Lock()

        # Effective mode after resolution (may downgrade to
        # controllable if prompt_text is missing).
        self._effective_mode = cloning_mode
        self._warned_style_suppressed = False

        # Populated by _introspect_signature() at load time.
        self._accepted: Optional[set[str]] = None
        self._uses_seed_param = False

    # -- capability discovery ------------------------------------------------

    def _introspect_signature(self) -> None:
        """Work out which kwargs the installed voxcpm build accepts.

        Seed support requires the git build (pypi 2.0.3 predates the
        June 2026 merge); introspection lets the bridge adapt to
        whichever is installed.
        """
        fn = getattr(self.model, "_generate", None)
        if fn is None:
            fn = getattr(self.model, "generate_streaming", None)
        if fn is None:
            return

        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return

        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            self._accepted = None
            self._uses_seed_param = True
            _LOGGER.info("voxcpm _generate accepts **kwargs (no filtering)")
            return

        self._accepted = {
            name
            for name, p in params.items()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        self._uses_seed_param = "seed" in self._accepted
        _LOGGER.info(
            "voxcpm _generate accepts: %s",
            ", ".join(sorted(self._accepted)) or "(nothing)",
        )
        if not self._uses_seed_param:
            _LOGGER.warning(
                "installed voxcpm has no `seed` parameter — falling back to "
                "global RNG seeding for voice consistency. Install the git "
                "build for true seed pinning (pypi 2.0.3 predates it)."
            )

    # -- seeding -------------------------------------------------------------

    def _seed_rngs(self) -> None:
        """Fallback: reseed the global RNGs before each block. Used only
        when the model exposes no `seed` parameter of its own."""
        if self.seed < 0:
            return
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch  # noqa: PLC0415
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except Exception:  # pragma: no cover - torch is always present
            _LOGGER.warning("torch RNG seeding failed", exc_info=True)

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        from voxcpm import VoxCPM  # noqa: PLC0415

        _LOGGER.info("loading %s", self.model_id)
        t0 = time.monotonic()

        self.model = VoxCPM.from_pretrained(self.model_id, load_denoiser=False)
        self.sample_rate = int(self.model.tts_model.sample_rate)
        self.reference_wav = ensure_reference_16k(self.prompt_wav)
        self._introspect_signature()

        if self._effective_mode == "ultimate":
            if self.prompt_text:
                _LOGGER.info(
                    "cloning mode: ULTIMATE (dual-pathway reference). "
                    "NOTE: style instructions and inline tags are "
                    "SUPPRESSED in this mode — upstream ignores the "
                    "control prefix in Hi-Fi/transcript cloning and may "
                    "read it aloud; we drop it instead."
                )
            else:
                _LOGGER.warning(
                    "--cloning-mode ultimate requires --prompt-text; "
                    "falling back to controllable"
                )
                self._effective_mode = "controllable"
        else:
            _LOGGER.info(
                "cloning mode: CONTROLLABLE (reference only). Style "
                "parentheticals and inline tags are active; timbre "
                "fidelity is slightly lower than ultimate mode."
            )

        _LOGGER.info(
            "style vocabulary: identity clause %r shared by %d tags "
            "(low-arousal delivery)",
            self.style_tags["warm"].split(",")[0], len(self.style_tags),
        )

        _LOGGER.info(
            "model ready in %.1fs (sample_rate=%d, reference=%s)",
            time.monotonic() - t0, self.sample_rate, self.reference_wav,
        )

    def warm_up(self) -> None:
        """Run one tiny synthesis so torch.compile pays its one-time
        cost here, during startup, instead of on the first request."""
        _LOGGER.info("warm-up synthesis (compiling kernels)...")
        t0 = time.monotonic()
        try:
            for out in self._generate(
                "Warm up.", "speaking with a neutral tone"
            ):
                _ = to_pcm16(out)
        except Exception:
            _LOGGER.warning(
                "warm-up synthesis failed — torch.compile cost will land "
                "on the first real request instead",
                exc_info=True,
            )
            return
        _LOGGER.info("warm-up complete in %.1fs", time.monotonic() - t0)

    # -- generation ----------------------------------------------------------

    def _styled_text(self, text: str, style: Optional[str]) -> str:
        """Apply VoxCPM2 style control: parenthetical prefix on text."""
        parts = []
        if style:
            parts.append(style)
        if self.default_style:
            parts.append(self.default_style)
        if self.speed != 1.0:
            parts.append(speed_hint(self.speed))
        if parts:
            return "(" + "; ".join(parts) + ") " + text
        return text

    @property
    def allow_inline_tags(self) -> bool:
        return self._effective_mode == "controllable"

    def _generate(self, text: str, style: Optional[str]) -> Iterator[Any]:
        ultimate = self._effective_mode == "ultimate"

        if style and ultimate:
            if not self._warned_style_suppressed:
                _LOGGER.warning(
                    "style instruction %r suppressed: ultimate cloning "
                    "mode ignores control prefixes upstream (they would "
                    "be read aloud). Switch to --cloning-mode "
                    "controllable if you need style/emotion tags.",
                    style[:60],
                )
                self._warned_style_suppressed = True
            style = None
        elif style and self.default_style and self.default_style not in style:
            style = f"{style}; {self.default_style}"

        # VOICE CONSISTENCY: a block with a style prefix and a block with
        # none are conditioned differently, which can sound like two
        # different speakers within one reply. Give every block the same
        # identity clause (controllable mode; ultimate mode sends no prefix).
        if style is None and not ultimate and self.constant_identity:
            style = self.voice_descriptor

        text_out = text if (ultimate or not style) else self._styled_text(text, style)

        kwargs: dict[str, Any] = {
            "text": text_out,
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.inference_timesteps,
        }

        if self.reference_wav:
            kwargs["reference_wav_path"] = self.reference_wav
            if ultimate:
                kwargs["prompt_wav_path"] = self.reference_wav
                kwargs["prompt_text"] = self.prompt_text

        if self.seed >= 0:
            if self._uses_seed_param:
                kwargs["seed"] = self.seed
            else:
                self._seed_rngs()

        if self._accepted is not None:
            for key in [k for k in kwargs if k not in self._accepted]:
                _LOGGER.warning(
                    "installed voxcpm does not accept %r — dropping "
                    "(feature degraded)", key,
                )
                kwargs.pop(key)

        return self.model.generate_streaming(**kwargs)

    def synthesize_streaming(
        self, text: str, style: Optional[str], on_chunk: Callable[[bytes], None]
    ) -> None:
        """Blocking (call from a worker thread). Invokes on_chunk(pcm)
        for every model yield the moment it is produced, so audio
        flows downstream progressively."""
        with self._lock:
            t0 = time.monotonic()
            n = 0
            total = 0
            for out in self._generate(text, style):
                pcm = to_pcm16(out)
                if not pcm:
                    continue
                n += 1
                total += len(pcm)
                if n == 1:
                    _LOGGER.info(
                        "[timing] first chunk: req +%.3fs", time.monotonic() - t0
                    )
                on_chunk(pcm)
            dur = total / (2 * self.sample_rate)
            if dur > 0:
                _LOGGER.info(
                    "[timing] block done: %d chunks, %.2fs audio in %.2fs (rtf %.2f)",
                    n, dur, time.monotonic() - t0,
                    (time.monotonic() - t0) / dur,
                )


# ---------------------------------------------------------------------------
# Wyoming handler
# ---------------------------------------------------------------------------

class BridgeHandler(AsyncEventHandler):
    def __init__(
        self,
        *args: Any,
        engine: SynthEngine,
        info: Info,
        voice: str,
        sample_rate: int,
        ready: asyncio.Event,
        min_merged_seconds: float,
        prebuffer_seconds: float,
        samples_per_chunk: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.info = info
        self.voice = voice
        self.sample_rate = sample_rate
        self._ready = ready
        self._min_merged = min_merged_seconds
        self._prebuffer = prebuffer_seconds
        self._samples_per_chunk = samples_per_chunk

        self._stash = ""
        self._buf = ""
        self._got_chunks = False
        self._audio_started = False
        self._instr: Optional[str] = None

        self._pending_sentences: list[str] = []
        self._first_queued = False

        # Stream bookkeeping (plain one-shot requests / late duplicates)
        self._streaming = False
        self._stream_text = ""
        self._last_stream_norm = ""
        self._last_stream_end = 0.0

        self._work_q: Optional[asyncio.Queue] = None
        self._play_q: Optional[asyncio.Queue] = None
        self._fetch_task: Optional[asyncio.Task] = None
        self._play_task: Optional[asyncio.Task] = None

    # -- sentence splitter ---------------------------------------------------

    def _take_sentences(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        pos = 0
        while (m := _TERM_RE.search(self._buf, pos)) is not None:
            word = _WORD_RE.search(self._buf[: m.start()])
            if word and word.group(0).lower() in _ABBREVIATIONS:
                pos = m.start() + 1
                continue
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end():]
            pos = 0
            if sentence:
                out.append(sentence)
        return out

    # -- sentence merging ------------------------------------------------------

    def _feed_sentence(self, sentence: str) -> None:
        self._pending_sentences.append(sentence)
        if not self._first_queued:
            # LATENCY-FIRST: the opening sentence goes out on its own instead
            # of waiting for ~4 s of text (which, for a short reply, meant
            # waiting for the whole LLM answer).
            self._first_queued = True
            self._flush_pending()
            return
        est = sum(estimate_seconds(s) for s in self._pending_sentences)
        if est >= self._min_merged or len(self._pending_sentences) >= 6:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if self._pending_sentences:
            merged = " ".join(self._pending_sentences)
            self._pending_sentences = []
            self._work_q.put_nowait(merged)

    # -- pipeline lifecycle ------------------------------------------------------

    async def _start_pipeline(self) -> None:
        self._work_q = asyncio.Queue()
        self._play_q = asyncio.Queue()
        self._fetch_task = asyncio.create_task(self._fetch_worker())
        self._play_task = asyncio.create_task(self._play_worker())

    async def _stop_pipeline(self) -> None:
        if self._fetch_task is None:
            return
        self._flush_pending()
        await self._work_q.put(None)
        await self._fetch_task
        await self._play_q.put(None)
        await self._play_task
        self._fetch_task = None
        self._play_task = None
        self._work_q = None
        self._play_q = None

    # -- workers -------------------------------------------------------------------

    async def _fetch_worker(self) -> None:
        await self._ready.wait()

        loop = asyncio.get_running_loop()

        async def _put(item: tuple[bytes, int]) -> None:
            await self._play_q.put(item)

        def on_chunk(pcm: bytes) -> None:
            asyncio.run_coroutine_threadsafe(
                _put((pcm, self.engine.sample_rate)), loop
            ).result()

        while (text := await self._work_q.get()) is not None:
            clean, style = parse_tags(
                text,
                style_tags=self.engine.style_tags,
                allow_inline=self.engine.allow_inline_tags,
            )
            if not clean:
                continue

            if style:
                self._instr = style

            _LOGGER.info("synthesizing: text=%r style=%r", clean[:60], self._instr)

            try:
                await asyncio.to_thread(
                    self.engine.synthesize_streaming, clean, self._instr, on_chunk
                )
            except Exception as err:
                _LOGGER.exception("block synthesis failed: %r", clean[:60])
                try:
                    await self.write_event(
                        Error(
                            text=f"synthesis failed: {err}",
                            code=err.__class__.__name__,
                        ).event()
                    )
                except Exception:
                    pass  # connection may already be gone
                continue

    async def _play_worker(self) -> None:
        """Jitter-buffer before AudioStart; write-through after; uniform
        re-chunking into fixed-size AudioChunks with a single remainder
        flush at the end of the utterance."""
        width = 2
        bytes_per_chunk = width * self._samples_per_chunk

        buf = bytearray()
        rate: Optional[int] = None
        started = False
        buffered_bytes = 0

        async def _maybe_emit_start() -> None:
            nonlocal started
            if not self._audio_started and buf:
                await self.write_event(
                    AudioStart(rate=rate, width=width, channels=1).event()
                )
                self._audio_started = True
            started = True

        while (item := await self._play_q.get()) is not None:
            audio, r = item
            if not audio:
                continue

            if rate is None:
                rate = r
            elif r != rate:
                _LOGGER.warning("sample rate changed mid-stream (%s -> %s)", rate, r)

            buf.extend(audio)
            buffered_bytes += len(audio)

            if not started:
                prebuffer_bytes = int(
                    self._prebuffer * 2 * (rate or self.sample_rate)
                )
                if buffered_bytes < prebuffer_bytes:
                    continue
                await _maybe_emit_start()

            while len(buf) >= bytes_per_chunk:
                await self.write_event(
                    AudioChunk(
                        audio=bytes(buf[:bytes_per_chunk]),
                        rate=rate,
                        width=width,
                        channels=1,
                    ).event()
                )
                del buf[:bytes_per_chunk]

        if buf:
            await _maybe_emit_start()
            await self.write_event(
                AudioChunk(audio=bytes(buf), rate=rate, width=width, channels=1).event()
            )
            buf.clear()

    # -- utterance helpers ------------------------------------------------------------

    def _reset_utterance(self) -> None:
        self._stash = ""
        self._buf = ""
        self._got_chunks = False
        self._audio_started = False
        self._instr = None
        self._pending_sentences = []
        self._first_queued = False

    async def _finish_utterance(self) -> None:
        """Flush, wait for all audio to be written, close the audio stream."""
        await self._stop_pipeline()
        if self._audio_started:
            await self.write_event(AudioStop().event())
            self._audio_started = False

    # -- wyoming event handling -----------------------------------------------------

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        if SynthesizeStart.is_type(event.type):
            self._reset_utterance()
            self._streaming = True
            self._stream_text = ""
            await self._start_pipeline()
            return True

        if SynthesizeChunk.is_type(event.type):
            self._got_chunks = True
            chunk_text = SynthesizeChunk.from_event(event).text
            self._stream_text += chunk_text
            for sentence in self._take_sentences(chunk_text):
                self._feed_sentence(sentence)
            return True

        if SynthesizeStop.is_type(event.type):
            if self._got_chunks:
                tail = self._buf.strip()
                self._buf = ""
                if tail:
                    self._feed_sentence(tail)
            else:
                text = self._stash
                self._stash = ""
                self._stream_text = text
                if text.strip():
                    self._feed_sentence(text.strip())

            await self._finish_utterance()

            self._streaming = False
            self._last_stream_norm = norm_text(self._stream_text)
            self._last_stream_end = time.monotonic()
            await self.write_event(SynthesizeStopped().event())
            return True

        if Synthesize.is_type(event.type):
            text = Synthesize.from_event(event).text or ""

            # (1) Compat one-shot that HA sends alongside a stream: ignore
            #     (but keep it as the text for a start/stop with no chunks).
            if self._streaming:
                self._stash = text
                return True

            # (2) Late duplicate right after a finished stream.
            if (self._last_stream_norm
                    and time.monotonic() - self._last_stream_end
                    < LATE_DUPLICATE_WINDOW_S
                    and norm_text(text) == self._last_stream_norm):
                _LOGGER.warning(
                    "late duplicate of finished stream suppressed: %r",
                    text[:60])
                return True

            # (3) Genuine non-streaming request (tts.speak, "Try voice", ...)
            if text.strip():
                self._reset_utterance()
                await self._start_pipeline()
                for sentence in self._take_sentences(text + " "):
                    self._feed_sentence(sentence)
                tail = self._buf.strip()
                self._buf = ""
                if tail:
                    self._feed_sentence(tail)
                await self._finish_utterance()
            return True

        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VoxCPM2 Wyoming bridge (direct)")
    p.add_argument("--uri", default="tcp://0.0.0.0:10211", help="Wyoming server URI")
    p.add_argument(
        "--model-id", default="openbmb/VoxCPM2",
        help="HuggingFace model id (weights auto-download on first run)",
    )
    p.add_argument(
        "--prompt-wav", required=True,
        help="Reference speaker WAV for cloning (resampled to 16 kHz if needed)",
    )
    p.add_argument(
        "--prompt-text", default="",
        help="EXACT transcript of the reference WAV. Enables ultimate "
             "cloning (highest voice similarity) — at the cost of style/"
             "emotion control, which upstream ignores in that mode.",
    )
    p.add_argument(
        "--cloning-mode", choices=("ultimate", "controllable"),
        default=None,
        help="ultimate: dual-pathway reference, best timbre, NO style/"
             "tags (default when --prompt-text is set). controllable: "
             "reference only, style parentheticals + inline tags work, "
             "slightly less faithful clone (default without --prompt-text).",
    )
    p.add_argument(
        "--voice-descriptor", default=DEFAULT_BASE_VOICE,
        help="Identity clause shared by ALL style instructions. Constant "
             "across blocks for timbre stability; tune per reference "
             "voice, e.g. 'a soft, delicate, gentle voice'.",
    )
    p.add_argument(
        "--style",
        default=None,
        help="Baseline style parenthetical applied to every utterance "
             "(controllable mode only)",
    )
    p.add_argument(
        "--no-constant-identity", action="store_true",
        help="Do not add the identity clause to unstyled blocks (default: "
             "every block in controllable mode carries it, so styled and "
             "unstyled sentences stay conditioned alike).",
    )
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = normal")
    p.add_argument("--cfg-value", type=float, default=2.0, help="Classifier-free guidance")
    p.add_argument(
        "--inference-timesteps", type=int, default=10,
        help="Diffusion solver steps (higher = slower, often better)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Fixed generation seed for voice consistency across blocks. "
             "Requires the git build of voxcpm; falls back to global RNG "
             "seeding on pypi 2.0.3 (-1 = unseeded/random)",
    )
    p.add_argument(
        "--min-merged-seconds", type=float, default=4.0,
        help="Batch sentences until ~this much estimated speech (0 disables)",
    )
    p.add_argument(
        "--prebuffer-seconds", type=float, default=1.5,
        help="Hold playback until this much audio is queued (0 disables)",
    )
    p.add_argument(
        "--samples-per-chunk", type=int, default=4096,
        help="AudioChunk size in samples (~85 ms at 48 kHz)",
    )
    p.add_argument(
        "--skip-warmup", action="store_true",
        help="Do not run the startup warm-up synthesis",
    )
    p.add_argument("--voice", default="nicoleheart", help="Voice name reported to HA")
    p.add_argument("--sample-rate", type=int, default=48000, help="Fallback rate")
    args = p.parse_args()

    if args.cloning_mode is None:
        args.cloning_mode = "ultimate" if args.prompt_text else "controllable"
    return args


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if not Path(args.prompt_wav).is_file():
        raise SystemExit(f"prompt WAV not found: {args.prompt_wav}")

    engine = SynthEngine(
        model_id=args.model_id,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        cloning_mode=args.cloning_mode,
        voice_descriptor=args.voice_descriptor,
        default_style=args.style,
        speed=args.speed,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        seed=args.seed,
        constant_identity=not args.no_constant_identity,
    )

    ready = asyncio.Event()

    async def _load_engine() -> None:
        try:
            await asyncio.to_thread(engine.load)
            if not args.skip_warmup:
                await asyncio.to_thread(engine.warm_up)
        except Exception:
            _LOGGER.critical("model load failed", exc_info=True)
            os._exit(1)
        _LOGGER.info("engine ready — synthesis enabled")
        ready.set()

    loader = asyncio.create_task(_load_engine())

    attribution = Attribution(name="OpenBMB", url="https://github.com/OpenBMB/VoxCPM")
    info = Info(
        tts=[
            TtsProgram(
                name="voxcpm2",
                attribution=attribution,
                installed=True,
                description="VoxCPM2 via direct Wyoming bridge",
                version="2B",
                voices=[
                    TtsVoice(
                        name=args.voice,
                        attribution=attribution,
                        installed=True,
                        description=f"VoxCPM2 cloned voice ({args.voice})",
                        version="2B",
                        languages=["en"],
                    )
                ],
                supports_synthesize_streaming=True,
            )
        ]
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info(
        "listening on %s (model=%s) — loading model in background",
        args.uri, args.model_id,
    )

    server_task = asyncio.create_task(
        server.run(
            partial(
                BridgeHandler,
                engine=engine,
                info=info,
                voice=args.voice,
                sample_rate=args.sample_rate,
                ready=ready,
                min_merged_seconds=args.min_merged_seconds,
                prebuffer_seconds=args.prebuffer_seconds,
                samples_per_chunk=args.samples_per_chunk,
            )
        )
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, server_task.cancel)
    try:
        await server_task
    except asyncio.CancelledError:
        _LOGGER.info("server stopped")


if __name__ == "__main__":
    asyncio.run(main())
