#!/usr/bin/env python3
"""VoxCPM2 Wyoming TTS bridge — direct variant (no HTTP), streaming.

Adapted from the CosyVoice direct bridge. Engine-agnostic machinery
carried over unchanged: sentence merging, prebuffering (jitter buffer),
uniform re-chunking, Error events, graceful shutdown, bind-first /
load-later startup. VoxCPM2-specific behavior:

  * VOICE CONSISTENCY — three measures against timbre drift between
    blocks (each merged block is a separate generation call):
      1. FIXED SEED — official examples seed generation (seed=42);
         unseeded runs draw a new random voice bias each call.
         --seed (default 42) pins it.
      2. SIGNATURE-AWARE KWARGS — the public docs advertise
         `seed=` on model.generate(), but generate_streaming()
         forwards to the private _generate(), and not every released
         build accepts it there (TypeError). At load time the bridge
         introspects the signature, drops unsupported kwargs with a
         warning, and — when there is no `seed` parameter — pins the
         global RNGs (random/numpy/torch/CUDA) before each block
         instead, which stabilizes the diffusion sampler's draws.
      3. ULTIMATE CLONING — when --prompt-text (the reference clip's
         transcript) is provided, the reference is supplied through
         BOTH pathways (isolated reference + continuation prefix),
         which the technical report identifies as the highest-SIM
         cloning recipe.
  * TRUE STREAMING — each model yield is forwarded to the play queue
    the moment it is produced (run_coroutine_threadsafe from the GPU
    thread), so audio flows to HA progressively within a block.
  * STARTUP WARM-UP — a short dummy synthesis runs after model load
    so torch.compile's one-time cost is paid during startup rather
    than on HA's first real request. Non-fatal if it fails (the
    compile cost then lands on the first real request instead).
  * STYLE CONTROL — parenthetical instruction prefix on the text;
    sentence tags from the LLM map into this prefix.
  * INLINE NON-VERBAL TAGS — VoxCPM2's own vocabulary:
        [laughing] [sigh] [cough] [Uhm] [Shh]
        [Question-ah/-ei/-en/-oh] [Surprise-wa/-yo] [Dissatisfaction-hnn]
    CosyVoice-era tags are aliased where an equivalent exists;
    unmatched tags ([noise], [lipsmack], [mn]) are dropped with a
    warning rather than fed to the model.
  * 48 kHz OUTPUT, 16 kHz REFERENCE INPUT — reference resampled once
    at startup (polyphase filter if scipy is available, linear
    interpolation otherwise).

Known upstream limitation (worth remembering): the docs themselves
note that cloning/style results vary between runs and recommend
generating 1-3 times for the desired output; a fixed seed mitigates
but does not fully eliminate drift. For production-grade consistency
upstream recommends LoRA fine-tuning on 5-10 minutes of clean audio.

Must run inside a venv with `pip install voxcpm wyoming soundfile`
(Python >= 3.10, < 3.13; PyTorch >= 2.5).
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

# ---------------------------------------------------------------------------
# Tag handling
#
#   STYLE tags  -> parenthetical style prefix on the text (STYLE_TAGS);
#                  sticky for the rest of the utterance.
#   INLINE tags -> passed through into the text in VoxCPM2's own
#                  vocabulary (INLINE_TAGS + INTERACTION_TAGS).
#
# ALIASES folds CosyVoice-era vocabulary into VoxCPM2's equivalents so
# existing LLM prompts keep working.
# ---------------------------------------------------------------------------

STYLE_TAGS = {
    "warm": "speaking in a warm and friendly voice",
    "confident": "speaking with a confident and steady voice",
    "happy": "speaking with a happy and excited tone",
    "sad": "speaking with a sad and melancholic tone",
    "angry": "speaking with an angry tone",
    "excited": "speaking with an excited and energetic tone",
    "whisper": "speaking in a very soft whisper",
    "serious": "speaking in a serious and calm manner",
    "loud": "speaking as loudly as possible",
    "slow": "speaking slowly",
    "fast": "speaking quickly",
}

INLINE_TAGS = {"laughing", "sigh", "cough"}

INTERACTION_TAGS = {
    "uhm", "shh",
    "question-ah", "question-ei", "question-en", "question-oh",
    "surprise-wa", "surprise-yo",
    "dissatisfaction-hnn",
}

# CosyVoice-era tag -> VoxCPM2-native equivalent. Anything not mapped
# here and not in the native vocab is DROPPED with a warning.
TAG_ALIASES = {
    "laughter": "laughing",
    "laughs": "laughing",
    "laughing": "laughing",
    "laugh": "laughing",
    "sighs": "sigh",
    "sigh": "sigh",
    "coughs": "cough",
    "cough": "cough",
    "breath": "sigh",       # closest documented equivalent
    "breaths": "sigh",
    "um": "uhm",
    "hm": "uhm",
    # no VoxCPM2 equivalent -> dropped: noise, lipsmack, mn
}

TAG_RE = re.compile(r"\s*[\[\{]([^\]\}]+)[\]\}]")

_TERM_RE = re.compile(r"[.!?…][\"'”’)\]]*\s+")
_WORD_RE = re.compile(r"[A-Za-z.]+$")

_ABBREVIATIONS = {
    "e.g", "i.e", "eg", "ie", "etc", "mr", "mrs", "ms", "dr", "prof",
    "sr", "jr", "vs", "st",
}


def parse_tags(text: str) -> tuple[str, Optional[str]]:
    """Split text into (spoken text, style body or None).

    Square-bracket tags whose canonical form is a style keyword become
    the (sticky) style instruction; those with an inline equivalent are
    passed through in VoxCPM2's vocabulary; anything else is dropped
    with a warning.
    """
    tags: list[str] = []
    out: list[str] = []
    pos = 0

    for m in TAG_RE.finditer(text):
        tag = m.group(1).strip()
        if m.start() > pos:
            out.append(text[pos:m.start()])

        key = tag.lower()
        folded = TAG_ALIASES.get(key, key)

        if folded in INLINE_TAGS or folded in INTERACTION_TAGS:
            out.append(f"[{folded}]")
        elif folded in STYLE_TAGS:
            tags.append(STYLE_TAGS[folded])
        elif key in STYLE_TAGS:
            tags.append(STYLE_TAGS[key])
        else:
            _LOGGER.warning("dropping tag %r (no VoxCPM2 equivalent)", tag)

        pos = m.end()

    if pos < len(text):
        out.append(text[pos:])

    clean = " ".join("".join(out).split())
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
        default_style: Optional[str],
        speed: float,
        cfg_value: float,
        inference_timesteps: int,
        seed: int,
    ) -> None:
        self.model_id = model_id
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.default_style = default_style
        self.speed = speed
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.seed = seed

        self.model: Any = None
        self.reference_wav: Optional[str] = None
        self.sample_rate: int = 48000
        self._lock = threading.Lock()

        # Populated by _introspect_signature() at load time.
        #   _accepted is None  -> signature unknown / accepts anything,
        #                         so kwargs are passed unfiltered.
        #   _accepted is a set -> only these names may be passed.
        self._accepted: Optional[set[str]] = None
        self._uses_seed_param = False

    # -- capability discovery ------------------------------------------------

    def _introspect_signature(self) -> None:
        """Work out which kwargs the installed voxcpm build accepts.

        The docs advertise `seed=` on model.generate(), but
        generate_streaming() forwards to the private _generate(), and
        some released builds don't accept it there — passing it raises
        TypeError. Rather than guess, inspect the real signature.
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
            # Takes **kwargs: nothing to filter against.
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
                "global RNG seeding for voice consistency"
            )

    # -- seeding -------------------------------------------------------------

    def _seed_rngs(self) -> None:
        """Version-independent consistency fallback: reseed the global
        RNGs before each block so the diffusion sampler draws the same
        random numbers every time. Used only when the model exposes no
        `seed` parameter of its own."""
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

        if self.prompt_text:
            _LOGGER.info(
                "ultimate cloning enabled: reference supplied via both "
                "pathways (reference + continuation)"
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
                # Drain; we only care that the kernels get built.
                _ = to_pcm16(out)
        except Exception:
            # A failed warm-up is not fatal — the first real request
            # will just pay the compile cost instead.
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

    def _generate(self, text: str, style: Optional[str]) -> Iterator[Any]:
        kwargs: dict[str, Any] = {
            "text": self._styled_text(text, style),
            "cfg_value": self.cfg_value,
            "inference_timesteps": self.inference_timesteps,
        }

        if self.reference_wav:
            kwargs["reference_wav_path"] = self.reference_wav
            if self.prompt_text:
                # Ultimate cloning: supply the reference through BOTH
                # pathways — isolated reference AND continuation
                # prefix with its transcript. Per the technical report
                # this is the highest-similarity cloning recipe.
                kwargs["prompt_wav_path"] = self.reference_wav
                kwargs["prompt_text"] = self.prompt_text

        if self.seed >= 0:
            if self._uses_seed_param:
                kwargs["seed"] = self.seed
            else:
                # No seed kwarg in this build — pin the global RNGs so
                # every block draws identical randomness.
                self._seed_rngs()

        # Drop anything this build's signature doesn't accept, so a
        # version mismatch degrades gracefully instead of raising.
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
            # Called from the GPU thread; bounce onto the event loop.
            asyncio.run_coroutine_threadsafe(
                _put((pcm, self.engine.sample_rate)), loop
            ).result()

        while (text := await self._work_q.get()) is not None:
            clean, style = parse_tags(text)
            if not clean:
                continue

            if style:
                self._instr = style

            _LOGGER.info("synthesizing: text=%r style=%r", clean[:60], self._instr)

            try:
                # Streams: each model yield lands on the play queue
                # the moment it is produced.
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

    # -- wyoming event handling -----------------------------------------------------

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        if SynthesizeStart.is_type(event.type):
            self._stash = ""
            self._buf = ""
            self._got_chunks = False
            self._audio_started = False
            self._instr = None
            self._pending_sentences = []
            await self._start_pipeline()
            return True

        if SynthesizeChunk.is_type(event.type):
            self._got_chunks = True
            for sentence in self._take_sentences(
                SynthesizeChunk.from_event(event).text
            ):
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
                if text.strip():
                    self._feed_sentence(text.strip())

            await self._stop_pipeline()

            if self._audio_started:
                await self.write_event(AudioStop().event())
                self._audio_started = False
            await self.write_event(SynthesizeStopped().event())
            return True

        if Synthesize.is_type(event.type):
            self._stash = Synthesize.from_event(event).text
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
             "cloning (highest voice similarity). Strongly recommended.",
    )
    p.add_argument(
        "--style",
        default=None,
        help="Baseline style parenthetical applied to every utterance",
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
             "Used as the model's `seed` kwarg when supported, otherwise "
             "applied to the global RNGs (-1 = unseeded/random)",
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
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if not Path(args.prompt_wav).is_file():
        raise SystemExit(f"prompt WAV not found: {args.prompt_wav}")

    engine = SynthEngine(
        model_id=args.model_id,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        default_style=args.style,
        speed=args.speed,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        seed=args.seed,
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
