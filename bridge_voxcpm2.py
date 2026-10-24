#!/usr/bin/env python3
"""VoxCPM2 Wyoming TTS bridge — direct variant (no HTTP).

Adapted from the CosyVoice direct bridge. The pipeline, sentence
merging, prebuffering, uniform re-chunking, Error events and graceful
shutdown are engine-agnostic and carried over unchanged; the engine
adapter and the tag system are VoxCPM2-specific:

  * STREAMING  — model.generate_streaming(text, reference_wav_path=...)
    yields audio chunks as they are produced.
  * STYLE CONTROL — VoxCPM2 steers expression with a parenthetical
    instruction prefix on the text, e.g.
    "(speaking with a happy, excited tone) I got the job!"
    The reference clip fixes the timbre; the parenthetical steers the
    delivery. Sentence tags from the LLM map into this prefix.
  * INLINE NON-VERBAL TAGS — VoxCPM2's own vocabulary, inserted in
    square brackets:
        [laughing] [sigh] [cough] [Uhm] [Shh]
        [Question-ah/-ei/-en/-oh] [Surprise-wa/-yo] [Dissatisfaction-hnn]
    Docs recommend lowercase tags, used sparingly. CosyVoice-era tags
    are aliased where an equivalent exists ([laughter] -> [laughing]);
    tags with no equivalent ([breath], [noise], [lipsmack], [mn]) are
    dropped with a warning rather than fed to the model.
  * 48 kHz OUTPUT — sample rate is taken from the model. Defaults are
    scaled accordingly (chunk sizes are computed from the actual rate;
    --samples-per-chunk default of 4096 ≈ 85 ms at 48 kHz).
  * 16 kHz REFERENCE INPUT — VoxCPM2 expects 16 kHz reference audio.
    If the prompt WAV is at another rate it is resampled once at
    startup (numpy linear interpolation) into the cache dir.

Must run inside a venv with `pip install voxcpm wyoming soundfile`
(Python >= 3.10, < 3.13; PyTorch >= 2.5).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import signal
import sys
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional

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
# Two mechanisms, mirroring VoxCPM2's native control surfaces:
#
#   STYLE tags  -> parenthetical style prefix on the text (TAG_MAP);
#                  sticky for the rest of the utterance, like the
#                  CosyVoice bridge's instruction behavior.
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

# VoxCPM2 documented inline non-verbal tags (keep lowercase; the docs
# recommend lowercase forms).
INLINE_TAGS = {"laughing", "sigh", "cough"}

INTERACTION_TAGS = {
    "uhm", "shh",
    "question-ah", "question-ei", "question-en", "question-oh",
    "surprise-wa", "surprise-yo",
    "dissatisfaction-hnn",
}

# CosyVoice-era tag -> VoxCPM2-native equivalent. Anything not mapped
# here and not in the native vocab is DROPPED with a warning (feeding
# untrained tags to the model risks them being read aloud).
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
PAREN_RE = re.compile(r"\(\s*([a-zA-Z\-]+)\s*\)")

_TERM_RE = re.compile(r"[.!?…][\"'”’)\]]*\s+")
_WORD_RE = re.compile(r"[A-Za-z.]+$")

_ABBREVIATIONS = {
    "e.g", "i.e", "eg", "ie", "etc", "mr", "mrs", "ms", "dr", "prof",
    "sr", "jr", "vs", "st",
}


def _canonical_tag(tag: str) -> Optional[str]:
    """Map any incoming tag to a VoxCPM2-native inline tag, or None.

    None means: not expressible inline — either a STYLE tag (handled
    separately) or to be dropped.
    """
    key = tag.lower()
    if key in INLINE_TAGS or key in INTERACTION_TAGS:
        return key
    return TAG_ALIASES.get(key)


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

    def _scan(candidate: str) -> str:
        """Recursively resolve alias chains ([laughter] -> [laughing])."""
        c = _canonical_tag(candidate)
        if c is not None and (c in INLINE_TAGS or c in INTERACTION_TAGS):
            return c
        return candidate

    for m in TAG_RE.finditer(text):
        tag = m.group(1).strip()
        if m.start() > pos:
            out.append(text[pos:m.start()])

        key = tag.lower()
        # Alias-fold first, so [laughter] behaves like [laughing].
        folded = TAG_ALIASES.get(key, key)

        if folded in INLINE_TAGS or folded in INTERACTION_TAGS:
            out.append(f"[{folded}]")
        elif folded in STYLE_TAGS:
            tags.append(STYLE_TAGS[folded])
        elif key in STYLE_TAGS:  # e.g. alias "confidence"
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
        data = data.mean(axis=1)  # reference input is mono

    duration = len(data) / sr
    idx = np.linspace(0.0, duration, int(duration * 16000), endpoint=False)
    src_idx = np.arange(len(data)) / sr
    resampled = np.interp(idx, src_idx, data).astype(np.float32)

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
        default_style: Optional[str],
        speed: float,
        cfg_value: float,
        inference_timesteps: int,
    ) -> None:
        self.model_id = model_id
        self.prompt_wav = prompt_wav  # raw path; resampled in load()
        self.default_style = default_style
        self.speed = speed
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps

        self.model: Any = None
        self.reference_wav: Optional[str] = None
        self.sample_rate: int = 48000
        self._lock = threading.Lock()

    def load(self) -> None:
        from voxcpm import VoxCPM  # noqa: PLC0415

        _LOGGER.info("loading %s", self.model_id)
        t0 = time.monotonic()

        self.model = VoxCPM.from_pretrained(self.model_id, load_denoiser=False)
        self.sample_rate = int(self.model.tts_model.sample_rate)
        self.reference_wav = ensure_reference_16k(self.prompt_wav)

        _LOGGER.info(
            "model ready in %.1fs (sample_rate=%d, reference=%s)",
            time.monotonic() - t0, self.sample_rate, self.reference_wav,
        )

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

        return self.model.generate_streaming(**kwargs)

    def synthesize_chunks(
        self, text: str, style: Optional[str]
    ) -> list[tuple[bytes, int]]:
        """Blocking. Returns [(pcm16 chunk, sample_rate), ...] one per
        model yield so the play worker can forward them as produced."""
        with self._lock:
            chunks: list[tuple[bytes, int]] = []
            t0 = time.monotonic()
            for i, out in enumerate(self._generate(text, style)):
                if i == 0:
                    _LOGGER.info(
                        "[timing] first chunk: req +%.3fs",
                        time.monotonic() - t0,
                    )
                pcm = to_pcm16(out)
                if pcm:
                    chunks.append((pcm, self.sample_rate))
            dur = sum(len(p) for p, _ in chunks) / (2 * self.sample_rate)
            if dur > 0:
                _LOGGER.info(
                    "[timing] block done: %.2fs audio in %.2fs (rtf %.2f)",
                    dur, time.monotonic() - t0,
                    (time.monotonic() - t0) / dur,
                )
            return chunks


# ---------------------------------------------------------------------------
# Wyoming handler — carried over from the CosyVoice direct bridge
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

        while (text := await self._work_q.get()) is not None:
            clean, style = parse_tags(text)
            if not clean:
                continue

            if style:
                self._instr = style

            _LOGGER.info("synthesizing: text=%r style=%r", clean[:60], self._instr)

            try:
                chunks = await asyncio.to_thread(
                    self.engine.synthesize_chunks, clean, self._instr
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

            for audio, rate in chunks:
                if audio:
                    await self._play_q.put((audio, rate))

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
    p.add_argument("--uri", default="tcp://0.0.0.0:10212", help="Wyoming server URI")
    p.add_argument(
        "--model-id", default="openbmb/VoxCPM2",
        help="HuggingFace model id (weights auto-download on first run)",
    )
    p.add_argument(
        "--prompt-wav", required=True,
        help="Reference speaker WAV for cloning (resampled to 16 kHz if needed)",
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
        default_style=args.style,
        speed=args.speed,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
    )

    ready = asyncio.Event()

    async def _load_engine() -> None:
        try:
            await asyncio.to_thread(engine.load)
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
