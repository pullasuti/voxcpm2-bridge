#!/usr/bin/env python3
"""Render one sentence through the bridge's exact generation path with
different seeds / voice descriptors / cloning modes, so you can compare the
WAVs by ear and pick the combination that sounds most like the clone.

VoxCPM2 cloning is stochastic: a fixed seed pins ONE realisation, and some
seeds sound much closer to the reference than others. Scan a few, then put
the winner in .env as VOICE_SEED.

Run it from the bridge folder, with the bridge's venv:

  source .env
  .venv/bin/python bench_voxcpm2.py --prompt-wav "$PROMPT_WAV" \
      --descriptor "" --no-constant-identity                 # = current setup
  .venv/bin/python bench_voxcpm2.py --prompt-wav "$PROMPT_WAV" \
      --descriptor "a warm, natural, conversational voice"   # milder clause
  .venv/bin/python bench_voxcpm2.py --prompt-wav "$PROMPT_WAV" \
      --prompt-text "$PROMPT_TEXT" --mode ultimate           # best-clone reference

Ultimate mode ignores style instructions and tags (upstream behaviour), so
it only shows how close the clone CAN get.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import bridge_voxcpm2 as bv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-wav",
                    default=os.environ.get("PROMPT_WAV") or "voices/nicoleheart.wav",
                    help="reference clip (default: $PROMPT_WAV, else "
                         "voices/nicoleheart.wav like run.sh)")
    ap.add_argument("--prompt-text", default=os.environ.get("PROMPT_TEXT", ""),
                    help="exact transcript (needed for --mode ultimate; "
                         "default: $PROMPT_TEXT)")
    ap.add_argument("--mode", choices=("controllable", "ultimate"),
                    default="controllable")
    ap.add_argument("--descriptor", default=bv.DEFAULT_BASE_VOICE,
                    help="identity clause; '' = delivery-only instructions")
    ap.add_argument("--no-constant-identity", action="store_true",
                    help="unstyled sentences get no prefix at all")
    ap.add_argument("--style", default="",
                    help="optional style tag to apply, e.g. warm, sad")
    ap.add_argument("--text",
                    default="Well, that is a surprise. [laughing] I did not "
                            "expect that at all. Please, come in and sit down.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 7, 42, 123])
    ap.add_argument("--cfg-value", type=float, default=2.0)
    ap.add_argument("--timesteps", type=int, default=10)
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID", "openbmb/VoxCPM2"))
    ap.add_argument("--out", default="bench_out")
    args = ap.parse_args()

    if not os.path.isfile(args.prompt_wav):
        ap.error(f"reference wav not found: {args.prompt_wav!r} - pass "
                 "--prompt-wav voices/yourfile.wav (a plain 'source .env' "
                 "does not export PROMPT_WAV, so $PROMPT_WAV may be empty)")
    if args.mode == "ultimate" and not args.prompt_text.strip():
        ap.error("--mode ultimate needs --prompt-text")

    eng = bv.SynthEngine(
        model_id=args.model_id, prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text, cloning_mode=args.mode,
        voice_descriptor=args.descriptor, default_style=None, speed=1.0,
        cfg_value=args.cfg_value, inference_timesteps=args.timesteps,
        seed=args.seeds[0], constant_identity=not args.no_constant_identity,
    )
    eng.load()

    text = f"[{args.style}] {args.text}" if args.style else args.text
    clean, style = bv.parse_tags(text, eng.style_tags, eng.allow_inline_tags,
                                 base_voice=args.descriptor)
    os.makedirs(args.out, exist_ok=True)
    tag = "desc" if args.descriptor else "nodesc"
    if args.no_constant_identity:
        tag += "_noident"

    print(f"mode={args.mode} descriptor={args.descriptor!r} "
          f"cfg={args.cfg_value} steps={args.timesteps}")
    print(f"text : {clean!r}\nstyle: {style!r}\n")
    print("seed  audio_s  gen_s   file")
    for seed in args.seeds:
        eng.seed = seed
        t0 = time.monotonic()
        pcm = b"".join(bv.to_pcm16(o) for o in eng._generate(clean, style))
        gen = time.monotonic() - t0
        audio = np.frombuffer(pcm, dtype="<i2")
        path = os.path.join(args.out, f"{args.mode}_{tag}_seed{seed}.wav")
        sf.write(path, audio, eng.sample_rate)
        print(f"{seed:4d}  {len(audio) / eng.sample_rate:7.2f}  {gen:5.2f}   {path}")
    print(f"\nreference for comparison: {args.prompt_wav}")


if __name__ == "__main__":
    main()
