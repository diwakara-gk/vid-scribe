#!/usr/bin/env python3
"""Video/audio -> transcripts. MLX on Apple Silicon, faster-whisper elsewhere.

Usage: python vid_scribe.py [INPUT_DIR] [OUTPUT_DIR] [options]

Defaults: INPUT_DIR=./input, OUTPUT_DIR=./output
Options:
  --model NAME      Whisper model (default: large-v3-turbo)
  --language CODE   Spoken language, e.g. en, hi, or 'auto' to detect (default: en)
  --timestamps      Also write <name>.json with segment start/end times
Finished files are skipped, so reruns resume.
"""
import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".wav", ".mp3", ".m4a", ".flac"}
MLX = sys.platform == "darwin" and platform.machine() == "arm64"
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo" if MLX else "large-v3-turbo"


def make_transcriber(model, language):
    """Return fn(path) -> (text, [{start, end, text}], language)."""
    if MLX:
        import mlx_whisper

        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg not found. Install: brew install ffmpeg")

        def run(path):
            r = mlx_whisper.transcribe(str(path), path_or_hf_repo=model, verbose=False,
                                      language=language)
            segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in r["segments"]]
            return r["text"].strip(), segs, r.get("language")
        return run

    import ctranslate2
    from faster_whisper import WhisperModel

    cuda = ctranslate2.get_cuda_device_count() > 0
    m = WhisperModel(model, device="cuda" if cuda else "cpu",
                     compute_type="float16" if cuda else "int8")

    def run(path):
        it, info = m.transcribe(str(path), vad_filter=True, language=language)
        segs = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in it]
        return " ".join(s["text"] for s in segs), segs, info.language
    return run


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)  # atomic: no half-written files treated as done


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="?", type=Path, default=Path("input"))
    p.add_argument("output", nargs="?", type=Path, default=Path("output"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--language", default="en", help="spoken language code, or 'auto'")
    p.add_argument("--timestamps", action="store_true", help="also write .json segments")
    a = p.parse_args()

    a.input.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in a.input.iterdir()
                   if f.suffix.lower() in EXTS and not f.name.startswith("."))
    if not files:
        sys.exit(f"No media files in {a.input.resolve()}")

    transcribe = make_transcriber(a.model, None if a.language == "auto" else a.language)
    for i, f in enumerate(files, 1):
        txt, js = a.output / f"{f.stem}.txt", a.output / f"{f.stem}.json"
        if txt.exists() and (not a.timestamps or js.exists()):
            continue
        print(f"[{i}/{len(files)}] {f.name}", flush=True)
        try:
            text, segs, lang = transcribe(f)
            if a.timestamps:
                write(js, json.dumps({"language": lang, "segments": segs},
                                     ensure_ascii=False, indent=2))
            write(txt, text)  # last: marks the file done
        except Exception as e:
            print(f"  failed: {e}")


if __name__ == "__main__":
    main()
