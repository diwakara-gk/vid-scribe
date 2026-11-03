import json
import time
from pathlib import Path
import mlx_whisper

INPUT = Path("/Users/d0g0o5z/LocalDrive/pw/02. Notes/audio/")
OUTPUT = Path("/Users/d0g0o5z/LocalDrive/pw/02. Notes/transcript/")
MODEL = "mlx-community/whisper-large-v3-turbo"
SAVE_TIMESTAMPS = False  # True to save segment timestamps as JSON


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT.glob("*.wav"))
    total = len(files)
    print(f"Found {total} WAV files")

    for i, file in enumerate(files, 1):
        txt = OUTPUT / f"{file.stem}.txt"
        js = OUTPUT / f"{file.stem}.json"
        if txt.exists() and (not SAVE_TIMESTAMPS or js.exists()):
            print(f"[{i}/{total}] Skipping: {file.name}")
            continue
        try:
            start = time.time()
            result = mlx_whisper.transcribe(str(file), path_or_hf_repo=MODEL, verbose=False)
            txt.write_text(result["text"].strip(), encoding="utf-8")
            if SAVE_TIMESTAMPS:
                js.write_text(json.dumps({
                    "language": result.get("language"),
                    "segments": [
                        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                        for s in result.get("segments", [])
                    ],
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{total}] Done: {file.name} ({(time.time() - start) / 60:.1f} min)")
        except Exception as e:
            print(f"[{i}/{total}] Failed: {file.name} — {e}")

    print("All files processed!")


if __name__ == "__main__":
    main()
