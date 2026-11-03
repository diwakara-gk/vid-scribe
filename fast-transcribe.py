from pathlib import Path
from faster_whisper import WhisperModel

INPUT = Path("/Users/d0g0o5z/LocalDrive/pw/Notes/audio/")
OUTPUT = Path("/Users/d0g0o5z/LocalDrive/pw/Notes/transcript/modern history")
MODEL = "small"  # "small", "medium", "large-v3"
COMPUTE_TYPE = "int8"  # int8 = fastest on CPU


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT.glob("*.wav"))
    total = len(files)
    print(f"Found {total} WAV files")
    model = WhisperModel(MODEL, compute_type=COMPUTE_TYPE)

    for i, file in enumerate(files, 1):
        txt = OUTPUT / f"{file.stem}.txt"
        if txt.exists():
            print(f"[{i}/{total}] Skipping: {file.name}")
            continue
        print(f"[{i}/{total}] Transcribing: {file.name} → {txt.name}")
        try:
            segments, _ = model.transcribe(str(file))
            txt.write_text(" ".join(s.text.strip() for s in segments), encoding="utf-8")
        except Exception as e:
            print(f"[{i}/{total}] Failed: {file.name} — {e}")

    print("✅ All files processed!")


if __name__ == "__main__":
    main()
