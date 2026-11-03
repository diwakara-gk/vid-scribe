from pathlib import Path
import whisper

INPUT = Path("/Users/d0g0o5z/LocalDrive/pw/02. Notes/audio/")
OUTPUT = Path("/Users/d0g0o5z/LocalDrive/pw/02. Notes/transcript/")
MODEL = "small"  # "medium" / "large" for better accuracy


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT.glob("*.wav"))
    total = len(files)
    print(f"Found {total} WAV files")
    model = whisper.load_model(MODEL)

    for i, file in enumerate(files, 1):
        txt = OUTPUT / f"{file.stem}.txt"
        if txt.exists():
            print(f"[{i}/{total}] Skipping: {file.name}")
            continue
        print(f"[{i}/{total}] Transcribing: {file.name} → {txt.name}")
        try:
            result = model.transcribe(str(file), fp16=False)
            txt.write_text(result["text"].strip(), encoding="utf-8")
        except Exception as e:
            print(f"[{i}/{total}] Failed: {file.name} — {e}")

    print("✅ All files processed!")


if __name__ == "__main__":
    main()
