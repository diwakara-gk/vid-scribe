from pathlib import Path
import mlx_whisper

INPUT = Path("/Users/d0g0o5z/LocalDrive/pw/02. Notes/audio/")
OUTPUT = Path("/Users/d0g0o5z/LocalDrive/pw/02. Notes/transcript/")
MODEL = "mlx-community/whisper-large-v3-turbo"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT.glob("*.wav"))
    total = len(files)
    print(f"Found {total} WAV files\n")

    for i, file in enumerate(files, 1):
        txt = OUTPUT / f"{file.stem}.txt"
        if txt.exists():
            print(f"[{i}/{total}] Skipping (already done): {file.name}")
            continue
        print(f"[{i}/{total}] Transcribing: {file.name} → {txt.name}")
        try:
            result = mlx_whisper.transcribe(str(file), path_or_hf_repo=MODEL, verbose=False)
            txt.write_text(result["text"].strip(), encoding="utf-8")
            print("       ✅ Done")
        except Exception as e:
            print(f"       ❌ Failed: {e}")

    print("\n✅ All files processed!")


if __name__ == "__main__":
    main()
