import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/Users/d0g0o5z/LocalDrive/pw/02. Notes/audio/")
DST = Path("/Users/d0g0o5z/LocalDrive/pw/02. Notes/audio/")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}


def main():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found in PATH")
    DST.mkdir(parents=True, exist_ok=True)
    videos = sorted(f for f in SRC.iterdir() if f.suffix.lower() in VIDEO_EXTS)

    for video in videos:
        wav = DST / f"{video.stem}.wav"
        if wav.exists():
            print(f"Skipping (exists): {wav.name}")
            continue
        print(f"Processing {video.name}...")
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                 "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav)],
                check=True,
            )
            print(f"Saved {wav}")
        except subprocess.CalledProcessError as e:
            wav.unlink(missing_ok=True)
            print(f"Failed: {video.name} — {e}")

    print("All files processed!")


if __name__ == "__main__":
    main()
