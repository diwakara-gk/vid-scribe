# vid-scribe

Convert video/audio to text transcripts with Whisper.

- **Apple Silicon Mac:** MLX (`mlx-whisper`) for fast on-device processing
- **Windows / Linux / Intel Mac:** `faster-whisper` (CUDA GPU if available, else CPU)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
brew install ffmpeg            # macOS only (Windows/Linux: not needed)
```

## Usage
Put files in `input/`, then run:
```bash
python vid_scribe.py                                   # input/ -> output/
python vid_scribe.py IN_DIR OUT_DIR --language hi --timestamps
```

| Option | Default | Description |
|---|---|---|
| `INPUT_DIR` | `./input` | Folder with media files |
| `OUTPUT_DIR` | `./output` | Folder for transcripts |
| `--model` | `large-v3-turbo` | Whisper model (e.g. `small`, `medium`) |
| `--language` | `en` | Spoken language code, or `auto` to detect |
| `--timestamps` | off | Also write `<name>.json` with segment start/end |

Supported: `.mp4 .mov .mkv .avi .flv .webm .wav .mp3 .m4a .flac`

Finished files are skipped, so an interrupted run resumes where it stopped. `input/` and `output/` contents are git-ignored.
