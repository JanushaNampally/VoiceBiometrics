
from pathlib import Path
import soundfile as sf

# Dataset is inside the data folder.
DATASET_DIR = Path("data/Multi Dataset/Voice data")

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}

print("\nVOICE DATASET INSPECTION")
print("=" * 55)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR.resolve()}"
    )

audio_files = [
    path for path in DATASET_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in AUDIO_EXTENSIONS
]

print("Dataset location:", DATASET_DIR.resolve())
print("Total audio files:", len(audio_files))

print("\nFOLDER STRUCTURE")
print("-" * 55)

for path in sorted(DATASET_DIR.rglob("*")):
    if path.is_dir():
        print("FOLDER:", path.relative_to(DATASET_DIR))

print("\nAUDIO FILE DETAILS")
print("-" * 55)

for path in sorted(audio_files):
    try:
        info = sf.info(str(path))

        print(
            f"{path.relative_to(DATASET_DIR)} | "
            f"Duration: {info.duration:.2f}s | "
            f"Sample rate: {info.samplerate} | "
            f"Channels: {info.channels}"
        )

    except Exception as error:
        print(
            f"{path.name} | "
            f"Could not inspect: {error}"
        )

print("\nINSPECTION COMPLETED")
