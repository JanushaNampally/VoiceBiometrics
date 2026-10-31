
from pathlib import Path
import pandas as pd
import soundfile as sf

# Change this only if your actual dataset folder is different.
DATASET_DIR = Path("data/Multi Dataset/Voice data")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}

records = []

for file_path in DATASET_DIR.rglob("*"):

    if not file_path.is_file():
        continue

    if file_path.suffix.lower() not in AUDIO_EXTENSIONS:
        continue

    # Example filename: Voice_Code_ID(62) Pwd1.mp3
    filename = file_path.stem

    import re

    speaker_match = re.search(r"ID\((\d+)\)", filename)
    password_match = re.search(r"Pwd(\d+)", filename, re.IGNORECASE)

    if not speaker_match or not password_match:
        print("Skipping unrecognized filename:", file_path.name)
        continue

    speaker_id = int(speaker_match.group(1))
    password_id = int(password_match.group(1))

    try:
        info = sf.info(str(file_path))

        records.append({
            "speaker_id": speaker_id,
            "password_id": password_id,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels
        })

    except Exception as error:
        print("Could not read:", file_path.name, error)

df = pd.DataFrame(records)

if df.empty:
    print("No matching audio files found. Check DATASET_DIR.")
else:
    output_dir = Path("research")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "dataset_manifest.csv"
    df.to_csv(output_file, index=False)

    print("\nDATASET SUMMARY")
    print("-------------------------")
    print("Total recordings:", len(df))
    print("Unique speakers:", df["speaker_id"].nunique())
    print("Password categories:", sorted(df["password_id"].unique()))
    print("\nRecordings per speaker:")
    print(df.groupby("speaker_id").size().describe())

    print("\nRecordings per password:")
    print(df.groupby("password_id").size())

    print("\nManifest saved to:", output_file.resolve())
