
from pathlib import Path
import hashlib
import pandas as pd
import soundfile as sf
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "research" / "dataset_manifest.csv"

AUDIT_OUTPUT = ROOT / "research" / "audio_duplicate_audit.csv"
CLEAN_OUTPUT = ROOT / "research" / "deduplicated_manifest.csv"

df = pd.read_csv(MANIFEST)

results = []

print("Checking all audio files...\n")

for index, row in df.iterrows():
    path = ROOT / str(row["file_path"])

    try:
        audio, sr = sf.read(
            str(path),
            dtype="float32",
            always_2d=True
        )

        audio = audio.mean(axis=1)

        # Include sample rate and decoded audio in the hash.
        audio_bytes = np.asarray(audio, dtype="<f4").tobytes()
        hash_input = str(sr).encode() + audio_bytes
        audio_hash = hashlib.sha256(hash_input).hexdigest()

        results.append({
            "speaker_id": row["speaker_id"],
            "password_id": row["password_id"],
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "sample_rate": sr,
            "duration": len(audio) / sr,
            "audio_hash": audio_hash,
            "status": "OK"
        })

    except Exception as error:
        results.append({
            "speaker_id": row["speaker_id"],
            "password_id": row["password_id"],
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "sample_rate": None,
            "duration": None,
            "audio_hash": None,
            "status": f"ERROR: {error}"
        })

    if (index + 1) % 100 == 0:
        print(f"Checked {index + 1}/{len(df)} files")

audit = pd.DataFrame(results)

# Save the full audit.
audit.to_csv(AUDIT_OUTPUT, index=False)

valid = audit[audit["status"] == "OK"].copy()

# Find exact audio hashes appearing under multiple speaker IDs.
speaker_counts = valid.groupby("audio_hash")["speaker_id"].nunique()
cross_speaker_hashes = speaker_counts[speaker_counts > 1].index

# Find repeated audio within the same speaker.
within_speaker_duplicates = valid.duplicated(
    subset=["speaker_id", "audio_hash"],
    keep="first"
)

# Keep one copy per speaker and exact audio signal.
clean = valid.drop_duplicates(
    subset=["speaker_id", "audio_hash"],
    keep="first"
).copy()

clean.to_csv(CLEAN_OUTPUT, index=False)

print("\n========== DUPLICATE AUDIT RESULTS ==========")
print(f"Original recordings: {len(df)}")
print(f"Successfully checked: {len(valid)}")
print(f"Files with read errors: {len(audit) - len(valid)}")
print(f"Unique audio signals: {valid['audio_hash'].nunique()}")
print(f"Repeated files within same speaker: {within_speaker_duplicates.sum()}")
print(f"Audio signals shared across different speakers: {len(cross_speaker_hashes)}")
print(f"Rows in deduplicated manifest: {len(clean)}")

print("\n========== SPEAKER SUMMARY ==========")
print(
    clean.groupby("speaker_id").size()
    .describe()
    .to_string()
)

if len(cross_speaker_hashes) > 0:
    print("\nWARNING: Identical audio found under different speaker IDs.")
    print("These recordings need investigation before evaluation.")

    cross = valid[
        valid["audio_hash"].isin(cross_speaker_hashes)
    ]

    print(
        cross[["speaker_id", "password_id", "file_name"]]
        .sort_values(["audio_hash", "speaker_id"])
        .head(30)
        .to_string(index=False)
    )

print("\nFull audit saved to:", AUDIT_OUTPUT)
print("Deduplicated manifest saved to:", CLEAN_OUTPUT)
print("\nOriginal audio files have NOT been deleted.")
