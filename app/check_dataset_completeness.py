
from pathlib import Path
import pandas as pd

df = pd.read_csv("research/dataset_manifest.csv")

expected_speakers = range(1, 151)
expected_passwords = range(1, 7)

existing = set(zip(df["speaker_id"], df["password_id"]))

missing = [
    (speaker, password)
    for speaker in expected_speakers
    for password in expected_passwords
    if (speaker, password) not in existing
]

print("Missing recording combinations:", len(missing))

for speaker, password in missing:
    print(f"Speaker ID {speaker} - Pwd{password}")

print("\nDuplicate speaker/password combinations:")
duplicates = df[
    df.duplicated(["speaker_id", "password_id"], keep=False)
]
print(duplicates[["speaker_id", "password_id", "file_name"]])
