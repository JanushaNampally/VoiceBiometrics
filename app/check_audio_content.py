
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "research" / "dataset_manifest.csv"

df = pd.read_csv(MANIFEST)

def load_audio(speaker_id, password_id):
    rows = df[
        (df["speaker_id"] == speaker_id) &
        (df["password_id"] == password_id)
    ]

    if rows.empty:
        print(f"Missing: Speaker {speaker_id}, Pwd{password_id}")
        return None

    path = ROOT / rows.iloc[0]["file_path"]
    audio, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )

    audio = audio.mean(axis=1)
    digest = hashlib.sha256(audio.tobytes()).hexdigest()

    print(
        f"Speaker {speaker_id}, Pwd{password_id}: "
        f"SR={sr}, Samples={len(audio)}, "
        f"Duration={len(audio)/sr:.2f}s, "
        f"Hash={digest[:16]}"
    )

    return audio

print("\n--- SAME SPEAKER: DIFFERENT PASSWORDS ---")

recordings = {}

for pwd in range(1, 7):
    audio = load_audio(10, pwd)
    if audio is not None:
        recordings[pwd] = audio

print("\n--- COMPARISON FOR SPEAKER 10 ---")

passwords = list(recordings.keys())

for i in range(len(passwords)):
    for j in range(i + 1, len(passwords)):
        p1, p2 = passwords[i], passwords[j]
        a, b = recordings[p1], recordings[p2]

        if len(a) == len(b):
            difference = np.max(np.abs(a - b))
            print(
                f"Pwd{p1} vs Pwd{p2}: "
                f"Identical={np.array_equal(a, b)}, "
                f"Max difference={difference:.8f}"
            )
        else:
            print(
                f"Pwd{p1} vs Pwd{p2}: "
                f"Different sample lengths ({len(a)}, {len(b)})"
            )

print("\n--- DIFFERENT SPEAKERS: SAME PASSWORD ---")

load_audio(100, 1)
