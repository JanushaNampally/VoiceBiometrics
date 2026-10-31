
from pathlib import Path
import librosa
import numpy as np
import hashlib

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "new_recordings"

files = [
    "A_original.wav", "A_2.wav", "A_3.wav",
    "B_original.wav", "B_2.wav", "B_3.wav"
]

audio_data = {}

for filename in files:
    path = AUDIO_DIR / filename

    if not path.exists():
        print(f"MISSING: {filename}")
        continue

    audio, sr = librosa.load(path, sr=16000, mono=True)
    audio_data[filename] = audio

    print(f"\n{filename}")
    print("Duration:", round(len(audio) / sr, 2), "seconds")
    print("SHA256:", hashlib.sha256(path.read_bytes()).hexdigest()[:16])

print("\n========== AUDIO DIFFERENCE CHECK ==========")

pairs = [
    ("A_original.wav", "A_2.wav"),
    ("A_original.wav", "A_3.wav"),
    ("A_2.wav", "A_3.wav"),
    ("B_original.wav", "B_2.wav"),
    ("B_original.wav", "B_3.wav"),
    ("B_2.wav", "B_3.wav"),
]

for first, second in pairs:
    if first not in audio_data or second not in audio_data:
        continue

    a = audio_data[first]
    b = audio_data[second]

    if len(a) == len(b):
        difference = np.max(np.abs(a - b))
        print(f"{first} vs {second}: max difference = {difference:.8f}")
    else:
        print(f"{first} vs {second}: Different lengths")
