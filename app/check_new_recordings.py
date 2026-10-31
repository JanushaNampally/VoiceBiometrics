
from pathlib import Path
import librosa

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "new_recordings"

files = [
    "A_original.wav",
    "A_2.wav",
    "A_3.wav",
    "B_original.wav",
    "B_2.wav",
    "B_3.wav"
]

for filename in files:
    file_path = AUDIO_DIR / filename

    print("\nChecking:", filename)

    if not file_path.exists():
        print("ERROR: File not found!")
        continue

    audio, sr = librosa.load(file_path, sr=None, mono=True)

    duration = len(audio) / sr

    print("Sample rate:", sr)
    print("Duration:", round(duration, 2), "seconds")
    print("Audio shape:", audio.shape)

    if duration >= 4:
        print("Status: ACCEPTABLE")
    else:
        print("Status: TOO SHORT - regenerate if possible")
