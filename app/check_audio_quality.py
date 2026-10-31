
from pathlib import Path
import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "new_recordings"

files = [
    "A_original.wav",
    "A_2.wav",
    "A_3.wav",
    "A_4.wav",
    "A_5.wav",
    "B_original.wav",
    "B_2.wav",
    "B_3.wav"
]

print("========== AUDIO QUALITY CHECK ==========")

for filename in files:
    path = AUDIO_DIR / filename

    if not path.exists():
        print(f"Missing: {filename}")
        continue

    audio, sr = librosa.load(
        path,
        sr=None,
        mono=True
    )

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    silence_ratio = np.mean(np.abs(audio) < 0.001)
    clipping_ratio = np.mean(np.abs(audio) >= 0.999)

    print(f"\nFile: {filename}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(audio) / sr:.2f} seconds")
    print(f"RMS loudness: {rms:.5f}")
    print(f"Peak amplitude: {peak:.5f}")
    print(f"Near-silence: {silence_ratio * 100:.2f}%")
    print(f"Clipping: {clipping_ratio * 100:.2f}%")

print("\nAudio quality check completed!")
