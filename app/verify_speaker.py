
from pathlib import Path

import librosa
import torch
import numpy as np

from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parent.parent

PROFILE_PATH = ROOT / "data" / "enrollments" / "speaker_A_profile.npy"

# Test recordings
TEST_FILES = [
    "A_5.wav",
    "B_original.wav",
    "B_2.wav",
    "B_3.wav"
]

AUDIO_DIR = ROOT / "data" / "new_recordings"

# Provisional demonstration threshold only.
THRESHOLD = 0.528

print("Loading enrolled speaker profile...")

if not PROFILE_PATH.exists():
    raise FileNotFoundError("Speaker A profile not found. Run enrollment first.")

profile = np.load(PROFILE_PATH)

print("Profile loaded:", profile.shape)

print("\nLoading ECAPA-TDNN model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
    run_opts={"device": "cpu"}
)


def cosine_similarity(a, b):
    return float(
        np.dot(a, b) /
        ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10)
    )


print("\n========== SPEAKER VERIFICATION ==========")

for filename in TEST_FILES:

    file_path = AUDIO_DIR / filename

    if not file_path.exists():
        print(f"\nMissing file: {filename}")
        continue

    audio, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    signal = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze().cpu().numpy()

    score = cosine_similarity(profile, embedding)

    decision = "ACCEPT" if score >= THRESHOLD else "REJECT"

    print(f"\nTest recording: {filename}")
    print(f"Cosine similarity: {score:.4f}")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Decision: {decision}")

print("\nVerification experiment completed!")
