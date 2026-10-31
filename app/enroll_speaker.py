
from pathlib import Path

import librosa
import torch
import numpy as np

from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parent.parent

AUDIO_DIR = ROOT / "data" / "new_recordings"
ENROLLMENT_DIR = ROOT / "data" / "enrollments"

ENROLLMENT_DIR.mkdir(parents=True, exist_ok=True)

# Use four Speaker A recordings for the initial demo profile.
ENROLLMENT_FILES = [
    "A_2.wav",
    "A_3.wav",
    "A_4.wav"
]

print("Loading ECAPA-TDNN model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
    run_opts={"device": "cpu"}
)

embeddings = []

for filename in ENROLLMENT_FILES:

    file_path = AUDIO_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing audio file: {file_path}")

    print(f"\nProcessing: {filename}")

    audio, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError(f"Empty audio file: {filename}")

    signal = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze().cpu().numpy()

    # Normalize each embedding before averaging.
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

    embeddings.append(embedding)

    print("Duration:", round(len(audio) / sr, 2), "seconds")
    print("Embedding shape:", embedding.shape)

# Average the normalized embeddings.
profile = np.mean(embeddings, axis=0)

# Normalize the final speaker profile.
profile = profile / (np.linalg.norm(profile) + 1e-10)

output_path = ENROLLMENT_DIR / "speaker_A_profile.npy"

np.save(output_path, profile)

print("\n========== ENROLLMENT COMPLETED ==========")
print("Speaker: A")
print("Enrollment recordings:", len(ENROLLMENT_FILES))
print("Profile dimensions:", profile.shape)
print("Saved profile:", output_path)
