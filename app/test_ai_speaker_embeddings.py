
from pathlib import Path
from itertools import combinations

import librosa
import torch
import numpy as np

from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "new_recordings"

recordings = {
    "A1": "A_original.wav",
    "A2": "A_2.wav",
    "A3": "A_3.wav",
    "A4": "A_4.wav",
    "A5": "A_5.wav",
    "B1": "B_original.wav",
    "B2": "B_2.wav",
    "B3": "B_3.wav"
}

print("Loading ECAPA-TDNN model...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}
)

embeddings = {}

# Generate embeddings
for speaker, filename in recordings.items():
    file_path = AUDIO_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    print(f"\nProcessing {speaker}: {filename}")

    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    signal = torch.tensor(
        audio, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze().cpu().numpy()
    embeddings[speaker] = embedding

    print("Duration:", round(len(audio) / sr, 2), "seconds")
    print("Embedding shape:", embedding.shape)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# All same-speaker comparisons
same_speaker_pairs = (
    list(combinations(["A1", "A2", "A3", "A4", "A5"], 2))
    + list(combinations(["B1", "B2", "B3"], 2))
)

# All different-speaker comparisons
different_speaker_pairs = [
    (a, b)
    for a in ["A1", "A2", "A3", "A4", "A5"]
    for b in ["B1", "B2", "B3"]
]

same_scores = []
different_scores = []

print("\n========== SAME-SPEAKER SIMILARITY ==========")

for first, second in same_speaker_pairs:
    score = cosine_similarity(
        embeddings[first], embeddings[second]
    )
    same_scores.append(score)
    print(f"{first} vs {second}: {score:.4f}")

print("\n========== DIFFERENT-SPEAKER SIMILARITY ==========")

for first, second in different_speaker_pairs:
    score = cosine_similarity(
        embeddings[first], embeddings[second]
    )
    different_scores.append(score)
    print(f"{first} vs {second}: {score:.4f}")


# Individual speaker averages
print("\n========== SPEAKER-WISE RESULTS ==========")

for group_name, group in [
    ("Speaker A", ["A1", "A2", "A3", "A4", "A5"]),
    ("Speaker B", ["B1", "B2", "B3"])
]:
    scores = [
        cosine_similarity(embeddings[x], embeddings[y])
        for x, y in combinations(group, 2)
    ]

    print(
        f"{group_name}: mean={np.mean(scores):.4f}, "
        f"min={np.min(scores):.4f}, max={np.max(scores):.4f}"
    )


print("\n========== EXPERIMENT SUMMARY ==========")

print("Total recordings:", len(embeddings))
print("Embedding dimensions:", len(next(iter(embeddings.values()))))
print("Same-speaker comparisons:", len(same_scores))
print("Different-speaker comparisons:", len(different_scores))

print(
    "Average same-speaker similarity:",
    round(float(np.mean(same_scores)), 4)
)

print(
    "Average different-speaker similarity:",
    round(float(np.mean(different_scores)), 4)
)

print("\nExperiment completed!")
