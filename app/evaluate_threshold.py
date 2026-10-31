
from pathlib import Path
from itertools import combinations

import librosa
import torch
import numpy as np

from speechbrain.inference.speaker import EncoderClassifier

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "new_recordings"

speaker_a = {
    "A2": "A_2.wav",
    "A3": "A_3.wav",
    "A4": "A_4.wav",
    "A5": "A_5.wav"
}

speaker_b = {
    "B1": "B_original.wav",
    "B2": "B_2.wav",
    "B3": "B_3.wav"
}

print("Loading ECAPA-TDNN...")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
    run_opts={"device": "cpu"}
)


def get_embedding(filename):
    path = AUDIO_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    audio, sr = librosa.load(path, sr=16000, mono=True)

    signal = torch.tensor(
        audio, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    embedding = embedding.squeeze().cpu().numpy()

    return embedding / (np.linalg.norm(embedding) + 1e-10)


def make_profile(embeddings):
    profile = np.mean(embeddings, axis=0)
    return profile / (np.linalg.norm(profile) + 1e-10)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    ))


print("\nExtracting embeddings...")

a_embeddings = {
    name: get_embedding(filename)
    for name, filename in speaker_a.items()
}

b_embeddings = {
    name: get_embedding(filename)
    for name, filename in speaker_b.items()
}

genuine_scores = []
impostor_scores = []

print("\n========== GENUINE TRIALS ==========")

for test_name in speaker_a:

    enrollment_embeddings = [
        embedding
        for name, embedding in a_embeddings.items()
        if name != test_name
    ]

    temporary_profile = make_profile(enrollment_embeddings)

    score = cosine_similarity(
        temporary_profile,
        a_embeddings[test_name]
    )

    genuine_scores.append(score)

    print(f"{test_name}: {score:.4f}")

    print(f"\nImpostor trials against profile excluding {test_name}:")

    for impostor_name, impostor_embedding in b_embeddings.items():

        impostor_score = cosine_similarity(
            temporary_profile,
            impostor_embedding
        )

        impostor_scores.append(impostor_score)

        print(f"{impostor_name}: {impostor_score:.4f}")


print("\n========== SCORE SUMMARY ==========")

print("Genuine trials:", len(genuine_scores))
print("Impostor trials:", len(impostor_scores))

print(
    "Genuine minimum:",
    round(min(genuine_scores), 4)
)

print(
    "Genuine mean:",
    round(float(np.mean(genuine_scores)), 4)
)

print(
    "Impostor maximum:",
    round(max(impostor_scores), 4)
)

print(
    "Impostor mean:",
    round(float(np.mean(impostor_scores)), 4)
)

print("\n========== THRESHOLD ANALYSIS ==========")

if min(genuine_scores) > max(impostor_scores):

    threshold = (
        min(genuine_scores) + max(impostor_scores)
    ) / 2

    print("A separation exists in these trials.")
    print("Exploratory threshold:", round(threshold, 4))

else:

    print("Genuine and impostor scores overlap.")
    print("No clean threshold separates these trials.")
    print("More independent recordings are needed.")

# Investigate the excluded A1 recording separately.
a1_embedding = get_embedding("A_original.wav")

full_profile = make_profile(list(a_embeddings.values()))

a1_score = cosine_similarity(full_profile, a1_embedding)

print("\nA1 investigation score:", round(a1_score, 4))

print("\nEvaluation completed!")
