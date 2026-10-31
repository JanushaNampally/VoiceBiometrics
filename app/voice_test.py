
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

from speechbrain.inference.speaker import EncoderClassifier

print("STEP 1: Starting voice test...")

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = BASE_DIR / "data" / "test_audio"
MODEL_DIR = BASE_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"

print("Audio folder:", AUDIO_DIR)
print("STEP 2: Loading ECAPA-TDNN...")

model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(MODEL_DIR),
    run_opts={"device": "cpu"},
    local_strategy=2,
)

model.eval()

print("STEP 3: Model loaded successfully!")


def get_embedding(audio_path):

    print(f"\nLoading audio: {audio_path.name}")

    # Load audio using SoundFile
    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True
    )

    # Convert NumPy audio into a PyTorch tensor
    waveform = torch.from_numpy(audio.T.copy())

    print("Original sample rate:", sample_rate)
    print("Audio shape:", tuple(waveform.shape))

    # Convert stereo audio to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert sample rate to 16 kHz
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=16000
        )

    # Extract speaker embedding
    with torch.no_grad():
        embedding = model.encode_batch(waveform)

    print("Embedding shape:", tuple(embedding.shape))

    return embedding.reshape(1, -1)


def main():

    file1 = AUDIO_DIR / "voice1.wav"
    file2 = AUDIO_DIR / "voice2.wav"

    print("\nSTEP 4: Checking audio files...")

    print("voice1.wav exists:", file1.exists())
    print("voice2.wav exists:", file2.exists())

    if not file1.exists() or not file2.exists():
        print("ERROR: Audio files are missing.")
        return

    print("\nSTEP 5: Extracting voice embeddings...")

    emb1 = get_embedding(file1)
    emb2 = get_embedding(file2)

    print("\nSTEP 6: Comparing voices...")

    score = torch.nn.functional.cosine_similarity(
        emb1,
        emb2,
        dim=1
    ).item()

    print("\n" + "=" * 45)
    print(f"COSINE SIMILARITY: {score:.4f}")
    print("=" * 45)

    print("Voice comparison completed!")
    print("Similarity score is not proof of identity.")


if __name__ == "__main__":
    main()


import numpy as np
import soundfile as sf

audio1, sr1 = sf.read(
    str(AUDIO_DIR / "voice1.wav"),
    dtype="float32"
)

audio2, sr2 = sf.read(
    str(AUDIO_DIR / "voice2.wav"),
    dtype="float32"
)

print("\nAUDIO VALIDATION")
print("Same sample rate:", sr1 == sr2)
print("Same shape:", audio1.shape == audio2.shape)
print("Exactly identical:", np.array_equal(audio1, audio2))
print("Maximum absolute difference:",
      np.max(np.abs(audio1 - audio2)))