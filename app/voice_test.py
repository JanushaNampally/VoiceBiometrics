
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
print("STEP 3: Model loaded.")

def get_embedding(audio_path):
    print(f"Loading audio: {audio_path.name}")
    print("Original sample rate:", sample_rate)


    print("Original sample rate:", sample_rate)
    print("Audio shape:", tuple(waveform.shape))

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, 16000
        )

    with torch.no_grad():
        embedding = model.encode_batch(waveform)

    print("Embedding shape:", tuple(embedding.shape))
    return embedding.reshape(1, -1)

def main():
    file1 = AUDIO_DIR / "voice1.wav"
    file2 = AUDIO_DIR / "voice2.wav"

    print("STEP 4: Checking audio files...")
    print("voice1.wav exists:", file1.exists())
    print("voice2.wav exists:", file2.exists())

    if not file1.exists() or not file2.exists():
        print("ERROR: Put both WAV files inside data/test_audio.")
        return

    print("STEP 5: Extracting voice embeddings...")
    emb1 = get_embedding(file1)
    emb2 = get_embedding(file2)

    score = torch.nn.functional.cosine_similarity(
        emb1, emb2, dim=1
    ).item()

    print("\n" + "=" * 40)
    print(f"COSINE SIMILARITY: {score:.4f}")
    print("=" * 40)
    print("This is a similarity score, not proof of identity.")

if __name__ == "__main__":
    main()
