import torchaudio
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "research" / "dataset_manifest.csv"
MODEL_DIR = ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"

OUTPUT = ROOT / "research" / "speaker_embeddings.csv"

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
TARGET_SR = 16000

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

print("\nSTEP 1: Loading dataset manifest...")

df = pd.read_csv(MANIFEST)

print("Recordings:", len(df))
print("Speakers:", df["speaker_id"].nunique())
print("Passwords:", sorted(df["password_id"].unique()))

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nSTEP 2: Loading ECAPA-TDNN...")

model = EncoderClassifier.from_hparams(
    source=MODEL_ID,
    savedir=str(MODEL_DIR.resolve()),
    run_opts={"device": "cpu"},
    local_strategy=LocalStrategy.COPY,
)

model.eval()
print("ECAPA-TDNN loaded successfully.")

# --------------------------------------------------
# EXTRACT SPEAKER EMBEDDINGS
# --------------------------------------------------

print("\nSTEP 3: Extracting speaker embeddings...")

results = []

for index, row in df.iterrows():

    audio_path = ROOT / row["file_path"]

    if not audio_path.exists():
        print("Missing:", audio_path)
        continue

    try:
        audio, sr = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=True
        )

        # Convert stereo to mono if necessary
        audio = audio.mean(axis=1)

        waveform = torch.from_numpy(
            audio.copy()
        ).unsqueeze(0)

        # Resample to 16 kHz
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(
                waveform, sr, TARGET_SR
            )

        # Prevent invalid / silent inputs
        if waveform.numel() == 0:
            print("Empty audio:", audio_path.name)
            continue

        if not torch.isfinite(waveform).all():
            print("Invalid audio:", audio_path.name)
            continue

        if waveform.abs().max().item() < 1e-6:
            print("Silent audio:", audio_path.name)
            continue

        with torch.no_grad():
            embedding = model.encode_batch(waveform)

        vector = embedding.squeeze().cpu().numpy()

        results.append({
            "speaker_id": int(row["speaker_id"]),
            "password_id": int(row["password_id"]),
            "file_name": row["file_name"],
            "embedding": ",".join(
                map(str, vector.tolist())
            )
        })

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(df)} recordings"
            )

    except Exception as error:
        print(
            "Failed:",
            row["file_name"],
            type(error).__name__,
            str(error)
        )

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

print("\nSTEP 4: Saving embeddings...")

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT, index=False)

print("Successfully processed:", len(result_df))
print("Embedding dimensions: 192")
print("Saved to:", OUTPUT)

print("\nEXPERIMENT PREPARATION COMPLETED.")
