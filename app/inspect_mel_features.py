
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import matplotlib.pyplot as plt

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"

MODEL_DIR = Path("pretrained_models/spkrec-ecapa-voxceleb")

AUDIO_PATH = Path("data/test_audio/voice1.wav")

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_RATE = 16000


# --------------------------------------------------
# 2. LOAD PRETRAINED ECAPA-TDNN
# --------------------------------------------------

print("\nSTEP 1: Loading ECAPA-TDNN...")

model = EncoderClassifier.from_hparams(
    source=MODEL_ID,
    savedir=str(MODEL_DIR.resolve()),
    run_opts={"device": "cpu"},
    local_strategy=LocalStrategy.COPY,
)

model.eval()

print("Model loaded successfully!")


# --------------------------------------------------
# 3. LOAD AUDIO
# --------------------------------------------------

print("\nSTEP 2: Loading audio...")

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_PATH}"
    )

audio, sample_rate = sf.read(
    str(AUDIO_PATH),
    dtype="float32",
    always_2d=True,
)

print("Original sample rate:", sample_rate)
print("Original audio shape:", audio.shape)

# Convert stereo or multichannel audio to mono.
mono_audio = np.mean(audio, axis=1)

# Check that audio contains valid samples.
if not np.isfinite(mono_audio).all():
    raise ValueError("Audio contains invalid values.")

if len(mono_audio) == 0:
    raise ValueError("Audio file is empty.")

duration = len(mono_audio) / sample_rate

print(f"Duration: {duration:.2f} seconds")

if np.max(np.abs(mono_audio)) < 1e-6:
    raise ValueError("Audio appears to contain silence.")


# --------------------------------------------------
# 4. CONVERT TO PYTORCH WAVEFORM
# --------------------------------------------------

waveform = torch.from_numpy(
    mono_audio.copy()
).unsqueeze(0)

# Resample to 16 kHz if required.
if sample_rate != TARGET_SAMPLE_RATE:

    waveform = torchaudio.functional.resample(
        waveform,
        orig_freq=sample_rate,
        new_freq=TARGET_SAMPLE_RATE,
    )

    sample_rate = TARGET_SAMPLE_RATE

print("\nSTEP 3: Audio preprocessing completed.")
print("Final sample rate:", sample_rate)
print("Waveform shape:", waveform.shape)


# --------------------------------------------------
# 5. EXTRACT MEL FILTER-BANK FEATURES
# --------------------------------------------------

print("\nSTEP 4: Extracting Mel filter-bank features...")

with torch.no_grad():

    # Use the SAME feature extractor configured
    # inside the pretrained ECAPA-TDNN model.
    mel_features = model.mods.compute_features(
        waveform
    )

    # Apply the model's configured feature normalization.
    wav_lengths = torch.ones(
        waveform.shape[0],
        device=waveform.device,
    )

    normalized_features = model.mods.mean_var_norm(
        mel_features,
        wav_lengths,
    )

print("Mel feature shape:", mel_features.shape)
print("Normalized feature shape:", normalized_features.shape)

print("Number of Mel bins:", mel_features.shape[-1])
print("Number of time frames:", mel_features.shape[1])

print("All feature values finite:",
      torch.isfinite(mel_features).all().item())


# --------------------------------------------------
# 6. SAVE MEL FEATURE VALUES
# --------------------------------------------------

mel_array = mel_features.squeeze(0).cpu().numpy()

np.save(
    OUTPUT_DIR / "voice1_mel_features.npy",
    mel_array,
)

np.savetxt(
    OUTPUT_DIR / "voice1_mel_features.csv",
    mel_array,
    delimiter=",",
)

print("\nMel feature values saved successfully.")


# --------------------------------------------------
# 7. VISUALIZE MEL FEATURES
# --------------------------------------------------

print("\nSTEP 5: Generating Mel feature visualization...")

plt.figure(figsize=(12, 5))

plt.imshow(
    mel_array.T,
    origin="lower",
    aspect="auto",
    interpolation="nearest",
)

plt.title("ECAPA-TDNN Mel Filter-Bank Features")
plt.xlabel("Time Frames")
plt.ylabel("Mel Frequency Bins")
plt.colorbar(label="Feature Energy")

plt.tight_layout()

plot_path = OUTPUT_DIR / "voice1_mel_features.png"

plt.savefig(plot_path, dpi=200)
plt.show()

print("Mel visualization saved:", plot_path)

print("\nSTEP 6: Extracting speaker embedding...")

with torch.no_grad():

    embedding = model.encode_batch(waveform)

print("Embedding shape:", embedding.shape)
print("Embedding dimension:", embedding.shape[-1])

assert embedding.shape[-1] == 192

print("\nSUCCESS: Feature extraction and embedding completed.")
