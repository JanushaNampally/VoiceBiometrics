import sys
import torch
import torchaudio
import speechbrain
import fastapi

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("TorchAudio:", torchaudio.__version__)
print("SpeechBrain:", speechbrain.__version__)
print("FastAPI:", fastapi.__version__)
print("CUDA available:", torch.cuda.is_available())
print("\nEnvironment check completed.")
