
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
MODEL_DIR = Path("pretrained_models/spkrec-ecapa-voxceleb")

print("Loading pretrained ECAPA-TDNN model...")

try:
    model = EncoderClassifier.from_hparams(
        source=MODEL_ID,
        savedir=str(MODEL_DIR.resolve()),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )

    model.eval()
    print("\nSUCCESS: ECAPA-TDNN loaded successfully!")
    print("Device: CPU")
    print("Ready for voice enrollment.")

except Exception as error:
    print("\nMODEL LOADING FAILED")
    print(type(error).__name__, str(error))
    raise
