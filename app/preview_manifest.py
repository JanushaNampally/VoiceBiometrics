
from pathlib import Path
import pandas as pd

manifest_path = Path("research/dataset_manifest.csv")

df = pd.read_csv(manifest_path)

print("\nMANIFEST COLUMNS")
print("----------------")
print(df.columns.tolist())

print("\nFIRST 15 RECORDINGS")
print("-------------------")
print(df.head(15).to_string(index=False))

print("\nSPEAKER 1 RECORDINGS")
print("--------------------")
print(df[df["speaker_id"] == 1].to_string(index=False))

print("\nPASSWORD 1 RECORDINGS")
print("---------------------")
print(df[df["password_id"] == 1].head(10).to_string(index=False))
