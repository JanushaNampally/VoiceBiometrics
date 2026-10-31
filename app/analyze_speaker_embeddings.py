
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "research" / "speaker_embeddings.csv"
OUTPUT = ROOT / "research" / "impostor_similarity_scores.csv"

print("Loading speaker embeddings...")

df = pd.read_csv(INPUT)

print("Columns:", df.columns.tolist())
print("Total embedding records:", len(df))

if "speaker_id" not in df.columns:
    raise ValueError("speaker_id column not found.")

if "embedding" not in df.columns:
    raise ValueError(
        "embedding column not found. Check printed columns."
    )


def parse_embedding(value):
    if isinstance(value, str):
        try:
            vector = np.array(
                ast.literal_eval(value),
                dtype=np.float32
            )
        except (ValueError, SyntaxError):
            vector = np.fromstring(
                value.strip().strip("[]"),
                sep=",",
                dtype=np.float32
            )

            if vector.size == 0:
                vector = np.fromstring(
                    value.strip().strip("[]"),
                    sep=" ",
                    dtype=np.float32
                )
    else:
        vector = np.asarray(value, dtype=np.float32)

    vector = vector.flatten()

    if not np.all(np.isfinite(vector)):
        raise ValueError("Invalid values in embedding.")

    norm = np.linalg.norm(vector)

    if norm < 1e-10:
        raise ValueError("Zero-length embedding found.")

    return vector / norm


print("\nSTEP 1: Parsing embeddings...")

df["parsed_embedding"] = df["embedding"].apply(
    parse_embedding
)

print("Successfully parsed:", len(df))


print("\nSTEP 2: Checking unique speakers...")

speaker_embeddings = []

for speaker_id, group in df.groupby("speaker_id", sort=False):

    vectors = group["parsed_embedding"].tolist()

    reference = vectors[0]

    for vector in vectors[1:]:
        if not np.allclose(
            reference, vector, atol=1e-5, rtol=1e-5
        ):
            raise ValueError(
                f"Different embeddings found for speaker {speaker_id}. "
                "Investigate before proceeding."
            )

    speaker_embeddings.append({
        "speaker_id": speaker_id,
        "embedding": reference
    })

print("Unique speakers:", len(speaker_embeddings))


print("\nSTEP 3: Computing impostor similarities...")

scores = []

for a, b in combinations(speaker_embeddings, 2):

    similarity = float(
        np.dot(a["embedding"], b["embedding"])
    )

    scores.append({
        "speaker_1": a["speaker_id"],
        "speaker_2": b["speaker_id"],
        "cosine_similarity": similarity
    })


results = pd.DataFrame(scores)

results.to_csv(OUTPUT, index=False)


print("\n========== EXPERIMENT RESULTS ==========")

print("Unique speakers:", len(speaker_embeddings))
print("Embedding dimensions:",
      len(speaker_embeddings[0]["embedding"]))

print("Different-speaker comparisons:", len(results))

print("\nCosine Similarity Statistics:")

print(results["cosine_similarity"].describe())

print("\nHighest similarity comparisons:")

print(
    results.nlargest(10, "cosine_similarity").to_string(
        index=False
    )
)

print("\nLowest similarity comparisons:")

print(
    results.nsmallest(10, "cosine_similarity").to_string(
        index=False
    )
)

print("\nSaved results to:")
print(OUTPUT)

print("\nEXPERIMENT COMPLETED SUCCESSFULLY.")
