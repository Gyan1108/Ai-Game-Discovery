import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
DATA_PATH = BASE_DIR / "data" / "processed" / "games_clean.json"
INDEX_PATH = BASE_DIR / "data" / "processed" / "games.index"
META_PATH = BASE_DIR / "data" / "processed" / "games_meta.json"


MODEL_NAME = "all-MiniLM-L6-v2"


def build_document(game):
    parts = []

    parts.append(f"{game['title']} is a video game.")

    if game["genres"]:
        parts.append(f"It belongs to the genres {', '.join(game['genres'])}.")

    if game["developers"]:
        parts.append(f"It was developed by {', '.join(game['developers'])}.")

    if game["platforms"]:
        parts.append(f"It is available on {', '.join(game['platforms'])}.")

    if game["playtime"]:
        parts.append(f"The average playtime is about {int(game['playtime'])} hours.")

    if game["metacritic"]:
        parts.append(f"It has a Metacritic score of {game['metacritic']}.")

    return " ".join(parts)


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        games = json.load(f)

    documents = []
    metadata = []

    for game in games:
        text = build_document(game)
        documents.append(text)
        metadata.append(game)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(documents, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Built FAISS index with {len(documents)} games")
    print(f"Index saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
