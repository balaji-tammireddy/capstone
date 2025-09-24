import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

INPUT_PATH = "data/processed/stories_tokenized.json"
OUTPUT_PATH = "data/processed/stories_scenes.json"

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_SENTENCES_PER_SCENE = 3
SIMILARITY_THRESHOLD = 0.65
WINDOW_SIZE = 2  # sliding window for context

def embed_sentences(sentences, model):
    return model.encode(sentences)

def segment_story(sentences, model):
    embeddings = embed_sentences(sentences, model)
    n = len(sentences)
    if n <= MIN_SENTENCES_PER_SCENE:
        return [sentences]

    scenes = []
    current_scene = [sentences[0]]
    current_scene_embedding = embeddings[0].reshape(1, -1)

    for i in range(1, n):
        sim = cosine_similarity(current_scene_embedding, embeddings[i].reshape(1, -1))[0][0]

        if sim >= SIMILARITY_THRESHOLD:
            current_scene.append(sentences[i])
            current_scene_embedding = np.mean(
                np.vstack([current_scene_embedding, embeddings[i].reshape(1, -1)]), axis=0, keepdims=True
            )
        else:
            if len(current_scene) < MIN_SENTENCES_PER_SCENE and scenes:
                scenes[-1].extend(current_scene)
            else:
                scenes.append(current_scene)
            current_scene = [sentences[i]]
            current_scene_embedding = embeddings[i].reshape(1, -1)

    if current_scene:
        if len(current_scene) < MIN_SENTENCES_PER_SCENE and scenes:
            scenes[-1].extend(current_scene)
        else:
            scenes.append(current_scene)

    return scenes

def segment_all_stories():
    print("Loading SBERT model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Reading tokenized stories from {INPUT_PATH}...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output = []
    for story in tqdm(data, desc="Segmenting stories"):
        sentences = story["sentences"]
        scenes = segment_story(sentences, model)
        output.append({
            "id": story["id"],
            "title": story["title"],
            "scenes": scenes,
            "age_group": story.get("age_group"),
            "source": story.get("source")
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"Scene segmentation complete. {len(output)} stories saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    segment_all_stories()
