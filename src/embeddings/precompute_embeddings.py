import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import os

INPUT_PATH = "data/processed/stories_char_emotion.json"
OUTPUT_PATH = "data/processed/all_scenes_tensor.pt"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
MAX_LEN = 256

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

model = SentenceTransformer(MODEL_NAME)

all_input_ids = []
all_attention_mask = []
tp_labels = []        
prominence = []       

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

scene_count = 0
for story in tqdm(data, desc="Processing stories"):
    for scene in story["scenes"]:
        sentences = scene["sentences"]
        text = " ".join(sentences)

        emb = model.encode(text, convert_to_tensor=True) 

        all_input_ids.append(emb)
        all_attention_mask.append(torch.ones(emb.shape[0], dtype=torch.long))

        tp_labels.append(torch.tensor(scene.get("tp_label", 0), dtype=torch.long))
        if "prominence" in scene and scene["prominence"]:
            p = torch.tensor(list(scene["prominence"].values()), dtype=torch.float)
        else:
            p = torch.ones(len(scene.get("characters", [])), dtype=torch.float)
        prominence.append(p)

        scene_count += 1

print(f"✅ Total scenes processed: {scene_count}")

torch.save({
    "input_ids": all_input_ids,
    "attention_mask": all_attention_mask,
    "tp_labels": tp_labels,
    "prominence": prominence
}, OUTPUT_PATH)

print(f"✅ Embeddings saved to {OUTPUT_PATH}")
