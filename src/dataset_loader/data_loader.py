import json
import os
import torch
from transformers import AutoTokenizer
from collections import Counter

LABEL_PATH = "data/processed/stories_labeled.json"
OUTPUT_PATH = "data/processed/all_scenes_tensor.pt"

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256

TP_EMOTIONS = {"fear", "surprise", "sadness", "anger"}

def compute_prominence(sentences, characters, emotions):
    """
    Prominence = frequency of mentions + emotional bonus
    """
    text = " ".join(sentences)
    counts = Counter()
    for char in characters:
        counts[char] = text.count(char)

    emo_bonus = 1 if (set(emotions) & TP_EMOTIONS) else 0
    scores = {c: cnt + emo_bonus for c, cnt in counts.items()}

    if not scores:
        return {}

    max_score = max(scores.values())
    return {c: round(v / max_score, 3) for c, v in scores.items()}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_input_ids = []
    all_attention_mask = []
    all_tp_labels = []
    all_characters = []
    all_prominence = []
    all_story_ids = []
    all_titles = []

    print(f"Processing {len(data)} stories...")

    for story in data:
        for scene in story["scenes"]:
            sentences = scene.get("sentences", [])
            characters = scene.get("characters", [])
            emotions = scene.get("emotions", [])
            tp_label = scene.get("tp_label", 0)

            prominence = scene.get("prominence", compute_prominence(sentences, characters, emotions))

            text = " ".join(sentences)
            encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )

            all_input_ids.append(encoding["input_ids"].squeeze(0))
            all_attention_mask.append(encoding["attention_mask"].squeeze(0))
            all_tp_labels.append(torch.tensor(tp_label, dtype=torch.long))
            all_characters.append(characters)
            prom_tensor = torch.tensor([prominence.get(c, 0.0) for c in characters], dtype=torch.float) if characters else torch.tensor([])
            all_prominence.append(prom_tensor)
            all_story_ids.append(story["id"])
            all_titles.append(story["title"])

    all_input_ids = torch.stack(all_input_ids)
    all_attention_mask = torch.stack(all_attention_mask)
    all_tp_labels = torch.stack(all_tp_labels)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "tp_labels": all_tp_labels,
        "characters": all_characters,
        "prominence": all_prominence,
        "story_ids": all_story_ids,
        "titles": all_titles
    }, OUTPUT_PATH)

    print(f"✅ All scenes processed and saved to {OUTPUT_PATH}")
    print(f"Total scenes: {len(all_input_ids)}")


if __name__ == "__main__":
    main()
