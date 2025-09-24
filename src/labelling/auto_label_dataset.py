import json
import os
import re
from collections import Counter

INPUT_PATH = "data/processed/stories_char_emotion.json"
OUTPUT_PATH = "data/processed/stories_labeled.json"

TP_EMOTIONS = {"fear", "surprise", "sadness", "anger"}
TP_KEYWORDS = {"but", "suddenly", "however", "tipped", "fell", "cried", "broke", "angry", "hurt"}

def is_turning_point(sentences, emotions):
    emo_match = bool(set(emotions) & TP_EMOTIONS)

    text = " ".join(sentences).lower()
    keyword_match = any(kw in text for kw in TP_KEYWORDS)

    return 1 if (emo_match or keyword_match) else 0

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
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    labeled_data = []
    for story in data:
        new_scenes = []
        for scene in story["scenes"]:
            sentences = scene.get("sentences", [])
            characters = scene.get("characters", [])
            emotions = scene.get("emotions", [])

            tp_label = is_turning_point(sentences, emotions)
            prominence = compute_prominence(sentences, characters, emotions)

            new_scenes.append({
                **scene,
                "tp_label": tp_label,
                "prominence": prominence
            })

        labeled_data.append({
            "id": story["id"],
            "title": story["title"],
            "scenes": new_scenes,
            "age_group": story.get("age_group"),
            "source": story.get("source")
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, indent=2)

    print(f"Labeled dataset saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
