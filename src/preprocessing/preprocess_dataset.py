import json
import os
import re
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')

RAW_PATH = "data/raw/stories.json"
OUTPUT_PATH = "data/processed/stories_tokenized.json"

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('“', '"').replace('”', '"').replace("’", "'")
    return text.strip()

def tokenize_story(text):
    """Split story into clean sentences."""
    cleaned = clean_text(text)
    return sent_tokenize(cleaned)

def preprocess_all():
    with open(RAW_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed = []
    for entry in data:
        tokenized = tokenize_story(entry["story"])
        processed.append({
            "id": entry["id"],
            "title": entry["title"],
            "sentences": tokenized,
            "age_group": entry.get("age_group"),
            "source": entry.get("source")
        })

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed, f, indent=2)

    print(f"Preprocessing complete. {len(processed)} stories saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess_all()