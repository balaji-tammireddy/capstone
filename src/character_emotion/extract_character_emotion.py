import json
import os
import re
from tqdm import tqdm
from nrclex import NRCLex
import spacy
from concurrent.futures import ProcessPoolExecutor, as_completed

INPUT_PATH = "data/processed/stories_scenes.json"
OUTPUT_PATH = "data/processed/stories_char_emotion.json"

nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Basic cleaning."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('“', '"').replace('”', '"').replace("’", "'")
    return text.strip()

def extract_characters(sentence):
    """Extract characters using spaCy NER."""
    doc = nlp(sentence)
    return [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]

def extract_emotions(scene_sentences):
    """Extract emotions using NRCLex for the whole scene at once."""
    combined_text = " ".join(scene_sentences)
    text_obj = NRCLex(combined_text)
    return [emotion for emotion, score in text_obj.raw_emotion_scores.items() if score > 0]

def process_story(story):
    """Process all scenes in a story."""
    output_scenes = []
    for scene in story["scenes"]:
        scene_chars = set()
        scene_emotions = set()
        for sentence in scene:
            sentence = clean_text(sentence)
            scene_chars.update(extract_characters(sentence))
        scene_emotions.update(extract_emotions(scene))
        output_scenes.append({
            "sentences": scene,
            "characters": list(scene_chars),
            "emotions": list(scene_emotions)
        })
    return {
        "id": story["id"],
        "title": story["title"],
        "scenes": output_scenes,
        "age_group": story.get("age_group"),
        "source": story.get("source")
    }

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_story, story) for story in data]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing stories"):
            output.append(future.result())

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"Character and emotion extraction complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()