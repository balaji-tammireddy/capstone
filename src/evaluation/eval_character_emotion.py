import json
from tqdm import tqdm
from nrclex import NRCLex

INPUT_PATH = "data/processed/stories_char_emotion.json"

def evaluate():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_scenes = 0
    scenes_with_characters = 0
    scenes_with_emotions = 0
    total_characters = 0
    jaccard_scores = []

    for story in tqdm(data, desc="Evaluating stories"):
        for scene in story["scenes"]:
            total_scenes += 1
            chars = scene.get("characters", [])
            emotions = scene.get("emotions", [])

            if chars:
                scenes_with_characters += 1
                total_characters += len(chars)
            if emotions:
                scenes_with_emotions += 1

            combined_text = ' '.join(scene.get("sentences", []))
            gold_emotions = set(NRCLex(combined_text).raw_emotion_scores.keys())
            pred_emotions = set(emotions)
            if gold_emotions:
                intersection = len(pred_emotions.intersection(gold_emotions))
                union = len(pred_emotions.union(gold_emotions))
                jaccard_scores.append(intersection / union if union > 0 else 0)

    print("📊 Evaluation Results:")
    print(f"Total scenes: {total_scenes}")
    print(f"Scenes with ≥1 character: {scenes_with_characters} ({scenes_with_characters/total_scenes:.2%})")
    print(f"Scenes with ≥1 emotion: {scenes_with_emotions} ({scenes_with_emotions/total_scenes:.2%})")
    print(f"Avg. characters per scene: {total_characters / total_scenes:.2f}")
    print(f"Avg. emotion Jaccard overlap: {sum(jaccard_scores)/len(jaccard_scores):.4f}" if jaccard_scores else "0.0")

if __name__ == "__main__":
    evaluate()