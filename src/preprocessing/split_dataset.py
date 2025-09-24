import json
import os
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/processed/stories_scenes.json"
OUTPUT_DIR = "data/processed/splits/"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    train_data, temp_data = train_test_split(
        data,
        test_size=(1 - TRAIN_RATIO),
        random_state=42
    )

    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_size),
        random_state=42
    )

    with open(os.path.join(OUTPUT_DIR, "train.json"), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "val.json"), 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "test.json"), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)

    print(f"Dataset split complete:")
    print(f"Train: {len(train_data)} stories")
    print(f"Validation: {len(val_data)} stories")
    print(f"Test: {len(test_data)} stories")

if __name__ == "__main__":
    split_dataset()