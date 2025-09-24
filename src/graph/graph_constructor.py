import json
import os
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

INPUT_PATH = "data/processed/stories_char_emotion.json"
OUTPUT_PATH = "data/processed/stories_graph.pt"

MODEL_NAME = "all-MiniLM-L6-v2"

def get_character_embeddings(characters, scene_sentences, model):
    """
    Generate embeddings for each character by averaging SBERT embeddings of sentences 
    where the character appears.
    """
    char_embeddings = {}
    for char in characters:
        relevant_sentences = [s for s in scene_sentences if char in s]
        if relevant_sentences:
            emb = model.encode(relevant_sentences)
            char_embeddings[char] = np.mean(emb, axis=0)
        else:
            char_embeddings[char] = np.zeros(model.get_sentence_embedding_dimension())
    return char_embeddings

def construct_graph(story, model):
    """
    Construct character-aware graph for a story.
    Nodes = characters
    Edges = co-occurrence in scenes
    Node features = embeddings
    """
    all_chars = set()
    for scene in story["scenes"]:
        all_chars.update(scene["characters"])
    all_chars = list(all_chars)

    if not all_chars:
        return None

    # Node features
    node_features = []
    for char in all_chars:
        char_embs = []
        for scene in story["scenes"]:
            if char in scene["characters"]:
                char_embs.append(get_character_embeddings([char], scene["sentences"], model)[char])
        node_features.append(np.mean(char_embs, axis=0))
    x = torch.tensor(node_features, dtype=torch.float)

    # Edge index (co-occurrence in scenes)
    edges = set()
    for scene in story["scenes"]:
        chars_in_scene = scene["characters"]
        for i, c1 in enumerate(chars_in_scene):
            for j, c2 in enumerate(chars_in_scene):
                if i < j:
                    idx1, idx2 = all_chars.index(c1), all_chars.index(c2)
                    edges.add((idx1, idx2))
                    edges.add((idx2, idx1))
    if edges:
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.story_id = story["id"]
    return data

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("Loading SBERT model...")
    model = SentenceTransformer(MODEL_NAME)

    graphs = []
    for story in tqdm(data, desc="Constructing graphs"):
        g = construct_graph(story, model)
        if g is not None:
            graphs.append(g)

    torch.save(graphs, OUTPUT_PATH)
    print(f"Graph construction complete. {len(graphs)} stories saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
