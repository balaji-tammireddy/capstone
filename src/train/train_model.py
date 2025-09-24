# train_model.py
import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch.serialization import safe_globals

# -------------------------------
# Fix imports: add 'src' to path
# -------------------------------
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from models.multi_task_model import MultiTaskModel  # Multi-task model

# -------------------------------
# Paths
# -------------------------------
SEQ_DATA_PATH = "data/processed/all_scenes_tensor.pt"
GRAPH_DATA_PATH = "data/processed/stories_graph.pt"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------
# Hyperparameters
# -------------------------------
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5

# -------------------------------
# Load sequence data
# -------------------------------
print("Loading sequence data...")
seq_data = torch.load(SEQ_DATA_PATH)
input_ids = seq_data["input_ids"]
attention_mask = seq_data["attention_mask"]
tp_labels = seq_data["tp_labels"]
prominence = seq_data["prominence"]  # list of tensors per scene

# -------------------------------
# Load graph data safely
# -------------------------------
print("Loading graph data...")
with safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
    graph_data_list = torch.load(GRAPH_DATA_PATH)

# -------------------------------
# Filter sequence data to match valid graphs
# -------------------------------
if len(graph_data_list) != len(input_ids):
    print("Filtering sequence data to match valid graphs...")
    valid_indices = [i for i, g in enumerate(graph_data_list) if g is not None]
    graph_data_list = [graph_data_list[i] for i in valid_indices]
    input_ids = input_ids[valid_indices]
    attention_mask = attention_mask[valid_indices]
    tp_labels = tp_labels[valid_indices]
    prominence = [prominence[i] for i in valid_indices]

print(f"✅ Total valid scenes: {len(input_ids)}")
print(f"✅ Total graphs: {len(graph_data_list)}")

# -------------------------------
# Dataset and DataLoader
# -------------------------------
dataset = TensorDataset(input_ids, attention_mask, tp_labels)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Model, optimizer, loss
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskModel().to(device)

criterion_tp = torch.nn.CrossEntropyLoss()
criterion_prom = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# Training loop
# -------------------------------
print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_tp_correct = 0
    total_tp_samples = 0

    for batch_idx, (batch_input_ids, batch_attention_mask, batch_tp_labels) in enumerate(dataloader):
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_tp_labels = batch_tp_labels.to(device)

        # Prepare graph batch for this mini-batch
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + batch_input_ids.size(0)
        batch_graphs = graph_data_list[start_idx:end_idx]
        graph_batch = Batch.from_data_list(batch_graphs).to(device)

        # Forward pass
        tp_logits, prom_pred = model(batch_input_ids, batch_attention_mask, graph_batch)

        # TP loss
        loss_tp = criterion_tp(tp_logits, batch_tp_labels)

        # Prominence loss
        target_prom = []
        for p in prominence[start_idx:end_idx]:
            target_prom.append(p.to(device))
        if target_prom:
            target_prom = torch.cat(target_prom)
            lengths = torch.tensor([len(p) for p in prominence[start_idx:end_idx]], dtype=torch.long)
            prom_pred_flat = prom_pred.repeat_interleave(lengths, dim=0)
            # Ensure same shape for MSELoss
            loss_prom = criterion_prom(prom_pred_flat, target_prom.view(-1, 1))
        else:
            loss_prom = torch.tensor(0.0).to(device)

        # Total loss
        loss = loss_tp + loss_prom

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item() * batch_input_ids.size(0)
        preds = torch.argmax(tp_logits, dim=1)
        total_tp_correct += (preds == batch_tp_labels).sum().item()
        total_tp_samples += batch_input_ids.size(0)

    avg_loss = total_loss / len(dataset)
    tp_acc = total_tp_correct / total_tp_samples
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | TP Accuracy: {tp_acc:.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

print("Training complete!")
