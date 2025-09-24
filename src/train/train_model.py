import os, sys, torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch, Data
from torch.serialization import safe_globals

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.append(src_path)

from models.multi_task_model import MultiTaskModel

SEQ_DATA_PATH = "data/processed/all_scenes_tensor.pt"
GRAPH_DATA_PATH = "data/processed/stories_graph.pt"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading sequence data...")
seq_data = torch.load(SEQ_DATA_PATH)
input_ids = seq_data["input_ids"]
tp_labels = seq_data["tp_labels"]
prominence = seq_data["prominence"]

print("Loading graph data...")
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

with safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
    graph_data_list = torch.load(GRAPH_DATA_PATH)

valid_indices = [i for i in range(len(input_ids)) if i < len(graph_data_list)]
input_ids = [input_ids[i] for i in valid_indices]
tp_labels = [tp_labels[i] for i in valid_indices]
prominence = [prominence[i] for i in valid_indices]
graph_data_list = [graph_data_list[i] for i in valid_indices]

print(f"✅ Total valid scenes: {len(input_ids)}")

dataset = TensorDataset(torch.stack(input_ids), torch.tensor(tp_labels))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MultiTaskModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion_tp = torch.nn.CrossEntropyLoss()
criterion_prom = torch.nn.MSELoss()

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_tp_correct = 0
    total_samples = 0

    for batch_idx, (batch_seq, batch_tp_labels) in enumerate(dataloader):
        batch_seq = batch_seq.to(device)
        batch_tp_labels = batch_tp_labels.to(device)

        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + batch_seq.size(0)
        batch_graphs = graph_data_list[start_idx:end_idx]
        graph_batch = Batch.from_data_list(batch_graphs).to(device)

        optimizer.zero_grad()
        tp_logits, prom_pred = model(batch_seq, graph_batch)

        loss_tp = criterion_tp(tp_logits, batch_tp_labels)

        target_prom = torch.cat([prominence[i].to(device) for i in range(start_idx, end_idx)])
        prom_pred_flat = prom_pred.repeat_interleave([len(prominence[i]) for i in range(start_idx, end_idx)], dim=0)
        loss_prom = criterion_prom(prom_pred_flat.squeeze(), target_prom)

        loss = loss_tp + loss_prom
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_seq.size(0)
        total_tp_correct += (tp_logits.argmax(1) == batch_tp_labels).sum().item()
        total_samples += batch_seq.size(0)

    avg_loss = total_loss / total_samples
    tp_acc = total_tp_correct / total_samples
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | TP Acc: {tp_acc:.4f}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

print("✅ Training complete!")