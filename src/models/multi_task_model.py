import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import BertModel, BertConfig

class MultiTaskModel(nn.Module):
    def __init__(self, gnn_input_dim=384, gnn_hidden_dim=128, num_classes=2):
        super(MultiTaskModel, self).__init__()

        # -------------------------------
        # Sequence encoder (BERT)
        # -------------------------------
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.seq_hidden_dim = self.bert.config.hidden_size  # 768 for bert-base

        # -------------------------------
        # GNN for character-aware graph
        # -------------------------------
        self.gnn1 = GCNConv(gnn_input_dim, gnn_hidden_dim)
        self.gnn2 = GCNConv(gnn_hidden_dim, gnn_hidden_dim)

        # -------------------------------
        # Fusion
        # -------------------------------
        self.fc_fusion = nn.Linear(self.seq_hidden_dim + gnn_hidden_dim, 128)
        self.dropout = nn.Dropout(0.1)

        # -------------------------------
        # TP classification head
        # -------------------------------
        self.tp_classifier = nn.Linear(128, num_classes)

        # -------------------------------
        # Character prominence regression head
        # -------------------------------
        self.prominence_head = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, graph_batch):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        graph_batch: PyG Batch object
        """
        batch_size = input_ids.size(0)

        # -------------------------------
        # Sequence embedding
        # -------------------------------
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        seq_embed = bert_output.last_hidden_state[:, 0, :]  # [batch_size, 768]

        # -------------------------------
        # Graph embedding
        # -------------------------------
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = F.relu(self.gnn1(x, edge_index))
        x = F.relu(self.gnn2(x, edge_index))
        graph_embed = global_mean_pool(x, batch)  # [batch_size, gnn_hidden_dim]

        # -------------------------------
        # Fusion
        # -------------------------------
        combined = torch.cat([seq_embed, graph_embed], dim=1)  # [batch_size, seq+gnn]
        fused = F.relu(self.fc_fusion(combined))
        fused = self.dropout(fused)

        # -------------------------------
        # TP logits
        # -------------------------------
        tp_logits = self.tp_classifier(fused)

        # -------------------------------
        # Prominence prediction
        # -------------------------------
        prominence_pred = self.prominence_head(fused)  # [batch_size, 1]

        return tp_logits, prominence_pred