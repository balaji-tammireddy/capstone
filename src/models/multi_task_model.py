import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MultiTaskModel(nn.Module):
    def __init__(self, seq_emb_dim=384, gnn_hidden=128, tp_classes=2):
        super(MultiTaskModel, self).__init__()
        self.gnn1 = GCNConv(seq_emb_dim, gnn_hidden)
        self.gnn2 = GCNConv(gnn_hidden, gnn_hidden)

        self.fc_fusion = nn.Linear(seq_emb_dim + gnn_hidden, 128)

        self.tp_head = nn.Linear(128, tp_classes)
        self.prom_head = nn.Linear(128, 1)  

    def forward(self, seq_emb, graph_batch):
        x = torch.stack(seq_emb).to(graph_batch.x.device)  

        g_x = F.relu(self.gnn1(graph_batch.x, graph_batch.edge_index))
        g_x = F.relu(self.gnn2(g_x, graph_batch.edge_index))
        g_x = global_mean_pool(g_x, graph_batch.batch)

        fused = torch.cat([x, g_x], dim=1)
        fused = F.relu(self.fc_fusion(fused))

        tp_logits = self.tp_head(fused)
        prom_pred = self.prom_head(fused)

        return tp_logits, prom_pred