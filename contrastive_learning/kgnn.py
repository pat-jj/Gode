import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.nn import global_mean_pool
from torch.nn import functional as F


class KGNN(torch.nn.Module):
    def __init__(self, node_emb, rel_emb, num_nodes, num_rels, embedding_dim, hidden_dim, num_motifs,
                    lambda_edge=0.8, lambda_motif=1, lambda_mol_class=1, lambda_binary=1):
        super(KGNN, self).__init__()

        self.lambda_edge = lambda_edge
        self.lambda_motif = lambda_motif
        self.lambda_node_class = lambda_mol_class
        self.lambda_binary = lambda_binary


        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=False)

        if rel_emb is None:
            self.rel_emb = nn.Embedding(num_rels, embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=False)

        self.lin = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # GNN Layers
        self.conv1 = GINEConv(nn.Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(nn.Linear(hidden_dim, hidden_dim))

        # Task specific layers
        self.edge_class_layer = torch.nn.Linear(hidden_dim * 2, num_rels)
        self.motif_pred_layer = torch.nn.Linear(hidden_dim, num_motifs)
        self.node_class_layer = torch.nn.Linear(hidden_dim, 15)
        self.binary_pred_layer = torch.nn.Linear(hidden_dim, 1)


    def forward(self, node_ids, rel_ids, center_mol_idx, non_molecule_node_ids, edge_index, batch=None, output_emb=False):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_emb(node_ids).float()
        x = F.normalize(x, p=2, dim=1)
        edge_attr = self.rel_emb(rel_ids).float()
        edge_attr = F.normalize(edge_attr, p=2, dim=1)

        x = F.relu(self.lin(x))
        edge_attr = F.relu(self.lin(edge_attr))

        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)

        if output_emb:
            return x[center_mol_idx], global_mean_pool(x, batch=batch)  # return the embedding of the center molecule and the whole graph embedding

        # For edge prediction, we concatenate the embeddings of the two nodes for each edge
        edge_pred_input = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        edge_class = self.edge_class_layer(edge_pred_input)

        # For motif prediction, we apply mean pooling and use the result as input
        center_mol_embedding = x[center_mol_idx]
        motif_pred = self.motif_pred_layer(center_mol_embedding)

        # For non-molecule node multiclass classification
        if len(non_molecule_node_ids)>0:
            non_molecule_node_embedding = x[non_molecule_node_ids]
            node_class = self.node_class_layer(non_molecule_node_embedding)
        else:
            node_class = None

        # For binary classification
        binary_pred = self.binary_pred_layer(x)

        return edge_class, motif_pred, node_class, binary_pred


    def loss(self, edge_pred, motif_pred, node_class, binary_pred, edge_label, motif_label, node_label, binary_label):
        edge_loss = F.cross_entropy(edge_pred, edge_label)

        if node_class is not None:
            node_class_loss = F.cross_entropy(node_class, node_label)
        else:
            node_class_loss = 0

        motif_loss = F.binary_cross_entropy_with_logits(motif_pred, motif_label)
        binary_loss = F.binary_cross_entropy_with_logits(binary_pred.t(), binary_label)

        loss = self.lambda_edge * edge_loss + self.lambda_motif * motif_loss + self.lambda_node_class * node_class_loss + self.lambda_binary * binary_loss

        # Here, you can also include weights for each task if desired
        return loss, edge_loss, motif_loss, node_class_loss, binary_loss