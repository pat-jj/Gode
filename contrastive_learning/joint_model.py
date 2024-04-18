import torch
from torch import nn


class ContrastiveLearning(nn.Module):
    def __init__(self, model_mgnn, model_kgnn, use_projector=False, hidden_emb=None, temperature=0.5):
        super(ContrastiveLearning, self).__init__()
        self.model_mgnn = model_mgnn
        self.model_kgnn = model_kgnn
        self.use_projector = use_projector
        if use_projector == True:
            self.projector_1 = nn.Linear(1200, hidden_emb)
            self.projector_2 = nn.Linear(200, hidden_emb)
        self.temperature = temperature


    def forward(self, m_batch, k_batch):
            # Process the batches with the models
            m_b_emd, m_a_emb = self.model_mgnn(m_batch)

            k_node_emb, k_graph_emb = self.model_kgnn(k_batch.masked_node_ids, k_batch.relation, k_batch.center_molecule_id, k_batch.non_molecule_node_ids, k_batch.edge_index, output_emb=True)

            if self.use_projector == True:
                m_b_emb = self.projector_1(m_b_emb)
                m_a_emb = self.projector_1(m_a_emb)
                k_node_emb = self.projector_2(k_node_emb)

            return m_b_emd, m_a_emb, k_node_emb