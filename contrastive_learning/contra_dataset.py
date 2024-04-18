import random
from argparse import Namespace
from typing import Callable, List, Union, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data.dataset import Dataset
from torch_geometric.utils import to_undirected, k_hop_subgraph

from grover.data.molfeaturegenerator import get_features_generator
from grover.data.scaler import StandardScaler
from grover.data import MoleculeDatapoint
from grover.data.molgraph import mol2graph

KHOP = 2

def get_k_hop_nodes(node_index, num_hops, edge_index):
    # Convert the edge_index to undirected
    edge_index = to_undirected(edge_index)

    # Compute the k-hop subgraph
    node_idx_k_hop, _, _, _ = k_hop_subgraph(node_index, num_hops, edge_index, relabel_nodes=False, num_nodes=None, flow='source_to_target')

    return node_idx_k_hop


def get_subgraph(G_tg, center_molecule_id, motifs, ent_type):
    masked_node_ids = get_k_hop_nodes(int(center_molecule_id), KHOP-1, G_tg.edge_index)
    subgraph = G_tg.subgraph(masked_node_ids)
    motif_labels = motifs[center_molecule_id]  # (num_masked_nodes, motif_len)
    node_labels = ent_type[masked_node_ids]  # (num_masked_nodes, num_ent_type)

    # Get the one-hot encoding of the 'molecule' type
    molecule_encoding = node_labels[:, 0] 
    # Find where the molecule_encoding is not 1 (i.e., not a molecule)
    non_molecule_mask = molecule_encoding != 1
    # Get the relative indices of non-molecule nodes in masked_node_ids
    non_molecule_node_ids_relative = torch.where(non_molecule_mask)[0]
    # Get the labels for non_molecule_nodes

    
    subgraph.masked_node_ids = masked_node_ids
    subgraph.center_molecule_id = torch.where(masked_node_ids == center_molecule_id)[0][0]
    subgraph.motif_labels = motif_labels
    subgraph.non_molecule_node_ids = non_molecule_node_ids_relative

    return subgraph



class ContrastiveDataset(Dataset):

    def __init__(self, G_tg, motifs, ent_type, data):

        self.data = data[0]
        self.args = self.data[0].args if len(self.data) > 0 else None
        self.scaler = None

        self.motifs = motifs # (num_ent, motif_len)
        self.ent_type = ent_type # (num_ent, num_ent_type)
        self.G_tg = G_tg
        self.kg_mol_ids = data[1]
        self.labels = data[2]

    def compound_names(self) -> List[str]:

        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    def smiles(self) -> List[str]:

        return [d.smiles for d in self.data]

    def features(self) -> List[np.ndarray]:

        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    def targets(self) -> List[List[float]]:

        return [d.targets for d in self.data]

    def num_tasks(self) -> int:

        if self.args.dataset_type == 'multiclass':
            return int(max([i[0] for i in self.targets()])) + 1
        else:
            return self.data[0].num_tasks() if len(self.data) > 0 else None

    def features_size(self) -> int:
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None

    def shuffle(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler

    def set_targets(self, targets: List[List[float]]):
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):

        self.data.sort(key=key)

    def __len__(self) -> int:

        return len(self.data)

    def __getitem__(self, idx) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        return (
            self.data[idx], 
            get_subgraph(
                G_tg=self.G_tg,
                center_molecule_id=self.kg_mol_ids[idx],
                motifs=self.motifs,
                ent_type=self.ent_type
            ),
            self.labels[idx]
            )

class ContraCollator(object):
    def __init__(self, shared_dict, args=None):
        self.args = args
        self.shared_dict = shared_dict

    def __call__(self, batch):
        smiles_batch = [d[0].smiles for d in batch]
        subgraph_batch = [d[1] for d in batch][0]
        target_batch = torch.tensor([d[2] for d in batch])
        batch_mol_graph = mol2graph(smiles_batch, self.shared_dict, self.args)
        mol_batch = batch_mol_graph.get_components()

        return mol_batch, subgraph_batch, target_batch
