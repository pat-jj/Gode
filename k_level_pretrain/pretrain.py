import os
import json
import pickle
import networkx as nx
import numpy as np
from KGNN import KGNN
from tqdm import tqdm
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.loader import DataLoader
from networkx.algorithms.traversal.breadth_first_search import bfs_edges
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected, k_hop_subgraph
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, cohen_kappa_score
import torch.nn.functional as F
import torch
import neptune
import logging

# Neptune API token
NEPTUNE_KEY = os.environ['NEPTUNE_API_TOKEN']
KHOP = 2
DEVICE = 4
KGE = True

# Get everything we prepared
def get_everything(data_path):
    # Training Labels
    ## Load entity type labels
    print('Loading entity type labels...')
    ent_type = torch.tensor(np.load(f'{data_path}/ent_type_onehot.npy')) # (num_ent, num_ent_type)

    ## Load center molecule motifs
    print('Loading center molecule motifs...')
    motifs = []
    with open(f'{data_path}/id2motifs.json', 'r') as f:
        id2motifs = json.load(f)
    motif_len = len(id2motifs['0'])
    for i in range(len(ent_type)):
        if str(i) in id2motifs.keys():
            motifs.append(np.array(id2motifs[str(i)]))
        else:
            motifs.append(np.array([0] * motif_len))

    motifs = torch.tensor(np.array(motifs), dtype=torch.long) # (num_ent, motif_len)

    ## Center molecule ids
    center_molecule_ids = torch.tensor([int(key) for key in id2motifs.keys()])

    # Entire Knowledge Graph (MolKG)
    print('Loading entire knowledge graph...')
    with open(f'{data_path}/graph.pt', 'rb') as f:
        G_tg = torch.load(f)

    # molecule_mask
    print('Loading molecule mask...')
    molecule_mask = torch.tensor(ent_type[:,0][G_tg.edge_index[0]] == 1) # (num_edges,)

    return ent_type, motifs, G_tg, center_molecule_ids, molecule_mask


def load_kge_embeddings(emb_path):
    # Load KGE embeddings
    print('Loading KGE embeddings...')
    with open(f'{emb_path}/entity_embedding_1200.pkl', 'rb') as f:
        entity_embedding = pickle.load(f)
    with open(f'{emb_path}/relation_embedding_1200.pkl', 'rb') as f:
        relation_embedding = pickle.load(f)
    
    return entity_embedding.clone().detach(), relation_embedding.clone().detach()


def get_k_hop_nodes(node_index, num_hops, edge_index):
    # Convert the edge_index to undirected
    edge_index = to_undirected(edge_index)

    # Compute the k-hop subgraph
    node_idx_k_hop, _, _, _ = k_hop_subgraph(node_index, num_hops, edge_index, relabel_nodes=False, num_nodes=None, flow='source_to_target')

    return node_idx_k_hop

# Get subgraph
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
    non_molecule_node_labels = node_labels[non_molecule_mask, 1:] # removed the first column
    # Create binary labels: 1 for molecule, 0 for non-molecule
    binary_labels = molecule_encoding
    
    subgraph.masked_node_ids = masked_node_ids
    subgraph.center_molecule_id = torch.where(masked_node_ids == center_molecule_id)[0][0]
    subgraph.motif_labels = motif_labels
    subgraph.node_labels = node_labels
    subgraph.non_molecule_node_ids = non_molecule_node_ids_relative
    subgraph.non_molecule_node_labels = non_molecule_node_labels
    subgraph.binary_labels = binary_labels

    # debug
    # print('masked_node_ids', masked_node_ids)
    # print('center_molecule_id', center_molecule_id)
    # print('motif_labels', motif_labels)
    # print('node_labels', node_labels)
    # print('non_molecule_node_ids', non_molecule_node_ids_relative)
    # print('non_molecule_node_labels', non_molecule_node_labels)
    # print('binary_labels', binary_labels)

    return subgraph



class Dataset(torch.utils.data.Dataset):
    def __init__(self, G_tg, center_molecule_ids, molecule_mask, motifs, ent_type):
        self.G_tg = G_tg # torch_geometric.data.Data
        self.center_molecule_ids = center_molecule_ids # list of int
        self.molecule_mask = molecule_mask # (num_edges,)
        self.motifs = motifs # (num_ent, motif_len)
        self.ent_type = ent_type # (num_ent, num_ent_type)

    def __len__(self):
        return len(self.center_molecule_ids)
    def __getitem__(self, idx):
        return get_subgraph(
            G_tg=self.G_tg, 
            center_molecule_id=self.center_molecule_ids[idx], 
            motifs=self.motifs, 
            ent_type=self.ent_type
            )


def get_dataloader(G_tg, center_molecule_ids, molecule_mask, motifs, ent_type, batch_size):
    dataset = Dataset(G_tg, center_molecule_ids, molecule_mask, motifs, ent_type)
    train_size = int(0.98 * len(dataset))
    val_size = len(dataset) - train_size 
    # Shuffle the dataset
    dataset = torch.utils.data.Subset(dataset, np.random.permutation(len(dataset)))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True)

    return train_loader, val_loader


def train(model, train_loader, val_loader, device, optimizer, run=None):
    model.train()
    training_loss = 0
    tot_loss = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar:
        pbar.set_description(f'loss: {training_loss}')
        if len(data.edge_index[0]) == 0:
            continue
        elif len(data.edge_index[0]) > 100000:
            continue

        data = data.to(device)
        optimizer.zero_grad()

        # Forward
        edge_class, motif_pred, node_class, binary_pred = model(data.masked_node_ids, data.relation, data.center_molecule_id, data.non_molecule_node_ids, data.edge_index)

        motif_labels = data.motif_labels.float()
        binary_labels = data.binary_labels.float()

        motif_pred = motif_pred.view(data.motif_labels.shape).float()
        binary_pred = binary_pred.view(data.binary_labels.shape).float()

        # Loss
        loss, edge_loss, motif_loss, node_class_loss, binary_loss = model.loss(edge_class, motif_pred, node_class, binary_pred, data.rel_label, motif_labels, data.non_molecule_node_labels, binary_labels)

        # Backward
        loss.backward()
        training_loss = loss
        tot_loss += loss
        optimizer.step()
        run["train/step_loss"].append(training_loss)
        run["train/step_edge_loss"].append(edge_loss)
        run["train/step_motif_loss"].append(motif_loss)
        run["train/step_node_class_loss"].append(node_class_loss)
        run["train/step_binary_loss"].append(binary_loss)

        if i != 0 and i % 6000 == 0:
            validate(model, val_loader, device, run=run)

    return tot_loss


def validate(model, val_loader, device, run=None):
    model.eval()
    val_loss = 0
    tot_loss = 0
    y_prob_edge_all, y_prob_motif_all, y_prob_node_all, y_prob_binary_all, y_true_edge_all, y_true_motif_all, y_true_node_all, y_true_binary_all = [], [], [], [], [], [], [], []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for i, data in pbar:
        pbar.set_description(f'loss: {val_loss}')
        if len(data.edge_index[0]) == 0:
            continue
        elif len(data.edge_index[0]) > 100000:
            continue

        data = data.to(device)
        with torch.no_grad():   

            # Forward
            edge_class, motif_pred, node_class, binary_pred = model(data.masked_node_ids, data.relation, data.center_molecule_id, data.non_molecule_node_ids, data.edge_index)

            motif_labels = data.motif_labels.float()
            binary_labels = data.binary_labels.float()

            motif_pred = motif_pred.view(data.motif_labels.shape).float()
            binary_pred = binary_pred.view(data.binary_labels.shape).float()

            # Loss
            loss, edge_loss, motif_loss, node_class_loss, binary_loss = model.loss(edge_class, motif_pred, node_class, binary_pred, data.rel_label, motif_labels, data.non_molecule_node_labels, binary_labels)

            val_loss = loss
            tot_loss += loss
            
            y_prob_edge = F.softmax(edge_class, dim=-1)
            y_prob_motif = F.sigmoid(motif_pred)
            if node_class == None:
                y_prob_node = None
                print("found none!!")
            else:
                y_prob_node = F.softmax(node_class, dim=-1)
            y_prob_binary = F.sigmoid(binary_pred)

            # node_labels = data.node_labels.reshape(int(val_loader.batch_size), int(len(data.node_labels)/val_loader.batch_size)).float()
            # rel_label = data.rel_label.reshape(int(val_loader.batch_size), int(len(data.rel_label)/val_loader.batch_size)).float()

            y_true_edge, y_true_motifs, y_true_node, y_true_binary = data.rel_label.cpu(), motif_labels.cpu(), data.non_molecule_node_labels.cpu(), binary_labels.cpu()
            print(y_true_node)

            y_prob_edge_all.append(y_prob_edge.cpu())
            y_prob_motif_all.append(y_prob_motif.cpu())
            y_prob_node_all.append(y_prob_node.cpu() if y_prob_node != None else y_true_node)
            y_prob_binary_all.append(y_prob_binary.cpu())
            y_true_edge_all.append(y_true_edge)
            y_true_motif_all.append(y_true_motifs)
            y_true_node_all.append(y_true_node)
            y_true_binary_all.append(y_true_binary)

            run["val/step_loss"].append(val_loss)
            run["val/step_edge_loss"].append(edge_loss)
            run["val/step_motif_loss"].append(motif_loss)
            run["val/step_node_class_loss"].append(node_class_loss)
            run["val/step_binary_loss"].append(binary_loss)

    
    y_prob_edge_all = np.concatenate(y_prob_edge_all, axis=0)
    y_prob_motif_all = np.concatenate(y_prob_motif_all, axis=0)
    y_prob_node_all = np.concatenate(y_prob_node_all, axis=0)
    y_prob_binary_all = np.concatenate(y_prob_binary_all, axis=0)
    y_true_edge_all = np.concatenate(y_true_edge_all, axis=0)
    y_true_motif_all = np.concatenate(y_true_motif_all, axis=0)
    y_true_node_all = np.concatenate(y_true_node_all, axis=0)
    y_true_binary_all = np.concatenate(y_true_binary_all, axis=0)
    model.train()
    
    return tot_loss, y_prob_edge_all, y_prob_motif_all, y_prob_node_all, y_prob_binary_all, y_true_edge_all, y_true_motif_all, y_true_node_all, y_true_binary_all


def metric_calculation(y_prob_all, y_true_all, mode='multiclass'):
    if mode == "binary":
        y_pred_all = (y_prob_all >= 0.5).astype(int)
        try:
            val_pr_auc = average_precision_score(y_true_all, y_prob_all)
        except:
            val_pr_auc = 0
        try:
            val_roc_auc = roc_auc_score(y_true_all, y_prob_all)
        except:
            val_roc_auc = 0
        try:
            val_jaccard = jaccard_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        except:
            val_jaccard = 0
        try:
            val_acc = accuracy_score(y_true_all, y_pred_all)
        except:
            val_acc = 0
        try:
            val_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        except:
            val_f1 = 0
        try:
            val_precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        except:
            val_precision = 0
        try:
            val_recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=1)
        except:
            val_recall = 0

    if mode == "multilabel":
        y_pred_all = (y_prob_all >= 0.5).astype(int)
        try:
            val_pr_auc = average_precision_score(y_true_all, y_prob_all, average="samples")
        except:
            val_pr_auc = 0
        try:
            val_roc_auc = roc_auc_score(y_true_all, y_prob_all, average="samples")
        except:
            val_roc_auc = 0
        try:
            val_jaccard = jaccard_score(y_true_all, y_pred_all, average="samples", zero_division=1)
        except:
            val_jaccard = 0
        try:
            val_acc = accuracy_score(y_true_all, y_pred_all)
        except:
            val_acc = 0
        try:
            val_f1 = f1_score(y_true_all, y_pred_all, average="samples", zero_division=1)
        except:
            val_f1 = 0
        try:
            val_precision = precision_score(y_true_all, y_pred_all, average="samples", zero_division=1)
        except:
            val_precision = 0
        try:
            val_recall = recall_score(y_true_all, y_pred_all, average="samples", zero_division=1)
        except:
            val_recall = 0

    elif mode == "multiclass":
        y_pred_all = np.argmax(y_prob_all, axis=-1)
        y_true_all = np.argmax(y_true_all, axis=-1)

        val_pr_auc = 0
        try:
            val_roc_auc = roc_auc_score(y_true_all, y_prob_all, multi_class="ovr", average="weighted")
        except:
            val_roc_auc = 0
        try:
            val_jaccard = cohen_kappa_score(y_true_all, y_pred_all)
        except:
            val_jaccard = 0
        try:
            val_acc = accuracy_score(y_true_all, y_pred_all)
        except:
            val_acc = 0
        try:
            val_f1 = f1_score(y_true_all, y_pred_all, average="weighted")
        except:
            val_f1 = 0
        val_precision = 0
        val_recall = 0

    return val_pr_auc, val_roc_auc, val_jaccard, val_acc, val_f1, val_precision, val_recall


def detach_numpy(tensor):
    return torch.cat(tensor, dim=0).cpu().detach().numpy()


def train_loop(model, train_loader, val_loader, optimizer, device, epochs, logger=None, run=None, early_stop=5):
    best_pr_auc = 0
    best_f1 = 0
    early_stop_indicator = 0
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, val_loader, device, optimizer, run=run)
        torch.save(model.state_dict(), f'/data/pj20/molkg/kgnn_last_{KHOP}_hops_kge_{KGE}_1200.pkl')

        valid_loss, y_prob_edge_all, y_prob_motif_all, y_prob_node_all, y_prob_binary_all, y_true_edge_all, y_true_motif_all, y_true_node_all, y_true_binary_all = validate(model, val_loader, device, run=run)

        print("calculating metrics for edge prediction...")
        edge_val_pr_auc, edge_val_roc_auc, edge_val_jaccard, edge_val_acc, edge_val_f1, edge_val_precision, edge_val_recall = metric_calculation(y_prob_edge_all, y_true_edge_all, mode="multiclass")
        print("calculating metrics for motif prediction...")
        motif_val_pr_auc, motif_val_roc_auc, motif_val_jaccard, motif_val_acc, motif_val_f1, motif_val_precision, motif_val_recall = metric_calculation(y_prob_motif_all, y_true_motif_all, mode="multilabel")
        print("calculating metrics for node prediction...")
        node_val_pr_auc, node_val_roc_auc, node_val_jaccard, node_val_acc, node_val_f1, node_val_precision, node_val_recall = metric_calculation(y_prob_node_all, y_true_node_all, mode="multiclass")
        print("calculating metrics for binary prediction...")
        binary_val_pr_auc, binary_val_roc_auc, binary_val_jaccard, binary_val_acc, binary_val_f1, binary_val_precision, binary_val_recall = metric_calculation(y_prob_binary_all, y_true_binary_all, mode="binary")

        if motif_val_pr_auc >= best_pr_auc:
            torch.save(model.state_dict(), f'/data/pj20/molkg/kgnn_best_{KHOP}_hops_kge_{KGE}_1200.pkl')
            print("best model saved")
            best_pr_auc = motif_val_pr_auc
            early_stop_indicator = 0
            # best_f1 = val_f1
        else:
            early_stop_indicator += 1
            if early_stop_indicator >= early_stop:
                break

        if run is not None:
            run["train/epoch_loss"].append(train_loss)
            run["val/loss"].append(valid_loss)
            run["val/edge_pr_auc"].append(edge_val_pr_auc)
            run["val/edge_roc_auc"].append(edge_val_roc_auc)
            run["val/edge_acc"].append(edge_val_acc)
            run["val/edge/f1"].append(edge_val_f1)
            run["val/edge/precision"].append(edge_val_precision)
            run["val/edge/recall"].append(edge_val_recall)
            run["val/edge/jaccard"].append(edge_val_jaccard)
            run["val/motif_pr_auc"].append(motif_val_pr_auc)
            run["val/motif_roc_auc"].append(motif_val_roc_auc)
            run["val/motif_acc"].append(motif_val_acc)
            run["val/motif/f1"].append(motif_val_f1)
            run["val/motif/precision"].append(motif_val_precision)
            run["val/motif/recall"].append(motif_val_recall)
            run["val/motif/jaccard"].append(motif_val_jaccard)
            run["val/node_pr_auc"].append(node_val_pr_auc)
            run["val/node_roc_auc"].append(node_val_roc_auc)
            run["val/node_acc"].append(node_val_acc)
            run["val/node/f1"].append(node_val_f1)
            run["val/node/precision"].append(node_val_precision)
            run["val/node/recall"].append(node_val_recall)
            run["val/node/jaccard"].append(node_val_jaccard)
            run["val/binary_pr_auc"].append(binary_val_pr_auc)
            run["val/binary_roc_auc"].append(binary_val_roc_auc)
            run["val/binary_acc"].append(binary_val_acc)
            run["val/binary/f1"].append(binary_val_f1)
            run["val/binary/precision"].append(binary_val_precision)
            run["val/binary/recall"].append(binary_val_recall)
            run["val/binary/jaccard"].append(binary_val_jaccard)

            
            if logger is not None:
                logger.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}, Val ROC-AUC: {motif_val_roc_auc:.4f}, Val F1: {motif_val_f1:.4f}, Val Precision: {motif_val_precision:.4f}, Val Recall: {motif_val_recall:.4f}, Val Jaccard: {motif_val_jaccard:.4f}')


def get_logger(lr, hidden_dim, epochs, lambda_):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(f'./training_logs/lr_{lr}_dim_{hidden_dim}_epochs_{epochs}_lambda_{lambda_}_hop_{KHOP}_kge_{KGE}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def run():
    # your credentials
    run = neptune.init_run(
        project="patrick.jiang.cs/Gode",
        api_token=NEPTUNE_KEY,
    )  

    params = {
        "lr": 1e-4,
        "hidden_dim": 1200,
        "epochs": 100,
        "lambda": "08_15_10",
        "k-hop": KHOP,
        "kge": KGE,
    }
    logger = get_logger(lr=params['lr'], hidden_dim=params['hidden_dim'], epochs=params['epochs'], lambda_=params['lambda'])
    run["parameters"] = params


    # Data path
    data_path = '/data/pj20/molkg/pretrain_data'
    print('Getting everything prepared...')
    ent_type, motifs, G_tg, center_molecule_ids, molecule_mask = get_everything(data_path)

    # Get dataloader
    train_loader, val_loader = get_dataloader(G_tg, center_molecule_ids, molecule_mask, motifs, ent_type, batch_size=1)

    # Load KGE embeddings
    # return:
    # entity_embedding: (num_ent, emb_dim)
    # relation_embedding: (num_rel, emb_dim)
    emb_path = '/data/pj20/molkg_kge/transe'
    print('Loading KGE embeddings...')
    entity_embedding, relation_embedding = load_kge_embeddings(emb_path)
    # entity_embedding, relation_embedding = None, None

    # Initialize model
    print('Initializing model...')
    model = KGNN(
        node_emb=entity_embedding if KGE else None,
        rel_emb=relation_embedding if KGE else None,
        num_nodes=ent_type.shape[0],
        num_rels=39,
        embedding_dim=1200,
        hidden_dim=1200,
        num_motifs=motifs.shape[1],
        lambda_edge=1.5,
        lambda_motif=1.8,
        lambda_mol_class=1.5,
        lambda_binary=1,
    )

    # # Load the last saved model
    # model.load_state_dict(torch.load(f'/data/pj20/molkg/kgnn_last_{KHOP}_hops_kge_{KGE}.pkl'))

    # Train
    device = torch.device(f'cuda:{DEVICE}' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.to(device)
    print('Model:', model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 200

    print('Start training !!!')
    train_loop(model, train_loader, val_loader, optimizer, device, epochs, logger=logger, run=run)


if __name__ == '__main__':
    run()
