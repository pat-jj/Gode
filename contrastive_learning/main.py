import torch
from torch import nn
from joint_model import ContrastiveLearning
import json
import numpy as np
from contra_dataset import ContrastiveDataset, ContraCollator
from grover.data import MoleculeDatapoint
from tqdm import tqdm
from torch.utils.data import DataLoader
from mgnn import build_model, load_checkpoint
from kgnn import KGNN
import neptune
import os

NEPTUNE_KEY = os.environ['NEPTUNE_API_TOKEN']
KHOP = 2
KGE = True
HIDDEN_EMB = 1200
NEGA = 16
TEMP = 0.8

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


    # Entire Knowledge Graph (MolKG)
    print('Loading entire knowledge graph...')
    with open(f'{data_path}/graph.pt', 'rb') as f:
        G_tg = torch.load(f)

    with open('/home/pj20/gode/contrastive_learning/data.json', 'r') as f:
        data_list_of_tuples = json.load(f)

    mol_smiles = []
    sample_mol_ids = []
    pos_neg_labels = []

    for i in range(len(data_list_of_tuples)):
        mol_smiles.append(data_list_of_tuples[i][0])
        sample_mol_ids.append(data_list_of_tuples[i][1])
        pos_neg_labels.append(data_list_of_tuples[i][2])
    
    mol_datapoints = [
        MoleculeDatapoint(
            line=line,
        ) for i, line in tqdm(enumerate(mol_smiles), total=len(mol_smiles), disable=True)
    ]

    data = (mol_datapoints, sample_mol_ids, pos_neg_labels)

    return ent_type, motifs, G_tg, data


def get_dataloader(G_tg, motifs, ent_type, data, batch_size, args):
    args.no_cache = True
    dataset = ContrastiveDataset(
        G_tg=G_tg,
        motifs=motifs,
        ent_type=ent_type,
        data=data
        )
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size 
    # Shuffle the dataset
    dataset = torch.utils.data.Subset(dataset, np.random.permutation(len(dataset)))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    contra_collator = ContraCollator(shared_dict={}, args=args)

    train_loader = DataLoader(train_set, batch_size=batch_size, pin_memory=True, shuffle=True, drop_last=True, collate_fn=contra_collator)
    val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=True, collate_fn=contra_collator)

    return train_loader, val_loader


def infoNCE_loss(emb_1, emb_2, label, temperature=TEMP, pos_weight=NEGA):
    pos_weight = torch.tensor(pos_weight)
    # Compute the similarity between the embeddings
    sim = torch.exp((emb_1 * emb_2).sum(dim=-1) / temperature)
    sim = sim / (sim + 1)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Compute the binary cross-entropy loss
    loss = loss_fn(sim, label.float())

    return loss


def train(model, train_loader, val_loader, device, optimizer, run=None):
    model.train()
    training_loss = 0
    tot_loss = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, data in pbar:
        try:
            pbar.set_description(f'loss: {training_loss}')
            if len(data[1].edge_index[0]) == 0:
                continue
            elif len(data[1].edge_index[0]) > 100000:
                continue
            
            m_batch, k_batch, label = data[0], data[1].to(device), data[2].to(device)
            optimizer.zero_grad()
            
            # Forward
            m_b_emd, m_a_emb, k_emb = model(m_batch, k_batch)

            loss_1 = infoNCE_loss(m_b_emd, k_emb, label)
            loss_2 = infoNCE_loss(m_a_emb, k_emb, label)
            loss_tot = loss_1 + loss_2

            # Backward
            loss_tot.backward()
            training_loss = loss_tot
            tot_loss += loss_tot
            optimizer.step()
            if run != None:
                run["train/loss_m_b_emb"].append(loss_1)
                run["train/loss_m_a_emb"].append(loss_2)
                run["train/loss"].append(loss_tot)

            if i != 0 and i % 30000 == 0:
                validate(model, val_loader, device, run=run)
        except:
            training_loss = 0
            continue

    return tot_loss


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            if len(data[1].edge_index[0]) == 0 or len(data[1].edge_index[0]) > 100000:
                continue

            data = data.to(device)

            m_batch, k_batch, label = data[0], data[1], data[2]
            m_b_emd, m_a_emb, k_emb = model(m_batch, k_batch, label)
            loss_1 = infoNCE_loss(m_b_emd, k_emb, label)
            loss_2 = infoNCE_loss(m_a_emb, k_emb, label)
            total_loss += loss_1.item() + loss_2.item()

    avg_loss = total_loss / len(val_loader)

    if run != None:
        run["val/loss"].append(avg_loss)
    return avg_loss

def train_loop(model, train_loader, val_loader, optimizer, device, epochs, run=None, early_stop=5):
    best_eval_loss = 10000000
    for epoch in range(1, epochs+1):
        train_loss = train(model, train_loader, val_loader, device, optimizer, run=run)

        # Save Joint model
        torch.save(model.state_dict(), f'/data/pj20/molkg/contrastive_learning/joint_last_{KHOP}_hops_kge_{KGE}_{HIDDEN_EMB}.pkl')
        # Svae MGNN
        mgnn_state = {
            'args': model.model_mgnn.args,
            'state_dict': model.model_mgnn.state_dict(),
        }
        torch.save(mgnn_state, f'/data/pj20/molkg/contrastive_learning/mgnn_last_{KHOP}_hops_kge_{KGE}_{HIDDEN_EMB}.pt')
        # Save KGNN
        torch.save(model.model_kgnn, f'/data/pj20/molkg/contrastive_learning/kgnn_last_{KHOP}_hops_kge_{KGE}_{HIDDEN_EMB}.pkl')


        eval_loss = validate(model, val_loader, device, run=run)

        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), f'/data/pj20/molkg/joint_best_{KHOP}_hops_kge_{KGE}_1200.pkl')
            torch.save(mgnn_state, f'/data/pj20/molkg/contrastive_learning/mgnn_best_{KHOP}_hops_kge_{KGE}_{HIDDEN_EMB}.pt')
            torch.save(model.model_kgnn, f'/data/pj20/molkg/contrastive_learning/kgnn_best_{KHOP}_hops_kge_{KGE}_{HIDDEN_EMB}.pkl')

            print("best model saved")
            best_eval_loss = eval_loss
            early_stop_indicator = 0
        else:
            early_stop_indicator += 1
            if early_stop_indicator >= early_stop:
                break

def run():
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
        "TEMP": 0.8
    }

    run["parameters"] = params

    print("Preparing data...")
    data_path = '/data/pj20/molkg/pretrain_data'
    ent_type, motifs, G_tg, data = get_everything(data_path)

    print("Loading Pretrained MGNN (Grover) ...")
    mgnn_path = "/data/pj20/grover/pretrain/grover_large.pt"
    m_gnn = load_checkpoint(mgnn_path)
    m_gnn = m_gnn.cuda()

    print("Loading KGNN ...")
    k_gnn = KGNN(
        node_emb=None,
        rel_emb=None,
        num_nodes=ent_type.shape[0],
        num_rels=39,
        embedding_dim=1200,
        hidden_dim=1200,
        num_motifs=motifs.shape[1],
    )

    print("Loading Pre-trained KGNN ...")
    k_gnn.load_state_dict(torch.load(f'/data/pj20/molkg/kgnn_last_{KHOP}_hops_kge_{KGE}_{HIDDEN_EMB}.pkl', map_location='cuda:0'))
    k_gnn = k_gnn.cuda()

    print("Creating Joint Model")
    joint_model = ContrastiveLearning(model_mgnn=m_gnn, model_kgnn=k_gnn)

    print("Getting Dataloader ...")
    train_loader, val_loader = get_dataloader(G_tg=G_tg, motifs=motifs, ent_type=ent_type, data=data, batch_size=1, args=m_gnn.args)

    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    joint_model.to(device)
    print('Model:', joint_model)

    optimizer = torch.optim.Adam(joint_model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 200

    print('Start training !!!')
    train_loop(joint_model, train_loader, val_loader, optimizer, device, epochs, run=run)


if __name__ == '__main__':
    run()

