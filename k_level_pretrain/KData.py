import os
import math 
import numpy as np
import torch
import json
import pandas as pd
from tqdm import tqdm
import dgl
import warnings
warnings.filterwarnings("ignore")

class KData:

    def __init__(self, data_folder_path):
        if not os.path.exists(data_folder_path):
            os.mkdir(data_folder_path)
            
        self.data_folder = data_folder_path
        self.df = None
        self.idx_map = {}
        self.G = None
        self.df, self.df_train, self.df_valid, self.df_test = None, None, None, None
        self.seed = None
        self.prepare_data()


    def prepare_data(self, seed=42):
        kg_path = os.path.join(self.data_folder, 'KG_processed.csv')

        if not os.path.exists(kg_path):
            print('KG_processed.csv does not exist, processing...')
            self.process_kg()
        else:
            print('KG_processed.csv already exists, loading...')
            self.df = pd.read_csv(kg_path)

            # Group the data by 'relation' and 'y_type', and apply the categorization function
            # value_rows = self.df['y_type'] == 'value'
            # self.df.loc[value_rows, 'y_id'] = self.df[value_rows].groupby(['relation', 'y_type'])['y_id'].transform(categorize_values)
            self.idx_map = self.retrieve_id_mapping()

        if not os.path.exists(os.path.join(self.data_folder, 'train.csv')):
            # split data
            df_train, df_valid, df_test = create_split(self.df, self.data_folder, seed=seed)
        else:
            # load data
            df_train = pd.read_csv(os.path.join(self.data_folder, 'train.csv'))
            df_valid = pd.read_csv(os.path.join(self.data_folder, 'valid.csv'))
            df_test = pd.read_csv(os.path.join(self.data_folder, 'test.csv'))
        
        # create graph
        # g = self.create_graph(df_train, self.df)

        # self.G = g
        self.df_train, self.df_valid, self.df_test = df_train, df_valid, df_test
        self.seed = seed
        print('Data preparation finished.')


    def process_kg(self):
        df = pd.read_csv(os.path.join(self.data_folder, 'KG.csv'))
        df = df[['x_type', 'x_id', 'relation', 'y_type', 'y_id']]
        unique_relation = np.unique(df.relation.values)
        undirected_index = []

        print('Iterating over relations...')
        for i in tqdm(unique_relation):
            if ('_' in i) and (i.split('_')[0] == i.split('_')[1]):
                # homogeneous graph
                df_temp = df[df.relation == i]
                df_temp['check_string'] = df_temp.apply(lambda row: '_'.join(sorted([str(row['x_id']), str(row['y_id'])])), axis=1)
                undirected_index.append(df_temp.drop_duplicates('check_string').index.values.tolist())
            else:
                # undirected, 去重 (a->b 和 b->a 只记其中一个)
                d_off = df[df.relation == i]
                undirected_index.append(d_off[d_off.x_type == d_off.x_type.iloc[0]].index.values.tolist())

        flat_list = [item for sublist in undirected_index for item in sublist]
        df = df[df.index.isin(flat_list)]
        unique_node_types = np.unique(np.append(np.unique(df.x_type.values), np.unique(df.y_type.values)))


        print('Categorizing values...')
        df['value_level'] = df['y_id'] 
        value_rows = df['y_type'] == 'value'
        grouped_df = df[value_rows].groupby(['relation', 'y_type'])
        for name, group in tqdm(grouped_df):
            idx = group.index
            df.loc[idx, 'value_level'] = categorize_values(group, 'value_level')

        df['x_idx'] = np.nan
        df['y_idx'] = np.nan
        df['x_id'] = df.x_id.apply(lambda x: convert2str(x))
        df['y_id'] = df.y_id.apply(lambda x: convert2str(x))

        print('Iterating over node types...')
        for i in tqdm(unique_node_types):
            names = np.unique(np.append(df[df.x_type == i]['x_id'].values, df[df.y_type == i]['y_id'].values))
            names2idx = dict(zip(names, list(range(len(names)))))
            df.loc[df.x_type == i, 'x_idx'] = df[df.x_type == i]['x_id'].apply(lambda x: names2idx[x])
            df.loc[df.y_type == i, 'y_idx'] = df[df.y_type == i]['y_id'].apply(lambda x: names2idx[x])
            self.idx_map[i] = names2idx

        df.to_csv(os.path.join(self.data_folder, 'KG_processed.csv'), index=False)
        with open(os.path.join(self.data_folder, 'idx_map.json'), 'w') as f:
            json.dump(self.idx_map, f)

        self.df = df


    def create_graph(self, df_train, df):
        unique_graph = df_train[['x_type', 'relation', 'y_type']].drop_duplicates()
        DGL_input = {}
        for i in unique_graph.values:
            o = df_train[(df_train.x_type == i[0]) & (df_train.relation == i[1]) & (df_train.y_type == i[2])][['x_idx', 'y_idx']].values.T
            triple_type = tuple(i)
            if triple_type[1] == 'cooccurence' or triple_type[1] == 'rev_cooccurence':
                triple_type = (triple_type[0], triple_type[1] + '_' + triple_type[0] + '_' + triple_type[2], triple_type[2])
            DGL_input[triple_type] = (o[0].astype(int), o[1].astype(int))

        output = {k: len(v) for k, v in self.idx_map.items()}

        g = dgl.heterograph(DGL_input, num_nodes_dict={i: int(output[i])+1 for i in output.keys()})

        node_dict = {}
        edge_dict = {}
        for ntype in g.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in g.etypes:
            edge_dict[etype] = len(edge_dict)
            g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

        return g

    def retrieve_id_mapping(self):
        with open(os.path.join(self.data_folder, 'idx_map.json'), 'r') as f:
            idx_map = json.load(f)
        return idx_map


def categorize_values(df, column_name):
    series = df[column_name]
    quantiles = series.quantile([x/10 for x in range(1, 10)])
    categories = pd.cut(series, bins=pd.unique([-float('inf'), *quantiles, float('inf')]),
                        labels=range(1, len(pd.unique([-float('inf'), *quantiles, float('inf')]))),
                        duplicates='drop')
    return categories


def reverse_rel_generation(df, df_valid, unique_rel):
    
    for i in unique_rel.values:
        temp = df_valid[df_valid.relation == i[1]]
        temp = temp.rename(columns={"x_type": "y_type", 
                     "x_id": "y_id", 
                     "x_idx": "y_idx",
                     "y_type": "x_type", 
                     "y_id": "x_id", 
                     "y_idx": "x_idx"})

        if i[0] != i[2]:
            # bi identity
            temp["relation"] = 'rev_' + i[1]
        df_valid = df_valid.append(temp)
    return df_valid.reset_index(drop = True)
    

def convert2str(x):
    try:
        if '_' in str(x): 
            pass
        else:
            x = float(x)
    except:
        pass

    return str(x)

def create_split(df, data_folder, seed=42):
    df_train, df_valid, df_test = create_fold(df, frac=[0.83125, 0.11875, 0.05], seed=seed)
    unique_rel = df[['x_type', 'relation', 'y_type']].drop_duplicates()
    df_train = reverse_rel_generation(df, df_train, unique_rel)
    df_valid = reverse_rel_generation(df, df_valid, unique_rel)
    df_test = reverse_rel_generation(df, df_test, unique_rel)
    df_train.to_csv(os.path.join(data_folder, 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(data_folder, 'valid.csv'), index=False)
    df_test.to_csv(os.path.join(data_folder, 'test.csv'), index=False)
    return df_train, df_valid, df_test

def create_fold(df, frac=[0.7, 0.1, 0.2], seed=42):
    out = random_fold(df, frac, seed)
    out['test'] = out['valid']

    return out['train'], out['valid'], out['test']


def random_fold(df, frac, seed):
    train_frac, val_frac, test_frac = frac
    df_train = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_test = pd.DataFrame()
    # to avoid extreme minority types don't exist in valid/test
    for i in df.relation.unique():
        df_temp = df[df.relation == i]
        test = df_temp.sample(frac = test_frac, replace = False, random_state = seed)
        train_val = df_temp[~df_temp.index.isin(test.index)]
        val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
        train = train_val[~train_val.index.isin(val.index)]
        df_train = df_train.append(train)
        df_valid = df_valid.append(val)
        df_test = df_test.append(test)        
        
    return {'train': df_train.reset_index(drop = True), 
            'valid': df_valid.reset_index(drop = True), 
            'test': df_test.reset_index(drop = True)}