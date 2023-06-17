import os
import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm

import torch_sparse
import numpy as np
import csv
from torch_geometric.utils import to_dense_adj
import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import Draw

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
s = []
'''
Datasets are from https://github.com/masashitsubaki/CPI_prediction
'''




'''
Molecular graphs generation
'''
VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25}


def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]


def atom_features(atom):
    s.append(atom.GetSymbol())

    encoding = one_of_k_encoding_unk(atom.GetSymbol(),
                                     ['O', 'I', 'P', 'Cl', 'N', 'F', 'C', 'S', 'Br', 'Unknown'])
    encoding += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
        atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other'])
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]

    return np.array(encoding)
def index2adj(edge_index):

    edge_index = edge_index.transpose(1,0).tolist()
    # 计算图中的节点数
    num_nodes = np.max(edge_index) + 1

    # 创建一个大小为 (num_nodes, num_nodes) 的零矩阵
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    # 遍历 edge_index 并将对应的元素设置为 1 或边的权重
    for e1, e2 in edge_index:
        adjacency_matrix[e1][e2] = 1  # 无权图
        adjacency_matrix[e2][e1] = 1  # 无向图
    return adjacency_matrix
def adj2index(edge_index):
    adj = to_dense_adj(edge_index)
    edge_index_square, _ = torch_sparse.spspmm(edge_index, None, edge_index, None, adj.shape[1], adj.shape[1],
                                               adj.shape[1], coalesced=True)
    edge_index_cube, _ = torch_sparse.spspmm(edge_index_square, None, edge_index, None, adj.shape[1], adj.shape[1],
                                             adj.shape[1], coalesced=True)
    edge_index_quad, _ = torch_sparse.spspmm(edge_index_cube, None, edge_index, None, adj.shape[1], adj.shape[1],
                                             adj.shape[1], coalesced=True)
    edge_index_quin, _ = torch_sparse.spspmm(edge_index_quad, None, edge_index, None, adj.shape[1], adj.shape[1],
                                             adj.shape[1], coalesced=True)
    A = index2adj(edge_index_quad) + index2adj(edge_index_quin)
    # A = index2adj(edge_index_cube)

    A = [[1. if x != 0. else 0. for x in row] for row in A]
    # 邻接矩阵A
    n = len(A)
    # 存储存在边的节点序号的大列表
    edge_list = []

    # 遍历邻接矩阵A，将存在边的节点序号添加到edge_list中
    for i in range(n):
        for j in range(n):
            if A[i][j] != 0:
                # 对每个边进行排序，然后判断是否已经在边列表中出现过
                edge = sorted([i, j])
                if edge not in edge_list and i != j:
                    edge_list.append(edge)
    g = nx.Graph(edge_list).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    return edge_index

def mol_to_graph(mol):
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / np.sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if len(edges) == 0:
        return features, [[0, 0]]

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    edge_index = torch.LongTensor(edge_index).transpose(1, 0)
    edge_index = adj2index(edge_index)
    return features, edge_index


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


class GNNDataset(InMemoryDataset):
    def __init__(self, root, types='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[1])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt','processed_data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def process_data(self, data_path, graph_dict):
        df = pd.read_csv(data_path)

        data_list = []
        delete_list = []
        for i, row in df.iterrows():
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence']
            label = row['affinity']

            if graph_dict.get(smi) == None:
                print("Unable to process: ", smi)
                delete_list.append(i)
                continue

            x, edge_index = graph_dict[smi]

            target = seqs2int(sequence)
            target_len = 1200
            if len(target) < target_len:
                target = np.pad(target, (0, target_len - len(target)))
            else:
                target = target[:target_len]

            data = DATA.Data(
                x=torch.FloatTensor(x),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([int(label)]),
                target=torch.LongTensor([target])
            )

            data_list.append(data)

        if len(delete_list) > 0:
            df = df.drop(delete_list, axis=0, inplace=False)
            df.to_csv(data_path, index=False)

        return data_list

    def process(self):
        df_train = pd.read_csv(self.raw_paths[0])
        df_test = pd.read_csv(self.raw_paths[1])
        df = pd.concat([df_train,df_test])
        smiles = df['compound_iso_smiles'].unique()

        graph_dict = dict()
        for smile in tqdm(smiles, total=len(smiles)):
            mol = Chem.MolFromSmiles(smile)
            if mol == None:
                print("Unable to process: ", smile)
                continue
            graph_dict[smile] = mol_to_graph(mol)

        train_list = self.process_data(self.raw_paths[0], graph_dict)
        test_list = self.process_data(self.raw_paths[1], graph_dict)

        if self.pre_filter is not None:
            train_list = [train for train in train_list if self.pre_filter(train)]
            test_list = [test for test in test_list if self.pre_filter(test)]

        if self.pre_transform is not None:
            train_list = [self.pre_transform(train) for train in train_list]
            test_list = [self.pre_transform(test) for test in test_list]

        print('Graph construction done. Saving to file.')

        # save preprocessed train data:
        data, slices = self.collate(train_list)
        torch.save((data, slices), self.processed_paths[0])


        # save preprocessed test data:
        data, slices = self.collate(test_list)
        torch.save((data, slices), self.processed_paths[1])


if __name__ == "__main__":

    GNNDataset(root='data/Kinase')
    print(set(s))



