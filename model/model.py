
import os
import numpy as np
import pandas as pd
from rdkit import Chem
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self,
                 input_d=23,
                 dropout=0.5,
                 layer1=64,
                 layer2=32,
                 layer3=32,
                 Cuda=True):
        super(DNN, self).__init__()
        self.input_d = input_d
        self.dropout = dropout
        self.Cuda = Cuda
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.fc1 = nn.Linear(self.input_d, self.layer1)
        self.BN1 = nn.BatchNorm1d(self.layer1)
        self.dropout1 = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.layer1, self.layer2)
        self.BN2 = nn.BatchNorm1d(self.layer2)
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc3= nn.Linear(self.layer2,self.layer3)
        self.BN3= nn.BatchNorm1d(self.layer3)
        self.dropout3=nn.Dropout(self.dropout)
        self.fc4 = nn.Linear(self.layer3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.BN1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.BN2(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.BN3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

def normalize_adj(adj):#对边进行标准化
    degrees = np.sum(adj,axis=2)
    d = np.zeros((adj.shape[0],adj.shape[1],adj.shape[2]))
    for i in range(d.shape[0]):
     #   d[i,:,:] = np.diag(np.power(degrees[i,:],-0.5))
        d[i, :, :] = np.power(degrees[i, :], -0.5)
    adj_normal = d@adj@d
    adj_normal[np.isnan(adj_normal)] = 0
    return adj_normal

def onehot(idx, length):
    z = [0 for _ in range(length+1)]
    z[idx] = 1
    return z

def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    a = np.zeros([1, n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[0, tgt_idx - 1][(e_type - 1) * n_nodes + src_idx - 1] = 1
        a[0, src_idx - 1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] = 1
    return a

def EncodeSmile(smile):
    bond_dict = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, "AROMATIC": 3}
    possible_atom_types = ['C', 'N', 'O', 'S', 'c', 'n', 'o', 's', 'H', 'F', 'I', 'Cl', 'Br']  # ZINC
    max_atom = 50
    # remove stereo information, such as inward and outward edges
    # Chem.RemoveStereochemistry(mol)
    mol = Chem.MolFromSmiles(smile)
    mol = Chem.RWMol(mol)
    Chem.SanitizeMol(mol)
    for idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetSymbol() == '*':
            mol.RemoveAtom(idx)
            break
    edges = []
    nodes = []
    # encoding node
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in possible_atom_types:
            nodes.append(onehot(possible_atom_types.index(symbol), len(possible_atom_types)))
        else:
            nodes.append(onehot(len(possible_atom_types), len(possible_atom_types)))
    for i in range(max_atom - mol.GetNumAtoms()):
        nodes.append([0] * (len(possible_atom_types)+1))
    # encoding edge
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond_dict[str(bond.GetBondType())], bond.GetEndAtomIdx()))
        # assert bond_dict[str(bond.GetBondType())] != 3
    edges = create_adjacency_matrix(edges, max_atom, 4)
    total_atoms = mol.GetNumAtoms()
    return nodes, edges, total_atoms

def EncodeSmileBatch(smi_list):
    final_nodes = []
    final_edges = np.zeros([1, 50, 50 * 4 * 2])
    final_total_atoms = []

    for smi in smi_list:
        nodes, edges, total_atoms = EncodeSmile(smi)
        final_nodes.append(nodes)
        final_edges = np.vstack((final_edges, edges))
        final_total_atoms.append(total_atoms)

    node = np.array(final_nodes)
    adj = final_edges[1:]
    total_atoms = final_total_atoms
    return adj, node, total_atoms

class GCN(nn.Module):
    def __init__(self,
                 max_atom=50,
                 edge_dim=1,
                 atom_type_dim=14,
                 in_channels=14,
                 out_channels=32,
                 property_dim=180,
                 linear1=128,
                 linear2=256,
                 linear3=128,
                 linear4=10,
                 dropout=0.5,
                 mask_null=True,
                 device='cuda'
                 ):
        super(GCN, self).__init__()

        self.max_atom = max_atom
        self.edge_dim = edge_dim
        self.atom_type_dim = atom_type_dim
        self.in_channels = in_channels
        self.out_channels = self.emb_size = out_channels
        self.linear1 = linear1
        self.linear2 = linear2
        self.linear3 = linear3
        self.linear4 = linear4
        self.dropout = dropout
        self.property_dim = property_dim
        self.mask_null = mask_null
        self.device = device

        # emb layer
        self.Dense0 = nn.Linear(self.atom_type_dim, self.in_channels, bias=False)  # size(N,H,W,C)
        self.BN0 = nn.BatchNorm2d(self.in_channels)  # C from an expected input of size (N,C,H,W)
        self.Dense1 = nn.Linear(self.max_atom*4*2,self.max_atom)

        # GCN1
        v1 = torch.FloatTensor(1, self.edge_dim, self.in_channels, self.out_channels)
        nn.init.xavier_uniform_(v1)
        self.v1 = nn.Parameter(v1)
        self.BN1 = nn.BatchNorm2d(self.emb_size)  # C from an expected input of size (N,C,H,W)

        # GCN2
        v2 = torch.FloatTensor(1, self.edge_dim, self.out_channels, self.out_channels)
        nn.init.xavier_uniform_(v2)
        self.v2 = nn.Parameter(v2)
        self.BN2 = nn.BatchNorm2d(self.emb_size)  # C from an expected input of size (N,C,H,W)

        # GCN3
        v3 = torch.FloatTensor(1, self.edge_dim, self.out_channels, self.out_channels)
        nn.init.xavier_uniform_(v3)
        self.v3 = nn.Parameter(v3)
        self.BN3 = nn.BatchNorm2d(self.emb_size)  # C from an expected input of size (N,C,H,W)

        self.linear = nn.Sequential(
            nn.Linear(self.emb_size + self.property_dim, self.linear1),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear1),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear1, self.linear2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear2),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear2, self.linear3),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear3),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear3, self.linear4),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear4),
            nn.Dropout(self.dropout),
            # 最后一层不需要添加激活函数
            nn.Linear(self.linear4, 1)
        )

    def forward(self, adj, node, property_data):  # 输入state 即observation
        adj = np.sum(adj.reshape((adj.shape[0], 50, 50, 8)), axis=-1)
        adj = adj.reshape([adj.shape[0], 1, adj.shape[1], adj.shape[2]])
        '''
        for i in range(adj.shape[0]):
            adj[i] = normalize_adj(adj[i])
        '''
        adj = torch.tensor(adj).float().to(self.device)  # 必须是浮点数
        node = torch.tensor(node).float().to(self.device)

        # GCN层
        emb_node = self.GCN_Mutilayer(adj, node)
        emb_node = torch.sum(emb_node.squeeze(1), dim=1)  # B*f

        # concat
        emb_node = torch.cat((emb_node, property_data), dim=-1)

        # linear
        logit = self.linear(emb_node)
        return logit

    def GCN_Mutilayer(self, adj, node):
        # emb layer
      #  adj = self.Dense1(adj)
        ob_node = self.Dense0(node)
        ob_node = ob_node.repeat(1, 1, 1, 1).permute(1, 0, 2, 3)
        ob_node = ob_node.permute(0, 3, 1, 2)  # 把最后一维提到第二维
        ob_node = self.BN0(ob_node)
        ob_node = ob_node.permute(0, 2, 3, 1)  # 再把第二维放到最后一维

        # GCN1
        emb_node = self.GCN_batch(adj, ob_node, self.emb_size, self.v1,
                                  aggregate='mean')  # 维度变成（batch_size,1,max_atom,emb_size）
        emb_node = emb_node.permute(0, 3, 1, 2)  # 把最后一维提到第二维
        emb_node = self.BN1(emb_node)
        emb_node = emb_node.permute(0, 2, 3, 1)  # 再把第二维放到最后一维

        # GCN2
        emb_node = self.GCN_batch(adj, emb_node, self.emb_size, self.v2,
                                  aggregate='mean')  # 维度变成（batch_size,1,max_atom,emb_size）
        emb_node = emb_node.permute(0, 3, 1, 2)  # 把最后一维提到第二维
        emb_node = self.BN2(emb_node)
        emb_node = emb_node.permute(0, 2, 3, 1)  # 再把第二维放到最后一维

        # GCN3
        emb_node = self.GCN_batch(adj, emb_node, self.emb_size, self.v3, is_act=False,
                                  aggregate='mean')  # 维度变成（batch_size,1,max_atom,emb_size）
        emb_node = emb_node.permute(0, 3, 1, 2)  # 把最后一维提到第二维
        emb_node = self.BN3(emb_node)
        emb_node = emb_node.permute(0, 2, 3, 1)  # 再把第二维放到最后一维

        # 去掉维度为1的那一维数据
        emb_node = emb_node.squeeze(1)  # B*n*f
        return emb_node

    # gcn mean aggregation over edge features
    def GCN_batch(self, adj, node_feature, out_channels, weight, is_act=True, is_normalize=True, name='gcn_simple',
                  aggregate='sum'):
        '''
        state s: (adj,node_feature)
        :param adj: none*b*n*n
        :param node_feature: none*1*n*d
        :param out_channels: scalar
        :param name:
        :return:
        '''

        node_embedding = adj @ node_feature.repeat(1, self.edge_dim, 1, 1) @ weight.repeat(node_feature.size(0), 1, 1, 1)
        if is_act:
            node_embedding = F.relu(node_embedding)
        if aggregate == 'sum':
            node_embedding = torch.sum(node_embedding, dim=1, keepdim=True)  # mean pooling
        elif aggregate == 'mean':
            node_embedding = torch.mean(node_embedding, dim=1, keepdim=True)  # mean pooling
        elif aggregate == 'concat':
            node_embedding = torch.concat(torch.split(node_embedding, self.edge_dim, dim=1), dim=3)
        else:
            print('GCN aggregate error!')
        if is_normalize:
            node_embedding = F.normalize(node_embedding, p=2, dim=-1)  # l2正则化
        return node_embedding


class GCN_none(nn.Module):
    def __init__(self,
                 max_atom=50,
                 edge_dim=1,
                 atom_type_dim=14,
                 in_channels=14,
                 out_channels=32,
                 linear1=128,
                 linear2=256,
                 linear3=128,
                 linear4=10,
                 dropout=0.5,
                 mask_null=True,
                 device='cuda'
                 ):
        super(GCN_none, self).__init__()

        self.max_atom = max_atom
        self.edge_dim = edge_dim
        self.atom_type_dim = atom_type_dim
        self.in_channels = in_channels
        self.out_channels = self.emb_size = out_channels
        self.linear1 = linear1
        self.linear2 = linear2
        self.linear3 = linear3
        self.linear4 = linear4
        self.dropout = dropout
        self.mask_null = mask_null
        self.device = device

        # emb layer
        self.Dense0 = nn.Linear(self.atom_type_dim, self.in_channels, bias=False)  # size(N,H,W,C)
        self.BN0 = nn.BatchNorm2d(self.in_channels)  # C from an expected input of size (N,C,H,W)
        self.Dense1 = nn.Linear(self.max_atom*4*2,self.max_atom)

        # GCN1
        v1 = torch.FloatTensor(1, self.edge_dim, self.in_channels, self.out_channels)
        nn.init.xavier_uniform_(v1)
        self.v1 = nn.Parameter(v1)
        self.BN1 = nn.BatchNorm2d(self.emb_size)  # C from an expected input of size (N,C,H,W)

        # GCN2
        v2 = torch.FloatTensor(1, self.edge_dim, self.out_channels, self.out_channels)
        nn.init.xavier_uniform_(v2)
        self.v2 = nn.Parameter(v2)
        self.BN2 = nn.BatchNorm2d(self.emb_size)  # C from an expected input of size (N,C,H,W)

        # GCN3
        v3 = torch.FloatTensor(1, self.edge_dim, self.out_channels, self.out_channels)
        nn.init.xavier_uniform_(v3)
        self.v3 = nn.Parameter(v3)
        self.BN3 = nn.BatchNorm2d(self.emb_size)  # C from an expected input of size (N,C,H,W)

        self.linear = nn.Sequential(
            nn.Linear(self.emb_size, self.linear1),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear1),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear1, self.linear2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear2),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear2, self.linear3),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear3),
            nn.Dropout(self.dropout),
            nn.Linear(self.linear3, self.linear4),
            nn.ReLU(True),
            nn.BatchNorm1d(self.linear4),
            nn.Dropout(self.dropout),
            # 最后一层不需要添加激活函数
            nn.Linear(self.linear4, 1)
        )

    def forward(self, adj, node):  # 输入state 即observation
        adj = np.sum(adj.reshape((adj.shape[0], 50, 50, 8)), axis=-1)
        adj = adj.reshape([adj.shape[0], 1, adj.shape[1], adj.shape[2]])
        '''
        for i in range(adj.shape[0]):
            adj[i] = normalize_adj(adj[i])
        '''
        adj = torch.tensor(adj).float().to(self.device)  # 必须是浮点数
        node = torch.tensor(node).float().to(self.device)

        # GCN层
        emb_node = self.GCN_Mutilayer(adj, node)
        emb_node = torch.sum(emb_node.squeeze(1), dim=1)  # B*f

        # linear
        logit = self.linear(emb_node)
        return logit

    def GCN_Mutilayer(self, adj, node):
        # emb layer
      #  adj = self.Dense1(adj)
        ob_node = self.Dense0(node)
        ob_node = ob_node.repeat(1, 1, 1, 1).permute(1, 0, 2, 3)
        ob_node = ob_node.permute(0, 3, 1, 2)  # 把最后一维提到第二维
        ob_node = self.BN0(ob_node)
        ob_node = ob_node.permute(0, 2, 3, 1)  # 再把第二维放到最后一维

        # GCN1
        emb_node = self.GCN_batch(adj, ob_node, self.emb_size, self.v1,
                                  aggregate='mean')  # 维度变成（batch_size,1,max_atom,emb_size）
        emb_node = emb_node.permute(0, 3, 1, 2)  # 把最后一维提到第二维
        emb_node = self.BN1(emb_node)
        emb_node = emb_node.permute(0, 2, 3, 1)  # 再把第二维放到最后一维

        # GCN2
        emb_node = self.GCN_batch(adj, emb_node, self.emb_size, self.v2,
                                  aggregate='mean')  # 维度变成（batch_size,1,max_atom,emb_size）
        emb_node = emb_node.permute(0, 3, 1, 2)  # 把最后一维提到第二维
        emb_node = self.BN2(emb_node)
        emb_node = emb_node.permute(0, 2, 3, 1)  # 再把第二维放到最后一维

        # GCN3
        emb_node = self.GCN_batch(adj, emb_node, self.emb_size, self.v3, is_act=False,
                                  aggregate='mean')  # 维度变成（batch_size,1,max_atom,emb_size）
        emb_node = emb_node.permute(0, 3, 1, 2)  # 把最后一维提到第二维
        emb_node = self.BN3(emb_node)
        emb_node = emb_node.permute(0, 2, 3, 1)  # 再把第二维放到最后一维

        # 去掉维度为1的那一维数据
        emb_node = emb_node.squeeze(1)  # B*n*f
        return emb_node

    # gcn mean aggregation over edge features
    def GCN_batch(self, adj, node_feature, out_channels, weight, is_act=True, is_normalize=True, name='gcn_simple',
                  aggregate='sum'):
        '''
        state s: (adj,node_feature)
        :param adj: none*b*n*n
        :param node_feature: none*1*n*d
        :param out_channels: scalar
        :param name:
        :return:
        '''

        node_embedding = adj @ node_feature.repeat(1, self.edge_dim, 1, 1) @ weight.repeat(node_feature.size(0), 1, 1, 1)
        if is_act:
            node_embedding = F.relu(node_embedding)
        if aggregate == 'sum':
            node_embedding = torch.sum(node_embedding, dim=1, keepdim=True)  # mean pooling
        elif aggregate == 'mean':
            node_embedding = torch.mean(node_embedding, dim=1, keepdim=True)  # mean pooling
        elif aggregate == 'concat':
            node_embedding = torch.concat(torch.split(node_embedding, self.edge_dim, dim=1), dim=3)
        else:
            print('GCN aggregate error!')
        if is_normalize:
            node_embedding = F.normalize(node_embedding, p=2, dim=-1)  # l2正则化
        return node_embedding