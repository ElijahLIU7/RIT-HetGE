import pandas as pd
import dgl
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from dgl.data import DGLDataset


def load_testDataset(dataset, is_classification=False, is_CNN=False):
    data_path = dataset
    graph_path = f'{data_path}/Test_fold0.bin'
    graphs, graph_attr = dgl.load_graphs(graph_path)

    relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
    relation_to_index = {rel: idx for idx, rel in enumerate(relations)}

    test_graphs = []
    test_labels = []
    test_name_protein = []
    adjacency_matrices = []

    # Binarized labels: Set to 1 if b is greater than or equal to 60, and set to 0 otherwise.
    if is_classification:
        graph_attr['labels'] = (graph_attr['labels'] >= 60).int()

    for g, cv, lbl, nap in zip(graphs, graph_attr['cv_folds'], graph_attr['labels'], graph_attr['name_protein']):
        name_protein = "".join(map(chr, nap.numpy())).strip()
        if name_protein[-1] == '\x00':
            name_protein = name_protein[:-4]
        test_graphs.append(g)
        test_labels.append(lbl)
        test_name_protein.append(name_protein)

        if is_CNN:
            num_nodes = g.num_nodes()

            indices_list = []
            values_list = []

            for etype in g.etypes:
                src, dst = g.edges(etype=etype)
                weights = g.edges[etype].data['weight']
                rel_idx = relation_to_index[etype]

                rel_indices = torch.stack([torch.full_like(src, rel_idx), src, dst], dim=0)  # [3, num_edges]
                indices_list.append(rel_indices)
                values_list.append(weights)

            indices = torch.cat(indices_list, dim=1)
            values = torch.cat(values_list, dim=0).squeeze()

            adj_matrix = torch.sparse_coo_tensor(indices, values, size=(6, num_nodes, num_nodes))
            adjacency_matrices.append(adj_matrix)

    test_dataset = Graphdataset(dataset, test_graphs, torch.FloatTensor(test_labels), test_name_protein,
                                adjacency_matrices)

    return test_dataset


def load_dataset(
        dataset,
        CV_FOLDS,
        cv_select=0,
        is_classification=False,
        is_CNN=False):
    data_path = dataset
    graphs, graph_attr = [], []
    for fold in range(CV_FOLDS):
        graph_path = f'{data_path}/Train_fold{fold}.bin'
        tmp1, tmp2 = dgl.load_graphs(graph_path)
        graphs.append(tmp1)
        graph_attr.append(tmp2)

    relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
    feat_dim = graphs[0][0].ndata['emb'].size(-1)
    relation_to_index = {rel: idx for idx, rel in enumerate(relations)}

    train_graphs, valid_graphs = [], []
    train_labels, valid_labels = [], []
    train_name_protein, valid_name_protein = [], []
    train_adj, valid_adj = [], []

    # 二值化标签：大于等于60的设置为1，否则设置为0
    if is_classification:
        for fold in range(CV_FOLDS):
            graph_attr[fold]['labels'] = (graph_attr[fold]['labels'] >= 60).int()

    for fold in range(CV_FOLDS):

        for g, cv, lbl, nap in zip(graphs[fold], graph_attr[fold]['cv_folds'], graph_attr[fold]['labels'],
                                   graph_attr[fold]['name_protein']):
            # 解码字符串
            name_protein = "".join(map(chr, nap.numpy())).strip()
            if name_protein[-1] == '\x00':
                name_protein = name_protein[:-4]
            if cv == cv_select:
                valid_graphs.append(g)
                valid_labels.append(lbl)
                valid_name_protein.append(name_protein)
            else:
                train_graphs.append(g)
                train_labels.append(lbl)
                train_name_protein.append(name_protein)

            if is_CNN:
                num_nodes = g.num_nodes()

                indices_list = []
                values_list = []

                for etype in g.etypes:
                    src, dst = g.edges(etype=etype)
                    weights = g.edges[etype].data['weight']
                    rel_idx = relation_to_index[etype]

                    rel_indices = torch.stack([torch.full_like(src, rel_idx), src, dst], dim=0)
                    indices_list.append(rel_indices)
                    values_list.append(weights)
                indices = torch.cat(indices_list, dim=1)
                values = torch.cat(values_list, dim=0).squeeze()
                adj_matrix = torch.sparse_coo_tensor(indices, values, size=(6, num_nodes, num_nodes))

                if cv == cv_select:
                    valid_adj.append(adj_matrix)
                else:
                    train_adj.append(adj_matrix)

    if is_classification:
        # 确保训练集正负样本比例为1:1
        train_labels = torch.tensor(train_labels)
        num_positive_train = (train_labels == 1).sum().item()
        negative_indices_train = (train_labels == 0).nonzero(as_tuple=True)[0]

        # 随机从负类中选择与正类数量相同的负类样本
        random_negative_indices_train = random.sample(list(negative_indices_train), num_positive_train)
        balanced_train_labels = torch.cat(
            [train_labels[train_labels == 1], train_labels[random_negative_indices_train]])

        # 重新排序以保持图和标签一一对应
        train_graphs = [train_graphs[i] for i in range(len(train_labels)) if train_labels[i] == 1] + [
            train_graphs[i] for i in random_negative_indices_train]
        train_name_protein = [train_name_protein[i] for i in range(len(train_labels)) if train_labels[i] == 1] + [
            train_name_protein[i] for i in random_negative_indices_train]

        # 确保验证集正负样本比例为1:1
        valid_labels = torch.tensor(valid_labels)
        num_positive_valid = (valid_labels == 1).sum().item()
        negative_indices_valid = (valid_labels == 0).nonzero(as_tuple=True)[0]

        # 随机从负类中选择与正类数量相同的负类样本
        random_negative_indices_valid = random.sample(list(negative_indices_valid), num_positive_valid)
        balanced_valid_labels = torch.cat(
            [valid_labels[valid_labels == 1], valid_labels[random_negative_indices_valid]])

        # 重新排序以保持图和标签一一对应
        valid_graphs = [valid_graphs[i] for i in range(len(valid_labels)) if valid_labels[i] == 1] + [
            valid_graphs[i] for i in random_negative_indices_valid]
        valid_name_protein = [valid_name_protein[i] for i in range(len(valid_labels)) if valid_labels[i] == 1] + [
            valid_name_protein[i] for i in random_negative_indices_valid]

        # 训练集和验证集
        train_dataset = Graphdataset(dataset, train_graphs, balanced_train_labels.float(), train_name_protein,
                                     train_adj)
        valid_dataset = Graphdataset(dataset, valid_graphs, balanced_valid_labels.float(), valid_name_protein,
                                     valid_adj)
    else:
        train_dataset = Graphdataset(dataset, train_graphs, torch.FloatTensor(train_labels), train_name_protein,
                                     train_adj)
        valid_dataset = Graphdataset(dataset, valid_graphs, torch.FloatTensor(valid_labels), valid_name_protein,
                                     valid_adj)

    return train_dataset, valid_dataset, feat_dim, relations


class Graphdataset(DGLDataset):
    def __init__(self, name, graphs, labels, name_protein, adj):
        super(Graphdataset, self).__init__(name=name)
        self.graphs = graphs
        self.labels = labels
        self.name_protein = name_protein
        self.adj = adj

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.name_protein is not None:
            return self.graphs[idx], self.labels[idx], self.name_protein[idx]
        else:
            return self.graphs[idx], self.labels[idx]

    def get_graphs(self):
        return self.graphs

    def get_labels(self):
        return self.labels

    def process(self):
        pass

    def get_adg(self):
        return self.adj


def collate_graphs_with_labels(batch):
    graphs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.cat(labels)
    return batched_graph, batched_labels


# 可视化异构图
def plot_hetero_graph(g):
    # 将异构图转换为NetworkX图
    nx_g = dgl.to_networkx(g, node_attrs=['_ID'], edge_attrs=['_TYPE'])

    # 设置颜色和形状
    color_map = []
    shape_map = []
    for node in nx_g:
        if nx_g.nodes[node]['_TYPE'] == 'user':
            color_map.append('blue')
            shape_map.append('o')
        elif nx_g.nodes[node]['_TYPE'] == 'item':
            color_map.append('green')
            shape_map.append('s')

    # 绘制图
    pos = nx.spring_layout(nx_g)
    nx.draw(nx_g, pos, node_color=color_map, with_labels=True, node_shape='o', node_size=500)
    plt.show()
