import pandas as pd
import dgl
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

from dgl.data import DGLDataset

pathNet = './data/real_struction'
pathNode = '../modifyNode'
dataNet = '../dataset/net'
dataNodeNoW = '../dataset/node/WAT'


def feature_lipase(layer_data, graph):
    feature = {}
    indxs = {}
    texts = []
    for _type in layer_data:
        if len(layer_data[_type]) == 0:
            continue
        idxs = np.array(list(layer_data[_type].keys()))
        # if 'emb' in graph.node_feature[_type]:
        #     feature[_type] = np.array(list(graph.node_feature[_type].loc[idxs, 'emb']), dtype=np.float64)
        # else:
        #     feature[_type] = np.zeros([len(idxs), 400])
        tmp_feature = graph.node_feature[_type]
        feature[_type] = pd.DataFrame(tmp_feature)
        feature[_type].set_index('position', inplace=True)
        # 展平嵌套列表
        flat_data = feature[_type].loc[idxs, 'emb'].apply(lambda x: np.array(x).flatten())
        # 转换为 numpy 数组
        feature[_type] = np.vstack(flat_data.values).astype(np.float64)

        indxs[_type] = idxs

    return feature, indxs, texts


def load_graphpred_testDataset(dataset, bidirected=False, is_classification=False, is_CNN=False):
    data_path = dataset
    # graph_path = f'{data_path}/graphs_with_labels_test2_fold0_stand.bin'
    graph_path = f'{data_path}/plddt_graphs_with_labels_test2_fold0_stand_last.bin'
    # graph_path = f'{data_path}/Ablation_position_graphs_with_labels_test2_fold0.bin'
    graphs, graph_attr = dgl.load_graphs(graph_path)

    relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
    relation_to_index = {rel: idx for idx, rel in enumerate(relations)}

    # for g in graphs:
    #     # g.ndata['emb'] = g.ndata['emb'][:, :-20]  # 去除每个节点的最后20维特征
    #     g.ndata['emb'] = g.ndata['emb'][:, -20:]  # 只保留最后20维特征

    if bidirected:
        graphs = [
            dgl.to_bidirected(g, copy_ndata=True)
            for g in graphs
        ]

    test_graphs = []
    test_labels = []
    test_name_protein = []
    adjacency_matrices = []  # 存储每个图的[6, num_nodes, num_nodes]矩阵

    # 二值化标签：b大于等于60的设置为1，否则设置为0
    if is_classification:
        graph_attr['labels'] = (graph_attr['labels'] >= 60).int()

    for g, cv, lbl, nap in zip(graphs, graph_attr['cv_folds'], graph_attr['labels'], graph_attr['name_protein']):
        # 解码字符串
        # 解码字符串
        name_protein = "".join(map(chr, nap.numpy())).strip()
        if name_protein[-1] == '\x00':
            name_protein = name_protein[:-4]
        test_graphs.append(g)
        test_labels.append(lbl)
        test_name_protein.append(name_protein)

        if is_CNN:
            # 获取图中的节点数量
            num_nodes = g.num_nodes()

            # 用稀疏矩阵构建邻接矩阵
            indices_list = []
            values_list = []

            for etype in g.etypes:
                src, dst = g.edges(etype=etype)
                weights = g.edges[etype].data['weight']
                rel_idx = relation_to_index[etype]

                # 收集稀疏矩阵的索引和对应值
                rel_indices = torch.stack([torch.full_like(src, rel_idx), src, dst], dim=0)  # [3, num_edges]
                indices_list.append(rel_indices)
                values_list.append(weights)

            # 合并所有关系的索引和权重
            indices = torch.cat(indices_list, dim=1)  # [3, total_edges]
            values = torch.cat(values_list, dim=0).squeeze()  # [total_edges]

            # 创建稀疏张量
            adj_matrix = torch.sparse_coo_tensor(indices, values, size=(6, num_nodes, num_nodes))
            adjacency_matrices.append(adj_matrix)

            # if adj_matrix.is_sparse:
            #     adj_dense = adj_matrix.to_dense().sum(dim=0)  # 聚合关系层，得到单一稠密矩阵
            # else:
            #     adj_dense = adj_matrix.numpy() if isinstance(adj_matrix, torch.Tensor) else adj_matrix
            #
            # plt.figure(figsize=(10, 8))
            # plt.imshow(adj_dense, cmap='Reds', interpolation='nearest')
            # plt.colorbar(label='Edge Weight')
            # plt.xlabel('Node Index')
            # plt.ylabel('Node Index')
            # plt.show(block=True)

    # if is_classification:
    #     # 确保正负样本比例为1:1
    #     test_labels = torch.tensor(test_labels)
    #     num_positive = (test_labels == 1).sum().item()
    #     negative_indices = (test_labels == 0).nonzero(as_tuple=True)[0]
    #
    #     # 随机从负类中选择与正类数量相同的负类样本
    #     random_negative_indices = random.sample(list(negative_indices), num_positive)
    #     balanced_test_labels = torch.cat([test_labels[test_labels == 1], test_labels[random_negative_indices]])
    #
    #     # 重新排序以保持图和标签一一对应
    #     test_graphs = [test_graphs[i] for i in range(len(test_labels)) if test_labels[i] == 1] + [test_graphs[i]
    #                                                                                               for i in
    #                                                                                               random_negative_indices]
    #     test_name_protein = [test_name_protein[i] for i in range(len(test_labels)) if test_labels[i] == 1] + [
    #         test_name_protein[i] for i in random_negative_indices]
    #     # adjacency_matrices = [adjacency_matrices[i] for i in range(len(test_labels)) if test_labels[i] == 1] + [
    #     #     adjacency_matrices[i] for i in
    #     #     random_negative_indices]
    #     test_dataset = Graphdataset(dataset, test_graphs, balanced_test_labels.float(), test_name_protein,
    #                                 adjacency_matrices)
    # else:
    test_dataset = Graphdataset(dataset, test_graphs, torch.FloatTensor(test_labels), test_name_protein,
                                adjacency_matrices)

    return test_dataset


def load_graphpred_dataset(
        dataset,
        CV_FOLDS,
        cv_select=None,
        bidirected=False,
        is_classification=False,
        is_CNN=False):
    data_path = dataset
    graphs, graph_attr = [], []
    for fold in range(CV_FOLDS):
        graph_path = f'{data_path}/plddt_graphs_with_labels_train_fold{fold}_stand_last.bin'
        tmp1, tmp2 = dgl.load_graphs(graph_path)
        graphs.append(tmp1)
        graph_attr.append(tmp2)
        # for g in tmp1:
        #     # g.ndata['emb'] = g.ndata['emb'][:, :-20]  # 去除每个节点的最后20维特征
        #     g.ndata['emb'] = g.ndata['emb'][:, -20:]  # 只保留最后20维特征
        # if fold == 1:
        #     break

    # # 将图的标签乘以10
    # for fold in range(CV_FOLDS):
    #     graph_attr[fold]['labels'] *= 10

    relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
    feat_dim = graphs[0][0].ndata['emb'].size(-1)
    relation_to_index = {rel: idx for idx, rel in enumerate(relations)}

    if bidirected:
        graphs = [
            dgl.to_bidirected(g, copy_ndata=True)
            for g in graphs
        ]

    if cv_select is None:
        train_dataset = Graphdataset(dataset, graphs, graph_attr['labels'].float())
        test_dataset = None
    else:
        train_graphs, valid_graphs = [], []
        train_labels, valid_labels = [], []
        train_name_protein, valid_name_protein = [], []
        train_adj, valid_adj = [], []

        # 二值化标签：大于等于60的设置为1，否则设置为0
        if is_classification:
            for fold in range(CV_FOLDS):
                graph_attr[fold]['labels'] = (graph_attr[fold]['labels'] >= 60).int()

        for fold in range(CV_FOLDS):
            # if fold != cv_select:
            #     continue
            # # Extract labels (assumed to be Tm temperatures) and plot histogram
            # labels = graph_attr[0]['labels'].numpy() if isinstance(graph_attr[0]['labels'], torch.Tensor) else \
            #     graph_attr[0]['labels']
            #
            # plt.figure(figsize=(10, 6))
            # plt.hist(labels, bins=30, alpha=0.6, color='b', edgecolor='black')
            # plt.xlabel('Tm Temperature')
            # plt.ylabel('Frequency')
            # plt.title(f'Tm Temperature Distribution for Fold {fold}')
            # plt.grid(True)
            # plt.show()
            for g, cv, lbl, nap in zip(graphs[fold], graph_attr[fold]['cv_folds'], graph_attr[fold]['labels'],
                                       graph_attr[fold]['name_protein']):
                # 解码字符串
                name_protein = "".join(map(chr, nap.numpy())).strip()
                if name_protein[-1] == '\x00':
                    name_protein = name_protein[:-4]
                # plddt = plddt_data[plddt_data.iloc[:, 0] == name_protein].iloc[0, 1]
                # if plddt <= 70:
                #     continue
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

        graph_path = f'{data_path}/plddt_graphs_with_labels_test2_fold0_stand_last.bin'
        graphs, graph_attr = dgl.load_graphs(graph_path)
        # 二值化标签：b大于等于60的设置为1，否则设置为0
        if is_classification:
            graph_attr['labels'] = (graph_attr['labels'] >= 60).int()
        for g, cv, lbl, nap in zip(graphs, graph_attr['cv_folds'], graph_attr['labels'], graph_attr['name_protein']):
            # 解码字符串
            name_protein = "".join(map(chr, nap.numpy())).strip()
            if name_protein[-1] == '\x00':
                name_protein = name_protein[:-4]
            train_graphs.append(g)
            train_labels.append(lbl)
            train_name_protein.append(name_protein)

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


def Ablation_load_graphpred_testDataset(dataset, bidirected=False, is_classification=False):
    data_path = dataset  # Ablation_aaindex_graphs_with_labels_      graphs_with_labels_
    graph_path = f'{data_path}/Ablation_aaindex_graphs_with_labels_test2_fold0_stand.bin'
    graphs, graph_attr = dgl.load_graphs(graph_path)

    feat_dim = graphs[0].ndata['emb'].size(-1)
    relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']

    if bidirected:
        graphs = [
            dgl.to_bidirected(g, copy_ndata=True)
            for g in graphs
        ]

    test_graphs = []
    test_labels = []
    test_name_protein = []

    # 二值化标签：b大于等于60的设置为1，否则设置为0
    if is_classification == True:
        graph_attr['labels'] = (graph_attr['labels'] >= 60).int()

    for g, cv, lbl, nap in zip(graphs, graph_attr['cv_folds'], graph_attr['labels'], graph_attr['name_protein']):
        # 解码字符串
        name_protein = "".join(map(chr, nap.numpy())).strip()
        test_graphs.append(g)
        test_labels.append(lbl)
        test_name_protein.append(name_protein)

    test_dataset = Graphdataset(dataset, test_graphs, torch.FloatTensor(test_labels), test_name_protein)

    return test_dataset, feat_dim, relations


def Ablation_load_graphpred_dataset(dataset, CV_FOLDS, cv_select=None, bidirected=False, is_classification=False):
    data_path = dataset
    graphs, graph_attr = [], []
    for fold in range(CV_FOLDS):  # Ablation_aaindex_graphs_with_labels_      graphs_with_labels_
        graph_path = f'{data_path}/Ablation_aaindex&position_graphs_with_labels_train_fold{fold}.bin'
        tmp1, tmp2 = dgl.load_graphs(graph_path)
        graphs.append(tmp1)
        graph_attr.append(tmp2)

    relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
    feat_dim = graphs[0][0].ndata['emb'].size(-1)

    if bidirected:
        graphs = [
            dgl.to_bidirected(g, copy_ndata=True)
            for g in graphs
        ]

    if cv_select is None:
        train_dataset = Graphdataset(dataset, graphs, graph_attr['labels'].float())
        test_dataset = None
    else:
        train_graphs, valid_graphs = [], []
        train_labels, valid_labels = [], []
        train_name_protein, valid_name_protein = [], []

        # 二值化标签：大于等于60的设置为1，否则设置为0
        if is_classification == True:
            for fold in range(CV_FOLDS):
                graph_attr[fold]['labels'] = (graph_attr[fold]['labels'] >= 60).int()

        for fold in range(CV_FOLDS):
            for g, cv, lbl, nap in zip(graphs[fold], graph_attr[fold]['cv_folds'], graph_attr[fold]['labels'],
                                       graph_attr[fold]['name_protein']):
                # 解码字符串
                name_protein = "".join(map(chr, nap.numpy())).strip()
                if cv == cv_select:
                    valid_graphs.append(g)
                    valid_labels.append(lbl)
                    valid_name_protein.append(name_protein)
                else:
                    train_graphs.append(g)
                    train_labels.append(lbl)
                    train_name_protein.append(name_protein)

        train_dataset = Graphdataset(dataset, train_graphs, torch.FloatTensor(train_labels), train_name_protein)
        valid_dataset = Graphdataset(dataset, valid_graphs, torch.FloatTensor(valid_labels), valid_name_protein)

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
