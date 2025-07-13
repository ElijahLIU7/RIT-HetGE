import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import logging
from abc import ABCMeta
import os
import numpy as np
from protein_wang.pyHGT.utils_homogeneous import load_graphpred_dataset, load_graphpred_testDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class BaseModel(nn.Module, metaclass=ABCMeta):
    @classmethod
    def build_model_from_args(cls, args, hg):
        r"""
        Build the model instance from args and hg.

        So every subclass inheriting it should override the method.
        """
        raise NotImplementedError("Models must implement the build_model_from_args method")

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *args):
        r"""
        The model plays a role of encoder. So the forward will encoder original features into new features.

        Parameters
        -----------
        hg : dgl.DGlHeteroGraph
            the heterogeneous graph
        h_dict : dict[str, th.Tensor]
            the dict of heterogeneous feature

        Return
        -------
        out_dic : dict[str, th.Tensor]
            A dict of encoded feature. In general, it should ouput all nodes embedding.
            It is allowed that just output the embedding of target nodes which are participated in loss calculation.
        """
        raise NotImplementedError

    def extra_loss(self):
        r"""
        Some model want to use L2Norm which is not applied all parameters.

        Returns
        -------
        th.Tensor
        """
        raise NotImplementedError

    def h2dict(self, h, hdict):
        pre = 0
        out_dict = {}
        for i, value in hdict.items():
            out_dict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return out_dict

    def get_emb(self):
        r"""
        Return the embedding of a model for further analysis.

        Returns
        -------
        numpy.array
        """
        raise NotImplementedError


def transform_relation_graph_list(hg, category, identity=True):
    r"""
        extract subgraph :math:`G_i` from :math:`G` in which
        only edges whose type :math:`R_i` belongs to :math:`\mathcal{R}`

        Parameters
        ----------
            hg : dgl.heterograph
                Input heterogeneous graph
            category : string
                Type of predicted nodes.
            identity : bool
                If True, the identity matrix will be added to relation matrix set.
    """

    # get target category id
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i
    g = dgl.to_homogeneous(hg, ndata='h')
    # find out the target node ids in g
    loc = (g.ndata[dgl.NTYPE] == category_id).to('cpu')
    category_idx = torch.arange(g.num_nodes())[loc]

    edges = g.edges()
    etype = g.edata[dgl.ETYPE]
    ctx = g.device
    # g.edata['w'] = th.ones(g.num_edges(), device=ctx)
    num_edge_type = torch.max(etype).item()

    # norm = EdgeWeightNorm(norm='right')
    # edata = norm(g.add_self_loop(), th.ones(g.num_edges() + g.num_nodes(), device=ctx))
    graph_list = []
    for i in range(num_edge_type + 1):
        e_ids = torch.nonzero(etype == i).squeeze(-1)
        sg = dgl.graph((edges[0][e_ids], edges[1][e_ids]), num_nodes=g.num_nodes())
        # sg.edata['w'] = edata[e_ids]
        sg.edata['w'] = torch.ones(sg.num_edges(), device=ctx)
        graph_list.append(sg)
    if identity == True:
        x = torch.arange(0, g.num_nodes(), device=ctx)
        sg = dgl.graph((x, x))
        # sg.edata['w'] = edata[g.num_edges():]
        sg.edata['w'] = torch.ones(g.num_nodes(), device=ctx)
        graph_list.append(sg)
    return graph_list, g.ndata['h'], category_idx



# GTN Layer (handling different edge types for heterographs)
class GTN(BaseModel):
    r"""
        GTN from paper `Graph Transformer Networks <https://arxiv.org/abs/1911.06455>`__
        in NeurIPS_2019. You can also see the extension paper `Graph Transformer
        Networks: Learning Meta-path Graphs to Improve GNNs <https://arxiv.org/abs/2106.06218.pdf>`__.

        `Code from author <https://github.com/seongjunyun/Graph_Transformer_Networks>`__.

        Given a heterogeneous graph :math:`G` and its edge relation type set :math:`\mathcal{R}`.Then we extract
        the single relation adjacency matrix list. In that, we can generate combination adjacency matrix by conv
        the single relation adjacency matrix list. We can generate :math:'l-length' meta-path adjacency matrix
        by multiplying combination adjacency matrix. Then we can generate node representation using a GCN layer.

        Parameters
        ----------
        num_edge_type : int
            Number of relations.
        num_channels : int
            Number of conv channels.
        in_dim : int
            The dimension of input feature.
        hidden_dim : int
            The dimension of hidden layer.
        num_class : int
            Number of classification type.
        num_layers : int
            Length of hybrid metapath.
        category : string
            Type of predicted nodes.
        norm : bool
            If True, the adjacency matrix will be normalized.
        identity : bool
            If True, the identity matrix will be added to relation matrix set.

    """
    def __init__(self, num_edge_type, num_channels, in_dim, hidden_dim, num_class, num_layers, category, norm,
                 identity):
        super(GTN, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        self.category = category
        self.identity = identity

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(GTLayer(num_edge_type, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge_type, num_channels, first=False))
        self.layers = nn.ModuleList(layers)
        self.gcn = GraphConv(in_feats=self.in_dim, out_feats=hidden_dim, norm='none', activation=F.relu)
        self.norm = EdgeWeightNorm(norm='right')
        self.linear1 = nn.Linear(self.hidden_dim * self.num_channels, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.num_class)
        self.category_idx = None
        self.A = None
        self.h = None

    def normalization(self, H):
        norm_H = []
        for i in range(self.num_channels):
            g = H[i]
            g = dgl.remove_self_loop(g)
            g.edata['w_sum'] = self.norm(g, g.edata['w_sum'])
            norm_H.append(g)
        return norm_H

    def forward(self, hg, h):
        with hg.local_scope():
            hg.ndata['h'] = h
            # * =============== Extract edges in original graph ================
            if self.category_idx is None:
                self.A, h, self.category_idx = transform_relation_graph_list(hg, category=self.category,
                                                                             identity=self.identity)
            else:
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
            # X_ = self.gcn(g, self.h)
            A = self.A
            # * =============== Get new graph structure ================
            for i in range(self.num_layers):
                if i == 0:
                    H, W = self.layers[i](A)
                else:
                    H, W = self.layers[i](A, H)
                if self.is_norm == True:
                    H = self.normalization(H)
                # Ws.append(W)
            # * =============== GCN Encoder ================
            for i in range(self.num_channels):
                g = dgl.remove_self_loop(H[i])
                edge_weight = g.edata['w_sum']
                g = dgl.add_self_loop(g)
                edge_weight = torch.cat((edge_weight, torch.full((g.number_of_nodes(),), 1, device=g.device)))
                edge_weight = self.norm(g, edge_weight)
                if i == 0:
                    X_ = self.gcn(g, h, edge_weight=edge_weight)
                else:
                    X_ = torch.cat((X_, self.gcn(g, h, edge_weight=edge_weight)), dim=1)
            X_ = self.linear1(X_)
            X_ = F.relu(X_)
            y = self.linear2(X_)
            return {self.category: y[self.category_idx]}


class GTLayer(nn.Module):
    r"""
        CTLayer multiply each combination adjacency matrix :math:`l` times to a :math:`l-length`
        meta-paths adjacency matrix.

        The method to generate :math:`l-length` meta-path adjacency matrix can be described as:

        .. math::
            A_{(l)}=\Pi_{i=1}^{l} A_{i}

        where :math:`A_{i}` is the combination adjacency matrix generated by GT conv.

        Parameters
        ----------
            in_channels: int
                The input dimension of GTConv which is numerically equal to the number of relations.
            out_channels: int
                The input dimension of GTConv which is numerically equal to the number of channel in GTN.
            first: bool
                If true, the first combination adjacency matrix multiply the combination adjacency matrix.

    """
    def __init__(self, in_channels, out_channels, first=True):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)

    def forward(self, A, H_=None):
        if self.first:
            result_A = self.conv1(A)
            result_B = self.conv2(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            result_A = H_
            result_B = self.conv1(A)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        H = []
        for i in range(len(result_A)):
            g = dgl.adj_product_graph(result_A[i], result_B[i], 'w_sum')
            H.append(g)
        return H, W


class GTConv(nn.Module):
    r"""
        We conv each sub adjacency matrix :math:`A_{R_{i}}` to a combination adjacency matrix :math:`A_{1}`:

        .. math::
            A_{1} = conv\left(A ; W_{c}\right)=\sum_{R_{i} \in R} w_{R_{i}} A_{R_{i}}

        where :math:`R_i \subseteq \mathcal{R}` and :math:`W_{c}` is the weight of each relation matrix
    """

    def __init__(self, in_channels, out_channels, softmax_flag=True):
        super(GTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.softmax_flag = softmax_flag
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, A):
        if self.softmax_flag:
            Filter = F.softmax(self.weight, dim=1)
        else:
            Filter = self.weight
        num_channels = Filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, g in enumerate(A):
                A[j].edata['w_sum'] = g.edata['w'] * Filter[i][j]
            sum_g = dgl.adj_sum_graph(A, 'w_sum')
            results.append(sum_g)
        return results


# Training function for DGL HeteroGraph
def train_gtn_model(train_dataset, valid_dataset, feat_dim, hidden_feats, num_classes, num_edge_types, num_channels,
                    num_layers, gcn_out_channels, epochs=50, lr=0.1, batch_size=1, fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GTN(num_edge_types=num_edge_types, num_channels=num_channels, num_layers=num_layers,
                out_feats=num_classes, num_nodes=feat_dim, gcn_out_channels=gcn_out_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.HuberLoss(reduction='mean', delta=1.0)

    # 数据加载器
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_model_path = f'best_model_fold_{fold}.pt' if fold is not None else 'best_model.pt'

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            graphs, labels, _ = batch
            graphs = graphs.to(device)
            labels = labels.float().to(device)
            features = graphs.ndata['emb'].to(device)  # 使用节点特征 'emb'

            # 获取异构图中每种关系的邻接矩阵
            A_list = []
            for etype in graphs.canonical_etypes:
                A = graphs.adj(etype=etype).to(device).to_dense()  # 提取特定关系类型的邻接矩阵
                A_list.append(A)

            logits = model(A_list, features)

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证模型
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in valid_loader:
                graphs, labels = batch[0].to(device), batch[1].to(device)
                features = graphs.ndata['emb'].to(device)

                A_list = []
                for etype in graphs.canonical_etypes:
                    A = graphs.adj(etype=etype).to(device).to_dense()
                    A_list.append(A)

                logits = model(A_list, features)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(logits.cpu().numpy())

        # 将预测值和实际值转换为 NumPy 数组
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # 计算验证集上的损失
        val_loss = np.mean((all_labels - all_preds) ** 2)

        # 计算各项指标
        mse = mean_squared_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        pcc = np.corrcoef(all_labels, all_preds)[0, 1]

        # 保存验证集上表现最好的模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Validation MSE: {mse:.4f}, '
              f'R²: {r2:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}')

    print("Training completed.")
    return model, best_loss, mse, r2, mae, pcc, best_model_path


def test_best_model(test_dataset, feat_dim, hidden_feats, num_classes, model_path, batch_size=8, relations=None):
    """使用测试集评估最优模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GTN(in_feats=feat_dim, hidden_feats=hidden_feats, out_feats=num_classes, num_heads=8,
                relations=relations).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            graphs, labels = batch[0].to(device), batch[1].to(device)
            logits = model(graphs)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.cpu().numpy())

    # 将预测值和实际值转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 计算 MSE
    mse = mean_squared_error(all_labels, all_preds)

    # 计算 R²
    r2 = r2_score(all_labels, all_preds)

    # 计算 MAE
    mae = mean_absolute_error(all_labels, all_preds)

    # 计算 PCC (皮尔逊相关系数)
    pcc = np.corrcoef(all_labels, all_preds)[0, 1]

    # 输出结果
    print(f'Used {model_path[:-3]}. Test MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}')
    logging.info(
        f'Used {model_path[:-3]}. Test Results - MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}')

    return mse, r2, mae, pcc


def log_metrics(epoch, total_loss, auc, f1, precision, recall, accuracy):
    """记录每个epoch的结果"""
    logging.info(f"Epoch {epoch}: Loss = {total_loss:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, "
                 f"Precision = {precision:.4f}, Recall = {recall:.4f}, Accuracy = {accuracy:.4f}")


def evaluate_performance(labels, preds, probs):
    """计算 AUC、F1、精确率、召回率、准确率等指标"""
    auc = roc_auc_score(labels, probs)
    labels = [int(label) for label in labels]
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    return auc, f1, precision, recall, accuracy


def cross_validation(dataset_path, CV_FOLDS=10, epochs=50, lr=0.1, batch_size=8, hidden_feats=512, bidirected=False):
    all_accuracies = []
    feat_dim = None
    best_model_path = None
    node_dict = {'node': 0}
    edge_dict = {'HBOND': 0, 'IONIC': 1, 'PICATION': 2, 'PIPISTACK': 3, 'SSBOND': 4, 'VDW': 5}

    for cv_select in range(CV_FOLDS):
        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # 加载数据集，根据当前折数选择训练集和验证集
        train_dataset, valid_dataset, feat_dim, relations = load_graphpred_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, bidirected=bidirected
        )

        # 训练模型并记录验证集的准确率
        _, best_loss, mse, r2, mae, pcc, model_path  = train_gtn_model(train_dataset, valid_dataset, feat_dim,
                                                                              hidden_feats=hidden_feats, num_classes=1,
                                                                              epochs=epochs, lr=lr, node_dict=node_dict,
                                                                              edge_dict=edge_dict,
                                                                              batch_size=batch_size, fold=cv_select, relations=relations)
        all_accuracies.append(best_loss)

        # 记录验证集中表现最好的模型路径
        if best_model_path is None or best_loss < max([loss for loss in all_accuracies]):
            best_model_path = model_path

    # 打印十折交叉验证的结果
    mean_accuracy = sum(all_accuracies) / CV_FOLDS
    logging.info(f"\n10-Fold Cross-Validation Mean Accuracy: {mean_accuracy:.4f}")
    print(f"\n10-Fold Cross-Validation Mean Accuracy: {mean_accuracy:.4f}")

    return best_model_path, mean_accuracy


# 运行十折交叉验证
dataset_path = 'D:/program/GitHub/protein_wang/data/output_Fold_regression'
hidden_feats = 512
CV_FOLDS = 10
# best_model_path, _ = cross_validation(dataset_path, CV_FOLDS=CV_FOLDS, epochs=50, lr=1e-3, batch_size=16,
#                                       hidden_feats=hidden_feats, bidirected=False)

# 存储所有折的指标
all_mse = []
all_r2 = []
all_mae = []
all_pcc = []

# 加载测试集并评估最优模型
relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
feat_dim = 572
test_dataset = load_graphpred_testDataset(dataset_path, bidirected=False)  # 加载测试集
for fold in range(CV_FOLDS):
    best_model_path = f'best_model_fold_{fold}.pt'
    mse, r2, mae, pcc = test_best_model(test_dataset, feat_dim, hidden_feats=hidden_feats, num_classes=1, model_path=best_model_path, batch_size=8, relations=relations)
    # 保存指标
    all_mse.append(mse)
    all_r2.append(r2)
    all_mae.append(mae)
    all_pcc.append(pcc)
    if fold == 6:
        break

# 计算平均值和标准差
mean_mse, std_mse = np.mean(all_mse), np.std(all_mse)
mean_r2, std_r2 = np.mean(all_r2), np.std(all_r2)
mean_mae, std_mae = np.mean(all_mae), np.std(all_mae)
mean_pcc, std_pcc = np.mean(all_pcc), np.std(all_pcc)

# 打印结果
print(f'\nOverall Test Metrics Across {CV_FOLDS} Folds:')
print(f'MSE: {mean_mse:.4f} ± {std_mse:.4f}')
print(f'R²: {mean_r2:.4f} ± {std_r2:.4f}')
print(f'MAE: {mean_mae:.4f} ± {std_mae:.4f}')
print(f'PCC: {mean_pcc:.4f} ± {std_pcc:.4f}')

# 记录到日志
logging.info(f'Overall Test Metrics Across {CV_FOLDS} Folds:')
logging.info(f'MSE: {mean_mse:.4f} ± {std_mse:.4f}')
logging.info(f'R²: {mean_r2:.4f} ± {std_r2:.4f}')
logging.info(f'MAE: {mean_mae:.4f} ± {std_mae:.4f}')
logging.info(f'PCC: {mean_pcc:.4f} ± {std_pcc:.4f}')
