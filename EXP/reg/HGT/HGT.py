import math
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from protein.pyHGT.conv import HGTConv
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import logging
import os
import numpy as np
from protein_wang.pyHGT.utils import load_graphpred_dataset, load_graphpred_testDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# from dgl import function as fn
# from dgl.nn.pytorch.linear import TypedLinear
# from dgl.nn.pytorch.softmax import edge_softmax
# from abc import ABCMeta
#
#
# class BaseModel(nn.Module, metaclass=ABCMeta):
#     @classmethod
#     def build_model_from_args(reg, args, hg):
#         r"""
#         Build the model instance from args and hg.
#
#         So every subclass inheriting it should override the method.
#         """
#         raise NotImplementedError("Models must implement the build_model_from_args method")
#
#     def __init__(self):
#         super(BaseModel, self).__init__()
#
#     def forward(self, *args):
#         r"""
#         The model plays a role of encoder. So the forward will encoder original features into new features.
#
#         Parameters
#         -----------
#         hg : dgl.DGlHeteroGraph
#             the heterogeneous graph
#         h_dict : dict[str, th.Tensor]
#             the dict of heterogeneous feature
#
#         Return
#         -------
#         out_dic : dict[str, th.Tensor]
#             A dict of encoded feature. In general, it should ouput all nodes embedding.
#             It is allowed that just output the embedding of target nodes which are participated in loss calculation.
#         """
#         raise NotImplementedError
#
#     def extra_loss(self):
#         r"""
#         Some model want to use L2Norm which is not applied all parameters.
#
#         Returns
#         -------
#         th.Tensor
#         """
#         raise NotImplementedError
#
#     def h2dict(self, h, hdict):
#         pre = 0
#         out_dict = {}
#         for i, value in hdict.items():
#             out_dict[i] = h[pre:value.shape[0]+pre]
#             pre += value.shape[0]
#         return out_dict
#
#     def get_emb(self):
#         r"""
#         Return the embedding of a model for further analysis.
#
#         Returns
#         -------
#         numpy.array
#         """
#         raise NotImplementedError
#
#
# class HGT(BaseModel):
#     def __init__(self, node_dict, edge_dict, hidden_dim, out_dim, num_layers, n_heads, dropout, category, use_norm=True):
#         super(HGT, self).__init__()
#         self.node_dict = node_dict
#         self.edge_dict = edge_dict
#
#         self.category = category
#         self.gcs = nn.ModuleList()
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.num_layers = num_layers
#         self.adapt_ws = nn.ModuleList()
#         for _ in range(num_layers):
#             self.gcs.append(HGTLayer(hidden_dim, hidden_dim, node_dict, edge_dict, n_heads, dropout, use_norm = use_norm))
#         self.out = nn.Linear(hidden_dim, out_dim)
#
#     def forward(self, G, h_in=None):
#         h = h_in
#         for i in range(self.num_layers):
#             h = self.gcs[i](G, h)
#
#         # 全局池化获取图级别的表示
#         with G.local_scope():
#             G.ndata['h'] = h
#             hg = dgl.mean_nodes(G, 'h')  # 也可以使用 sum 或 max 来进行全局池化
#         logits = self.classifier(hg).squeeze(-1)
#         return logits           # {self.category: self.out(h[self.category])}
#
#
# class HGTLayer(nn.Module):
#     def __init__(self,
#                  in_dim,
#                  out_dim,
#                  node_dict,
#                  edge_dict,
#                  n_heads,
#                  dropout = 0.2,
#                  use_norm = False):
#         super(HGTLayer, self).__init__()
#
#         self.in_dim        = in_dim
#         self.out_dim       = out_dim
#         self.node_dict     = node_dict
#         self.edge_dict     = edge_dict
#         self.num_types     = len(node_dict)
#         self.num_relations = len(edge_dict)
#         self.total_rel     = self.num_types * self.num_relations * self.num_types
#         self.n_heads       = n_heads
#         self.d_k           = out_dim // n_heads
#         self.sqrt_dk       = math.sqrt(self.d_k)
#         self.att           = None
#
#         self.k_linears   = nn.ModuleList()
#         self.q_linears   = nn.ModuleList()
#         self.v_linears   = nn.ModuleList()
#         self.a_linears   = nn.ModuleList()
#         self.norms       = nn.ModuleList()
#         self.use_norm    = use_norm
#
#         for t in range(self.num_types):
#             self.k_linears.append(nn.Linear(in_dim,   out_dim))
#             self.q_linears.append(nn.Linear(in_dim,   out_dim))
#             self.v_linears.append(nn.Linear(in_dim,   out_dim))
#             self.a_linears.append(nn.Linear(out_dim,  out_dim))
#             if use_norm:
#                 self.norms.append(nn.LayerNorm(out_dim))
#
#         self.relation_pri   = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
#         self.relation_att   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
#         self.relation_msg   = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
#         self.skip           = nn.Parameter(torch.ones(self.num_types))
#         self.drop           = nn.Dropout(dropout)
#
#         nn.init.xavier_uniform_(self.relation_att)
#         nn.init.xavier_uniform_(self.relation_msg)
#
#     def forward(self, G, h):
#         with G.local_scope():
#             node_dict, edge_dict = self.node_dict, self.edge_dict
#             for srctype, etype, dsttype in G.canonical_etypes:
#                 sub_graph = G[srctype, etype, dsttype]
#
#                 k_linear = self.k_linears[node_dict[srctype]]
#                 v_linear = self.v_linears[node_dict[srctype]]
#                 q_linear = self.q_linears[node_dict[dsttype]]
#
#                 k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
#                 v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
#                 q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)
#
#                 e_id = self.edge_dict[etype]
#
#                 relation_att = self.relation_att[e_id]
#                 relation_pri = self.relation_pri[e_id]
#                 relation_msg = self.relation_msg[e_id]
#
#                 k = torch.einsum("bij,ijk->bik", k, relation_att)
#                 v = torch.einsum("bij,ijk->bik", v, relation_msg)
#
#                 sub_graph.srcdata['k'] = k
#                 sub_graph.dstdata['q'] = q
#                 sub_graph.srcdata['v_%d' % e_id] = v
#
#                 sub_graph.apply_edges(fn.v_dot_u('q', 'k', 't'))
#                 attn_score = sub_graph.edata.pop('t').sum(-1) * relation_pri / self.sqrt_dk
#                 attn_score = edge_softmax(sub_graph, attn_score, norm_by='dst')
#
#                 sub_graph.edata['t'] = attn_score.unsqueeze(-1)
#
#             G.multi_update_all({etype : (fn.u_mul_e('v_%d' % e_id, 't', 'm'), fn.sum('m', 't')) \
#                                 for etype, e_id in edge_dict.items()}, cross_reducer = 'mean')
#
#             new_h = {}
#             for ntype in G.ntypes:
#                 '''
#                     Step 3: Target-specific Aggregation
#                     x = norm( W[node_type] * gelu( Agg(x) ) + x )
#                 '''
#                 n_id = node_dict[ntype]
#                 alpha = torch.sigmoid(self.skip[n_id])
#                 t = G.nodes[ntype].data['t'].view(-1, self.out_dim)
#                 trans_out = self.drop(self.a_linears[n_id](t))
#                 trans_out = trans_out * alpha + h[ntype] * (1-alpha)
#                 if self.use_norm:
#                     new_h[ntype] = self.norms[n_id](trans_out)
#                 else:
#                     new_h[ntype] = trans_out
#             return new_h


class HGT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, relations=None, num_layers= 2, num_heads=1, num_ntypes=1, num_etypes=6):
        super(HGT, self).__init__()
        self.num_ntypes = num_ntypes
        self.num_etypes = num_etypes  # 将 num_etypes 存储为类属性
        self.relations = relations

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HGTConv(
                in_dim=hidden_feats,
                out_dim=hidden_feats,
                num_types=1,
                num_relations=relations,
                n_heads=num_heads,
                use_norm=True
            ))

        self.input_proj = nn.Linear(in_feats, hidden_feats)
        self.classifier = nn.Linear(hidden_feats, out_feats)

    def forward(self, g):
        # 提取节点特征
        h = g.ndata['emb']
        h = self.input_proj(h)

        for layer in self.layers:
            for i, graph_layer in enumerate(self.relations):
                rel_graph = g['node', graph_layer, 'node']
                source_nodes, destination_nodes, edges = rel_graph.edges(form='all')
                edge_index = torch.stack((source_nodes, destination_nodes))
                edge_weights = rel_graph.edges[graph_layer].data['weight']
                edge_index = torch.cat((edge_index, edge_weights.T), dim=0)
                node_type = torch.zeros(h.size(0)).to(h.device)
                edge_type = torch.full((edge_index[0].size(0),), i, dtype=torch.long, device=rel_graph.device)
                edge_time = torch.zeros(edge_index[0].size(0), dtype=torch.long, device=rel_graph.device)
                h = layer(h, node_type, edge_index, edge_type, edge_time)


        # 全局池化获取图级别的表示
        with g.local_scope():
            g.ndata['h'] = h
            hg = dgl.mean_nodes(g, 'h')  # 也可以使用 sum 或 max 来进行全局池化
        logits = self.classifier(hg).squeeze(-1)
        return logits


def train_hgt_model(train_dataset, valid_dataset, feat_dim, hidden_feats, num_classes, epochs=50, lr=0.1, batch_size=8,
                    node_dict=None, edge_dict=None, fold=None, relations=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = HGT(node_dict=node_dict, edge_dict=edge_dict, hidden_dim=hidden_feats, out_dim=num_classes,
    #             num_layers=2, n_heads=8, dropout=0.2, category='category_name').to(device)
    model = HGT(in_feats=feat_dim, hidden_feats=hidden_feats, out_feats=num_classes, num_heads=8,
                relations=relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.HuberLoss(reduction='mean', delta=10.0)

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
            labels = labels.float().to(device)  # 转换为浮点型用于BCEWithLogitsLoss
            logits = model(graphs)
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

                logits = model(graphs)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(logits.cpu().numpy())

        # 将预测值和实际值转换为 NumPy 数组
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # 计算验证集上的损失
        val_loss = np.mean((np.array(all_labels) - np.array(all_preds)) ** 2)

        # 计算各项指标
        # 计算 MSE
        mse = mean_squared_error(all_labels, all_preds)

        # 计算 R²
        r2 = r2_score(all_labels, all_preds)

        # 计算 MAE
        mae = mean_absolute_error(all_labels, all_preds)

        # 计算 PCC (皮尔逊相关系数)
        pcc = np.corrcoef(all_labels, all_preds)[0, 1]

        # 保存验证集上表现最好的模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        train_loss = total_loss / len(train_loader)
        log_metrics(epoch + 1, train_loss, mse, r2, mae, pcc)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation MSE: {mse:.4f}, '
              f'R²: {r2:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}')
    print("Training completed.")
    return model, best_loss, mse, r2, mae, pcc, best_model_path


def test_best_model(test_dataset, feat_dim, hidden_feats, num_classes, model_path, batch_size=8, relations=None):
    """使用测试集评估最优模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGT(in_feats=feat_dim, hidden_feats=hidden_feats, out_feats=num_classes, num_heads=8,
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


def log_metrics(epoch, total_loss, mse, r2, mae, pcc):
    """记录每个epoch的结果"""
    logging.info(f"Epoch {epoch}: Loss = {total_loss:.4f}, MSE = {mse:.4f}, R2 = {r2:.4f}, "
                 f"MAE = {mae:.4f}, PCC = {pcc:.4f}")


# 十折交叉验证
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
        _, best_loss, mse, r2, mae, pcc, model_path = train_hgt_model(train_dataset, valid_dataset, feat_dim,
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
best_model_path, _ = cross_validation(dataset_path, CV_FOLDS=CV_FOLDS, epochs=50, lr=1e-3, batch_size=16,
                                      hidden_feats=hidden_feats, bidirected=False)

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
