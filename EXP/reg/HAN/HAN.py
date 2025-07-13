import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.nn.pytorch import GATConv
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import logging
import os
import numpy as np
from protein_wang.pyHGT.utils import load_graphpred_dataset, load_graphpred_testDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, num_walks_per_node, walk_length, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        self.num_walks_per_node = num_walks_per_node
        self.walk_length = walk_length

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(
                in_size,
                out_size,
                layer_num_heads,
                dropout,
                dropout,
                activation=F.elu,
                allow_zero_in_degree=True,
            )
        )

        # SemanticAttention的in_size是 node-level attention 的 out_size 乘以多头注意力机制的head数量 layer_num_heads
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            device = g.device
            self._cached_coalesced_graph.clear()

            # 通过随机游走生成新的元路径
            traces, eids, types = dgl.sampling.random_walk(
                g,
                nodes=torch.arange(g.num_nodes()).to(device),
                length=self.walk_length,
                return_eids=True,
                restart_prob=0.2
                # num_traces_per_node=self.num_walks_per_node
            )

            # 使用 traces 中的源节点和目标节点，注意 traces 的形状是 (num_seeds, length + 1)
            source_nodes = traces[:, :-1].contiguous().view(-1)  # 除去最后一列的所有节点作为源节点
            target_nodes = traces[:, 1:].contiguous().view(-1)  # 除去第一列的所有节点作为目标节点
            # 将生成的随机游走路径转换为邻接矩阵形式
            metapath_graph = dgl.graph((source_nodes, target_nodes), num_nodes=g.num_nodes())
            self._cached_coalesced_graph["random_walk_metapath"] = metapath_graph

            # 对每个随机生成的元路径图应用GAT层
        new_g = self._cached_coalesced_graph["random_walk_metapath"]
        semantic_embeddings.append(self.gat_layers[0](new_g, h).flatten(1))

        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(
        self, num_walks_per_node, walk_length, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(num_walks_per_node, walk_length, in_size, hidden_size, num_heads, dropout)
        )
        for l in range(1, num_heads):
            self.layers.append(
                HANLayer(
                    num_walks_per_node,
                    walk_length,
                    hidden_size * num_heads,
                    hidden_size,
                    num_heads,
                    dropout,
                )
            )

        # # Pooling layer to aggregate node embeddings into a graph-level embedding
        # self.pooling = dgl.nn.GlobalAttentionPooling(
        #     gate_nn=nn.Linear(hidden_size * num_heads, out_size)  # Attention pooling gate
        # )
        # Final fully connected layer for graph-level prediction
        self.predict = nn.Linear(hidden_size * num_heads, out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        # 全局池化获取图级别的表示
        with g.local_scope():
            g.ndata['h'] = h
            g_emb = dgl.mean_nodes(g, 'h')  # 也可以使用 sum 或 max 来进行全局池化

        # # Apply graph-level pooling to get a single embedding for the entire graph
        # g_emb = self.pooling(g, h)  # g_emb shape: (B, D), B is the number of graphs (batch size)

        return self.predict(g_emb)      # self.pooling(g, h)


def train_gcn_model(train_dataset, valid_dataset, num_walks_per_node, walk_length, feat_dim, hidden_feats, num_classes, epochs=50, lr=0.1, batch_size=1,
                    fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HAN(num_walks_per_node=num_walks_per_node, walk_length=walk_length, in_size=feat_dim, hidden_size=hidden_feats, out_size=num_classes, num_heads=8,dropout=0.1).to(device)
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
            labels = labels.float().to(device)  # 转换为浮点型用于BCEWithLogitsLoss
            features = graphs.ndata['emb'].to(device)  # 使用节点特征 'emb'
            logits = model(graphs, features)
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
                logits = model(graphs, features)

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

        log_metrics(epoch + 1, total_loss, mse, r2, mae, pcc)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Validation MSE: {mse:.4f}, '
              f'R²: {r2:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}')

    print("Training completed.")
    return model, best_loss, mse, r2, mae, pcc, best_model_path


def test_best_model(test_dataset, num_walks_per_node, walk_length, feat_dim, hidden_feats, num_classes, model_path, batch_size=8):
    """使用测试集评估最优模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HAN(num_walks_per_node=num_walks_per_node, walk_length=walk_length, in_size=feat_dim, hidden_size=hidden_feats, out_size=num_classes, num_heads=8,dropout=0.1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            graphs, labels = batch[0].to(device), batch[1].to(device)
            features = graphs.ndata['emb'].to(device)
            logits = model(graphs, features)

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
def cross_validation(dataset_path, num_walks_per_node, walk_length, CV_FOLDS=10, epochs=50, lr=0.1, batch_size=8, hidden_feats=512, bidirected=False):
    all_accuracies = []
    feat_dim = None
    relations = None
    best_model_path = None

    for cv_select in range(CV_FOLDS):
        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # 加载数据集，根据当前折数选择训练集和验证集
        train_dataset, valid_dataset, feat_dim, relations = load_graphpred_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, bidirected=bidirected
        )

        # 训练模型并记录验证集的准确率
        _, best_loss, mse, r2, mae, pcc, model_path = train_gcn_model(train_dataset, valid_dataset, num_walks_per_node, walk_length, feat_dim,
                                                                              hidden_feats=hidden_feats, num_classes=1,
                                                                              epochs=epochs, lr=lr,
                                                                              batch_size=batch_size, fold=cv_select)
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
num_walks_per_node = 2
walk_length = 5
best_model_path, _ = cross_validation(dataset_path, num_walks_per_node, walk_length, CV_FOLDS=CV_FOLDS, epochs=50, lr=1e-3, batch_size=1,
                                      hidden_feats=hidden_feats, bidirected=False)

# 存储所有折的指标
all_mse = []
all_r2 = []
all_mae = []
all_pcc = []

# 加载测试集并评估最优模型
test_dataset, feat_dim, _ = load_graphpred_testDataset(dataset_path, bidirected=True)  # 加载测试集
for fold in range(CV_FOLDS):
    best_model_path = f'best_model_fold_{fold}.pt'
    mse, r2, mae, pcc = test_best_model(test_dataset, num_walks_per_node, walk_length, feat_dim, hidden_feats=hidden_feats, num_classes=1, model_path=best_model_path, batch_size=8)
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
