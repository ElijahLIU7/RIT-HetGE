import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import random
from protein.pyHGT.conv import HGTConv
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import logging
import os
from dgl.data import DGLDataset
import numpy as np
from HGRIFN.utils import load_dataset, load_testDataset


# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Graphdataset(DGLDataset):
    def __init__(self, name, graphs, labels, name_protein):
        super(Graphdataset, self).__init__(name=name)
        self.graphs = graphs
        self.labels = labels
        self.name_protein = name_protein

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


class HGT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, relations=None, num_layers=1, num_heads=1, num_ntypes=1,
                 num_etypes=6):
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


def evaluate_performance(labels, preds, probs):
    """计算 AUC、F1、精确率、召回率、准确率等指标"""
    auc = roc_auc_score(labels, probs)
    labels = [int(label) for label in labels]
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    accuracy = accuracy_score(labels, preds)
    return auc, f1, precision, recall, accuracy


def log_metrics(epoch, total_loss, auc, f1, precision, recall, accuracy):
    """记录每个epoch的结果"""
    logging.info(f"Epoch {epoch}: Loss = {total_loss:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, "
                 f"Precision = {precision:.4f}, Recall = {recall:.4f}, Accuracy = {accuracy:.4f}")


def train_hgt_model(train_dataset, valid_dataset, feat_dim, hidden_feats, num_classes, epochs=50, lr=0.1, batch_size=8,
                    node_dict=None, edge_dict=None, fold=None, relations=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGT(in_feats=feat_dim, hidden_feats=hidden_feats, out_feats=num_classes, num_heads=8,
                relations=relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 数据加载器
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_accuracy = 0
    step_loss_early_stop = 10  # 早停器
    loss_early_stop = float('inf')
    pcc_early_stop = step_loss_early_stop
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
        all_probs = []
        with torch.no_grad():
            for batch in valid_loader:
                graphs, labels = batch[0].to(device), batch[1].to(device)

                logits = model(graphs)
                probs = torch.sigmoid(logits)  # 使用sigmoid获取概率
                predicted = (probs > 0.5).int()  # 预测为1的阈值

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # 计算各项指标
        auc, f1, precision, recall, accuracy = evaluate_performance(all_labels, all_preds, all_probs)

        # 保存验证集上表现最好的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
        if loss <= loss_early_stop:
            loss_early_stop = loss
            pcc_early_stop = step_loss_early_stop
        else:
            pcc_early_stop -= 1
        if pcc_early_stop == 0:
            print(f"Early stopping at epoch {epoch + 1}/{epochs}")
            break

        log_metrics(epoch + 1, total_loss, auc, f1, precision, recall, accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Validation AUC: {auc:.4f}, '
              f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

    print("Training completed.")
    return model, accuracy, auc, f1, precision, recall, best_model_path


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
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            graphs, labels = batch[0].to(device), batch[1].to(device)
            logits = model(graphs)
            probs = torch.sigmoid(logits)  # 使用sigmoid获取概率
            predicted = (probs > 0.5).int()  # 预测为1的阈值

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算各项指标
    auc, f1, precision, recall, accuracy = evaluate_performance(all_labels, all_preds, all_probs)
    print(
        f'Used {model_path[:-3]}. Test AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
    logging.info(
        f'Used {model_path[:-3]}. Test Results - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

    return auc, f1, precision, recall, accuracy


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
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, is_classification=True
        )

        # 训练模型并记录验证集的准确率
        _, accuracy, auc, f1, precision, recall, model_path = train_hgt_model(train_dataset, valid_dataset, feat_dim,
                                                                              hidden_feats=hidden_feats, num_classes=1,
                                                                              epochs=epochs, lr=lr, node_dict=node_dict,
                                                                              edge_dict=edge_dict,
                                                                              batch_size=batch_size, fold=cv_select,
                                                                              relations=relations)
        all_accuracies.append(accuracy)

        # 记录验证集中表现最好的模型路径
        if best_model_path is None or auc > max([auc for auc in all_accuracies]):
            best_model_path = model_path

    # 打印十折交叉验证的结果
    mean_accuracy = sum(all_accuracies) / CV_FOLDS
    logging.info(f"\n10-Fold Cross-Validation Mean Accuracy: {mean_accuracy:.4f}")
    print(f"\n10-Fold Cross-Validation Mean Accuracy: {mean_accuracy:.4f}")

    return best_model_path, mean_accuracy


# 运行十折交叉验证
dataset_path = 'D:/program/GitHub/protein_wang/data/output_Fold_regression'
hidden_feats = 1024
CV_FOLDS = 10
best_model_path, _ = cross_validation(dataset_path, CV_FOLDS=CV_FOLDS, epochs=50, lr=1e-4, batch_size=8,
                                      hidden_feats=hidden_feats, bidirected=False)

# 存储所有折的指标
all_aucs = []
all_f1s = []
all_precisions = []
all_recalls = []
all_accuracies = []

# 加载测试集并评估最优模型
relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
feat_dim = 572
test_dataset = load_graphpred_testDataset(dataset_path, bidirected=False, is_classification=True)  # 加载测试集
for fold in range(CV_FOLDS):
    best_model_path = f'best_model_fold_{fold}.pt'
    auc, f1, precision, recall, accuracy = test_best_model(test_dataset, feat_dim, hidden_feats=hidden_feats, num_classes=1, model_path=best_model_path, batch_size=8, relations=relations)
    # 保存指标
    all_aucs.append(auc)
    all_f1s.append(f1)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_accuracies.append(accuracy)

# 计算平均值和标准差
mean_auc, std_auc = np.mean(all_aucs), np.std(all_aucs)
mean_f1, std_f1 = np.mean(all_f1s), np.std(all_f1s)
mean_precision, std_precision = np.mean(all_precisions), np.std(all_precisions)
mean_recall, std_recall = np.mean(all_recalls), np.std(all_recalls)
mean_accuracy, std_accuracy = np.mean(all_accuracies), np.std(all_accuracies)

# 打印结果
print(f'\nOverall Test Metrics Across {CV_FOLDS} Folds:')
print(f'AUC: {mean_auc:.4f} ± {std_auc:.4f}')
print(f'F1: {mean_f1:.4f} ± {std_f1:.4f}')
print(f'Precision: {mean_precision:.4f} ± {std_precision:.4f}')
print(f'Recall: {mean_recall:.4f} ± {std_recall:.4f}')
print(f'Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}')

# 记录到日志
logging.info(f'Overall Test Metrics Across {CV_FOLDS} Folds:')
logging.info(f'AUC: {mean_auc:.4f} ± {std_auc:.4f}')
logging.info(f'F1: {mean_f1:.4f} ± {std_f1:.4f}')
logging.info(f'Precision: {mean_precision:.4f} ± {std_precision:.4f}')
logging.info(f'Recall: {mean_recall:.4f} ± {std_recall:.4f}')
logging.info(f'Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
