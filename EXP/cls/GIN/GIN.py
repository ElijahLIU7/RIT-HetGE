import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.nn.pytorch import GINConv
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import logging
import os
import numpy as np
from protein_wang.pyHGT.utils_homogeneous_classify import load_graphpred_dataset, load_graphpred_testDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.relu(self.batch_norm(self.linear1(x)))
        x = self.linear2(x)
        return x

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=4):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层
        mlp = MLP(in_feats, hidden_feats, hidden_feats)
        self.layers.append(GINConv(mlp, 'sum'))

        # 隐藏层
        for _ in range(num_layers - 2):
            mlp = MLP(hidden_feats, hidden_feats, hidden_feats)
            self.layers.append(GINConv(mlp, 'sum'))

        # 输出层
        mlp = MLP(hidden_feats, hidden_feats, out_feats)
        self.layers.append(GINConv(mlp, 'sum'))

    def forward(self, g, x):
        all_layer_outputs = []
        for conv in self.layers:
            x = conv(g, x)
            all_layer_outputs.append(x)  # 保存每层的输出特征

        g.ndata['h'] = x
        g.ndata['penultimate'] = all_layer_outputs[-2]

        return dgl.mean_nodes(g, 'h').squeeze(), dgl.mean_nodes(g, 'penultimate').squeeze()     # 图级表示，用于分类


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


def train_gcn_model(train_dataset, valid_dataset, feat_dim, hidden_feats,num_classes, epochs=50, lr=0.1, batch_size=1,
                    fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_feats=feat_dim, hidden_feats=hidden_feats, out_feats=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 数据加载器
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_accuracy = 0
    best_model_path = f'best_model_fold_{fold}.pt' if fold is not None else 'best_model.pt'

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            graphs, labels, _ = batch
            graphs = graphs.to(device)
            labels = labels.float().to(device)  # 转换为浮点型用于BCEWithLogitsLoss
            features = graphs.ndata['emb'].to(device)  # 使用节点特征 'emb'
            logits, _ = model(graphs, features)
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
                features = graphs.ndata['emb'].to(device)
                logits, _ = model(graphs, features)
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

        log_metrics(epoch + 1, total_loss, auc, f1, precision, recall, accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Validation AUC: {auc:.4f}, '
              f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

    print("Training completed.")
    return model, accuracy, auc, f1, precision, recall, best_model_path


def test_best_model(test_dataset, feat_dim, hidden_feats, num_classes, model_path, batch_size=8):
    """使用测试集评估最优模型并进行t-SNE可视化"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(in_feats=feat_dim, hidden_feats=hidden_feats, out_feats=num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []
    all_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            graphs, labels = batch[0].to(device), batch[1].to(device)
            features = graphs.ndata['emb'].to(device)
            logits, layer_outputs = model(graphs, features)
            probs = torch.sigmoid(logits)  # 使用sigmoid获取概率
            predicted = (probs > 0.5).int()  # 预测为1的阈值

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_embeddings.extend(layer_outputs.cpu().numpy())  # 提取节点嵌入特征

    # 转换为NumPy数组
    all_embeddings = np.array(all_embeddings)

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_embeddings)

    # 可视化t-SNE结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label="Class")
    plt.title("t-SNE Visualization of Node Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

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
    relations = None
    best_model_path = None

    for cv_select in range(CV_FOLDS):
        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # 加载数据集，根据当前折数选择训练集和验证集
        train_dataset, valid_dataset, feat_dim, relations = load_graphpred_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, bidirected=bidirected
        )

        # 训练模型并记录验证集的准确率
        _, accuracy, auc, f1, precision, recall, model_path = train_gcn_model(train_dataset, valid_dataset, feat_dim,
                                                                              hidden_feats=hidden_feats, num_classes=1,
                                                                              epochs=epochs, lr=lr,
                                                                              batch_size=batch_size, fold=cv_select)
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
hidden_feats = 512
CV_FOLDS = 10
best_model_path, _ = cross_validation(dataset_path, CV_FOLDS=CV_FOLDS, epochs=50, lr=1e-3, batch_size=8,
                                      hidden_feats=hidden_feats, bidirected=False)

# 存储所有折的指标
all_aucs = []
all_f1s = []
all_precisions = []
all_recalls = []
all_accuracies = []

# 加载测试集并评估最优模型
test_dataset, feat_dim, _ = load_graphpred_testDataset(dataset_path, bidirected=False)  # 加载测试集
for fold in range(CV_FOLDS):
    best_model_path = f'best_model_fold_{fold}.pt'
    auc, f1, precision, recall, accuracy = test_best_model(test_dataset, feat_dim, hidden_feats=hidden_feats, num_classes=1, model_path=best_model_path, batch_size=8)
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
