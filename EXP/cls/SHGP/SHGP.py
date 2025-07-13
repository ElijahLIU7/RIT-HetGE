import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import dgl
import numpy as np
import logging
import optuna
import argparse

from tqdm import tqdm
from abc import ABCMeta
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from protein_wang.pyHGT.utils import load_graphpred_dataset, load_graphpred_testDataset


# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


class ATT_HGCN(BaseModel):

    def __init__(self, net_schema, layer_shape, type_fusion='att', type_att_size=64):
        super(ATT_HGCN, self).__init__()
        self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size)
        self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size)

        self.layer_shape = layer_shape
        self.hidden_size = layer_shape[-1]
        # Added MLP for graph-level embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )

    def forward(self, ft_dict, adj_dict, g):
        """
            ft_dict: 节点特征字典
            adj_dict: 邻接矩阵字典
            pool: 'mean' 或 'max'，用于整图嵌入的池化方式
        """
        attention_list = []

        # 第一个HeteGCN层
        x_dict, attention_dict = self.hgc1(ft_dict, adj_dict)
        attention_list.append((attention_dict))
        x_dict = self.non_linear(x_dict)
        x_dict = self.dropout_ft(x_dict, 0.5)

        # 第二个HeteGCN层
        x_dict, attention_dict = self.hgc2(x_dict, adj_dict)
        attention_list.append((attention_dict))

        with g.local_scope():
            g.ndata['h'] = x_dict['node']
            hg = dgl.mean_nodes(g, 'h')  # 也可以使用 sum 或 max 来进行全局池化

        # 整图池化
        graph_embd = self.mlp(hg)

        return graph_embd, attention_list

    def non_linear(self, x_dict):
        y_dict = {}
        for k in x_dict:
            y_dict[k] = F.elu(x_dict[k])
        return y_dict

    def dropout_ft(self, x_dict, dropout):
        y_dict = {}
        for k in x_dict:
            y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
        return y_dict


class HeteGCNLayer(nn.Module):

    def __init__(self, net_schema, in_layer_shape, out_layer_shape, type_fusion, type_att_size):
        super(HeteGCNLayer, self).__init__()
        self.net_schema = net_schema
        self.in_layer_shape = in_layer_shape
        self.out_layer_shape = out_layer_shape

        self.hete_agg = nn.ModuleDict()
        for k in net_schema:
            self.hete_agg[k] = HeteAggregateLayer(k, net_schema[k], in_layer_shape, out_layer_shape, type_fusion,
                                                  type_att_size)

    def forward(self, x_dict, adj_dict):
        attention_dict = {}
        ret_x_dict = {}
        for k in self.hete_agg.keys():
            ret_x_dict[k], attention_dict[k] = self.hete_agg[k](x_dict, adj_dict)

        return ret_x_dict, attention_dict


class HeteAggregateLayer(nn.Module):

    def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, type_fusion, type_att_size):
        super(HeteAggregateLayer, self).__init__()

        self.nb_list = nb_list
        self.curr_k = curr_k
        self.type_fusion = type_fusion
        self.W_rel = nn.ParameterDict()
        for k in nb_list:
            try:
                self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
            except KeyError as ke:
                self.W_rel[k] = nn.Parameter(torch.FloatTensor(in_layer_shape[self.curr_k], out_shape))
            finally:
                nn.init.xavier_uniform_(self.W_rel[k].data, gain=1.414)

        self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
        nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

        self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)

        if type_fusion == 'att':
            self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
            self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
            self.w_att = nn.Parameter(torch.FloatTensor(2 * type_att_size, 1))
            nn.init.xavier_uniform_(self.w_att.data, gain=1.414)

    def forward(self, x_dict, adj_dict):
        attention_curr_k = 0
        self_ft = torch.mm(x_dict[self.curr_k], self.w_self)

        nb_ft_list = [self_ft]
        nb_name = [self.curr_k + '_self']
        for k in self.nb_list:
            try:
                nb_ft = torch.mm(x_dict[k], self.W_rel[k])
            except KeyError as ke:
                nb_ft = torch.mm(x_dict[self.curr_k], self.W_rel[k])
            finally:
                nb_ft = torch.spmm(adj_dict[k], nb_ft)
                nb_ft_list.append(nb_ft)
                nb_name.append(k)

        if self.type_fusion == 'mean':
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)
            attention = []
        elif self.type_fusion == 'att':
            att_query = torch.mm(self_ft, self.w_query).repeat(len(nb_ft_list), 1)
            att_keys = torch.mm(torch.cat(nb_ft_list, 0), self.w_keys)
            att_input = torch.cat([att_keys, att_query], 1)
            att_input = F.dropout(att_input, 0.5, training=self.training)
            e = F.elu(torch.matmul(att_input, self.w_att))
            attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0, 1), dim=1)  # 4025*3
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)

        output = agg_nb_ft + self.bias

        return output, attention


def extract_features_and_adjacency(batch_graph):
    """
    从DGL异构图提取特征和邻接矩阵，构造ft_dict和adj_dict
    """
    ft_dict = {}
    adj_dict = {}

    # 遍历异构图中的每种节点类型，提取节点特征
    for ntype in batch_graph.ntypes:
        ft_dict[ntype] = batch_graph.nodes[ntype].data['emb']

    # 遍历异构图中的每种边类型，提取邻接矩阵
    for etype in batch_graph.etypes:
        src, dst = batch_graph.edges(etype=etype)
        num_nodes = batch_graph.number_of_nodes()

        # 构造稀疏邻接矩阵
        indices = torch.stack([src, dst])
        num_nodes = batch_graph.num_nodes()
        adj = torch.sparse_coo_tensor(indices, torch.ones(len(src)),
                                      size=(num_nodes, num_nodes), device=batch_graph.device)

        # adj = dgl.create_sparse((src, dst), shape=(batch_graph.num_nodes(), batch_graph.num_nodes()))
        # adj = dgl.backend.sparse_matrix(src, dst, shape=(num_nodes, num_nodes))
        adj_dict[etype] = adj

    return ft_dict, adj_dict


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


def train_model(train_dataset, valid_dataset, net_schema, embed_size,
                         epochs=50, lr=[1e-4, 1e-4], batch_size=8, type_fusion='att', type_att_size=64, fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize ATT_HGCN model
    layer_shape = [572, embed_size, embed_size]
    model = ATT_HGCN(net_schema, layer_shape, type_fusion, type_att_size).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr[0], weight_decay=lr[1])

    # Loss function
    classification_loss = nn.BCEWithLogitsLoss()

    # Data loaders
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_accuracy = 0
    step_loss_early_stop = 10  # 早停器
    loss_early_stop = float('inf')
    best_model_path = f'best_model_fold_{fold}.pt' if fold is not None else 'best_att_hgcn_model.pt'

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        # Set progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for i, batch in enumerate(pbar):
                batch_graph, labels, _ = batch
                batch_graph = batch_graph.to(device)
                # 提取ft_dict和adj_dict
                ft_dict, adj_dict = extract_features_and_adjacency(batch_graph)

                # 将ft_dict和adj_dict移到GPU
                ft_dict = {k: v.to(device) for k, v in ft_dict.items()}
                adj_dict = {k: v.to(device) for k, v in adj_dict.items()}
                labels = labels.to(device)

                # Train the model
                optimizer.zero_grad()

                # Forward propagation
                logits, attention_list = model(ft_dict, adj_dict, batch_graph)

                logits = logits.squeeze()
                # Compute loss
                loss = classification_loss(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Update progress bar description
                pbar.set_postfix({'Loss': total_loss / (i + 1)})

        # Validation
        model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch in valid_loader:
                batch_graph, labels, _ = batch
                batch_graph = batch_graph.to(device)
                # 提取ft_dict和adj_dict
                ft_dict, adj_dict = extract_features_and_adjacency(batch_graph)
                ft_dict = {k: v.to(device) for k, v in ft_dict.items()}
                adj_dict = {k: v.to(device) for k, v in adj_dict.items()}
                labels = labels.to(device)

                logits, attention_list = model(ft_dict, adj_dict, batch_graph)
                logits = logits.squeeze()
                predicted = (logits > 0.5).int()  # 预测为1的阈值

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(logits.cpu().numpy())

        # Compute validation metrics
        auc, f1, precision, recall, accuracy = evaluate_performance(all_labels, all_preds, all_probs)

        # Save model with best validation performance
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
        if loss < loss_early_stop:
            loss_early_stop = loss
            pcc_early_stop = step_loss_early_stop
        else:
            pcc_early_stop -= 1
        if pcc_early_stop == 0:
            print(f"Early stopping at epoch {epoch + 1}/{epochs}")
            break

        log_metrics(epoch + 1, total_loss, auc, f1, precision, recall, accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation AUC: {auc:.4f}, '
              f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
    print("Training completed.")
    return model, accuracy, auc, f1, precision, recall, best_model_path


def test_best_model(test_dataset, embed_size, model_path, batch_size=8):
    """使用测试集评估最优模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    layer_shape = [572, embed_size, embed_size]
    net_schema = {'node': ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']}
    model = ATT_HGCN(net_schema, layer_shape, type_fusion='att', type_att_size=64).to(device)

    model.load_state_dict(torch.load(model_path))
    # Validation
    model.eval()

    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            batch_graph, labels, _ = batch
            batch_graph = batch_graph.to(device)
            # 提取ft_dict和adj_dict
            ft_dict, adj_dict = extract_features_and_adjacency(batch_graph)
            ft_dict = {k: v.to(device) for k, v in ft_dict.items()}
            adj_dict = {k: v.to(device) for k, v in adj_dict.items()}
            labels = labels.to(device)

            logits, attention_list = model(ft_dict, adj_dict, batch_graph)
            logits = logits.squeeze()
            predicted = (logits > 0.5).int()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(logits.cpu().numpy())

        auc, f1, precision, recall, accuracy = evaluate_performance(all_labels, all_preds, all_probs)

        print(
            f'Used {model_path[:-3]}. Test AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
        logging.info(
            f'Used {model_path[:-3]}. Test Results - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

    return auc, f1, precision, recall, accuracy


def cross_validation(dataset_path, CV_FOLDS=10, epochs=50, lr=0.001, batch_size=8, hidden_feats=128):
    all_losses = []
    feat_dim = None
    best_model_path = None
    net_schema = {'node': ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']}

    all_accuracies = []
    for cv_select in range(CV_FOLDS):
        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # Load dataset and select train and validation sets for the current fold
        train_dataset, valid_dataset, feat_dim, relations = load_graphpred_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, is_classification=True
        )

        # Train the model and record the loss on the validation set
        model, accuracy, auc, f1, precision, recall, model_path = train_model(train_dataset, valid_dataset, net_schema=net_schema,
                                                                                embed_size=hidden_feats, epochs=epochs,
                                                                                lr=lr, batch_size=batch_size, fold=cv_select)
        all_accuracies.append(accuracy)

        # 记录验证集中表现最好的模型路径
        if best_model_path is None or auc > max([auc for auc in all_accuracies]):
            best_model_path = model_path
        if cv_select == 4:
            break

    # Print the results of 10-fold cross-validation
    mean_loss = sum(all_losses) / CV_FOLDS
    logging.info(f"\n10-Fold Cross-Validation Mean Loss: {mean_loss:.4f}")
    print(f"\n10-Fold Cross-Validation Mean Loss: {mean_loss:.4f}")

    return best_model_path, mean_loss


def objective(trial):
    epochs = 100
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    # hidden_feats = trial.suggest_categorical("hidden_feats", [64, 128, 256])
    hidden_feats = 256
    # 输出所有参数定义值
    print("Parameters:")
    print("     epochs:", epochs)
    print("     lr:", lr)
    print("     weight_decay:", weight_decay)
    print("     hidden_feats:", hidden_feats)
    # 运行十折交叉验证
    dataset_path = 'D:/program/GitHub/protein_wang/data/output_Fold_regression'
    CV_FOLDS = 10
    # 存储所有折的指标
    all_aucs = []
    all_f1s = []
    all_precisions = []
    all_recalls = []
    all_accuracies = []
    # best_model_path, _ = cross_validation(dataset_path, CV_FOLDS=CV_FOLDS, epochs=epochs, lr=[lr, weight_decay], batch_size=32,
    #                                       hidden_feats=hidden_feats)

    # 加载测试集并评估最优模型
    test_dataset = load_graphpred_testDataset(dataset_path, bidirected=False, is_classification=True)  # 加载测试集
    for fold in range(CV_FOLDS):
        best_model_path = f'best_model_fold_{fold}.pt'
        auc, f1, precision, recall, accuracy = test_best_model(test_dataset, embed_size=hidden_feats, model_path=best_model_path, batch_size=8)
        # 保存指标
        all_aucs.append(auc)
        all_f1s.append(f1)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_accuracies.append(accuracy)
        if fold == 4:
            break

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

    return mean_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=1,
                        help='Number of trial runs.')

    args = parser.parse_args()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)
    #
    # joblib.dump(study, 'study.pkl')

    print(f'Best trial: {study.best_trial.value}')
    print('Best hyperparameters: ', study.best_trial.params)