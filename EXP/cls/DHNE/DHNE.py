import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import dgl
import logging

from tqdm import tqdm
from abc import ABCMeta
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from protein_wang.pyHGT.utils import load_graphpred_dataset, load_graphpred_testDataset
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


class DHNE(BaseModel):
    r"""
        **Title:** Structural Deep Embedding for Hyper-Networks

        **Authors:** Ke Tu, Peng Cui, Xiao Wang, Fei Wang, Wenwu Zhu

        DHNE was introduced in `[paper] <https://arxiv.org/abs/1711.10146>`_
        and parameters are defined as follows:

        Parameters
        ----------
        nums_type : list
            the type of nodes
        dim_features : array
            the embedding dimension of nodes
        embedding_sizes : int
            the embedding dimension size
        hidden_size : int
            The hidden full connected layer size
        device : int
            the device DHNE working on
        """
    def __init__(self, nums_type, dim_features, embedding_sizes, hidden_size, device):
        super().__init__()
        self.dim_features = dim_features
        self.embedding_sizes = embedding_sizes
        self.hidden_size = hidden_size
        self.nums_type = nums_type
        self.device = device

        # auto-encoder
        self.encodeds = [
            nn.Linear(self.nums_type, self.embedding_sizes)]
        self.decodeds = [
            nn.Linear(self.embedding_sizes, self.nums_type)]
        # self.encodeds = [
        #     nn.Linear((self.nums_type if i == 0 else self.embedding_sizes[i]), self.embedding_sizes[i]) for i in range(3)]
        # self.decodeds = [
        #     nn.Linear(self.embedding_sizes[i], (self.nums_type if i == 0 else self.embedding_sizes[i]), ) for i in
        #     range(3)]
        self.hidden_layer = nn.Linear(
            self.embedding_sizes, self.hidden_size)

        self.output_layer = nn.Linear(self.hidden_size, 1)

        # Added MLP for graph-level embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        # 添加特征提取相关属性
        self.relu_features = None  # 用于存储ReLU后的特征
        self._register_hooks()  # 注册钩子获取中间层输出

    def _register_hooks(self):
        """注册前向钩子捕获ReLU层输出"""

        def hook_function(module, input, output):
            self.relu_features = output.detach().cpu()

        # 获取ReLU层实例
        relu_layer = self.mlp[1]
        relu_layer.register_forward_hook(hook_function)

    def forward(self, g):
        """
        The forward part of the DHNE.

        Parameters
        ----------
        g : dgl.DGLHeteroGraph
            The input DGL HeteroGraph

        Returns
        -------
        tensor
            The logits after DHNE training.
        """
        # Assuming input graph has node features that need to be used in the model.
        encodeds = []
        decodeds = []

        # Iterating through node types and extracting features
        for i, ntype in enumerate(g.ntypes):
            if 'emb' in g.nodes[ntype].data:
                input_features = g.nodes[ntype].data['emb'].to(self.device)
                encoded = torch.tanh(self.encodeds[i].to(self.device)(input_features))
                encodeds.append(encoded)
                decoded = torch.sigmoid(self.decodeds[i].to(self.device)(encoded))
                decodeds.append(decoded)

        # Merge encoded features
        merged = torch.cat(encodeds, dim=1).to(self.device)
        hidden = torch.tanh(self.hidden_layer(merged)).to(self.device)

        with g.local_scope():
            g.ndata['h'] = hidden
            hg = dgl.mean_nodes(g, 'h')  # 也可以使用 sum 或 max 来进行全局池化

        output = self.output_layer.to(self.device)(hg)
        graph_embed = self.mlp.to(self.device)(hg)

        return decodeds + [output, graph_embed], self.relu_features


def train_dhne_model(train_dataset, valid_dataset, embed_size, epochs=50, lr=0.001, batch_size=8, fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize DHNE model
    dhne = DHNE(nums_type=572, dim_features=embed_size, embedding_sizes=64,
                hidden_size=128, device=device).to(device)

    # Optimizer
    optimizer = optim.Adam(dhne.parameters(), lr=lr)

    # Loss function
    classification_loss = nn.BCEWithLogitsLoss()

    # Data loaders
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_accuracy = 0
    step_loss_early_stop = 10  # 早停器
    loss_early_stop = float('inf')
    pcc_early_stop = step_loss_early_stop
    best_model_path = f'best_dhne_model_fold_{fold}_last.pt' if fold is not None else 'best_dhne_model.pt'

    for epoch in range(epochs):
        dhne.train()
        total_loss = 0
        # Set progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for i, batch in enumerate(pbar):
                input_ids, labels, _ = batch
                input_ids, labels = input_ids.to(device), labels.to(device)

                # Train the model
                optimizer.zero_grad()

                # Forward propagation
                output, _ = dhne(input_ids)
                graph_embed = output[-1].squeeze(-1)

                # Compute loss
                loss = classification_loss(graph_embed.squeeze(), labels.float())
                total_loss += loss.item()

                # Backward propagation and optimization
                loss.backward()
                optimizer.step()

                # Update progress bar description
                pbar.set_postfix({'Loss': total_loss / (i + 1)})
                loss = total_loss / (i + 1)

        # Validation
        dhne.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, labels, _ = batch
                input_ids, labels = input_ids.to(device), labels.to(device)

                output, _ = dhne(input_ids)
                graph_embed = output[-1].squeeze(-1)
                predicted = (graph_embed > 0.5).int()  # 预测为1的阈值

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(graph_embed.cpu().numpy())

        # 计算各项指标
        auc, f1, precision, recall, accuracy = evaluate_performance(all_labels, all_preds, all_probs)

        # 保存验证集上表现最好的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(dhne.state_dict(), best_model_path)
        if loss <= loss_early_stop:
            loss_early_stop = loss
            pcc_early_stop = step_loss_early_stop
        else:
            pcc_early_stop -= 1
        if pcc_early_stop == 0:
            print(f"Early stopping at epoch {epoch + 1}/{epochs}")
            break


        log_metrics(epoch + 1, loss, auc, f1, precision, recall, accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation AUC: {auc:.4f}, '
              f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
    print("Training completed.")

    return dhne, accuracy, auc, f1, precision, recall, best_model_path


def test_best_model(test_dataset, embed_size, model_path, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize DHNE model
    dhne = DHNE(nums_type=572, dim_features=embed_size, embedding_sizes=64,
                hidden_size=128, device=device).to(device)
    dhne.load_state_dict(torch.load(model_path))
    dhne.eval()

    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []
    all_features = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels, _ = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            output, layer_features = dhne(input_ids)
            graph_embed = output[-1].squeeze(-1)
            predicted = (graph_embed > 0.5).int()  # 预测为1的阈值

            all_features.append(layer_features.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(graph_embed.cpu().numpy())

        # 计算各项指标
        # 转换为numpy数组
        all_embeddings = np.concatenate(all_features, axis=0)
        plot_tsne(all_embeddings, all_labels, all_preds)

        auc, f1, precision, recall, accuracy = evaluate_performance(all_labels, all_preds, all_probs)
        print(
            f'Used {model_path[:-3]}. Test AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
        logging.info(
            f'Used {model_path[:-3]}. Test Results - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

    return auc, f1, precision, recall, accuracy


def plot_tsne(embeddings, true_labels, pred_labels):
    """Visualize embeddings using t-SNE"""
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings) - 1), learning_rate=50)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create a dataframe for plotting
    plot_data = {
        'Dimension 1': embeddings_2d[:, 0],
        'Dimension 2': embeddings_2d[:, 1],
        'True Label': true_labels,
        'Predicted Label': pred_labels
    }
    df = pd.DataFrame(plot_data)

    binary_palette = {
        0: '#FF3333',
        1: '#3333FF'
    }

    sns.scatterplot(
        x='Dimension 1', y='Dimension 2',
        hue='True Label', palette=binary_palette,
        data=df,
        legend='full',
        alpha=0.7
    )

    plt.title('t-SNE Visualization of DHNE')

    plt.tight_layout()
    plt.savefig("t-SNE_DHNE.png", dpi=300)
    plt.show()


def cross_validation_dhne(dataset_path, CV_FOLDS=10, epochs=50, lr=0.001, batch_size=8, hidden_feats=128):
    all_losses = []
    feat_dim = None
    best_model_path = None

    for cv_select in range(CV_FOLDS):
        if cv_select == 5:
            break

        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # Load dataset and select train and validation sets for the current fold
        train_dataset, valid_dataset, feat_dim, relations = load_graphpred_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, is_classification=True
        )

        # Train the model and record the loss on the validation set
        dhne_model, best_loss, mse, r2, mae, pcc, model_path = train_dhne_model(train_dataset, valid_dataset,
                                                                                embed_size=hidden_feats, epochs=epochs,
                                                                                lr=lr, batch_size=batch_size, fold=cv_select)
        all_losses.append(best_loss)

        # Record the best model path based on validation set performance
        if best_model_path is None or best_loss < min(all_losses):
            best_model_path = model_path
        # if cv_select==0:
        #     break

    # Print the results of 10-fold cross-validation
    mean_loss = sum(all_losses) / 5
    logging.info(f"\n10-Fold Cross-Validation Mean Loss: {mean_loss:.4f}")
    print(f"\n10-Fold Cross-Validation Mean Loss: {mean_loss:.4f}")

    return best_model_path, mean_loss


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


if '__main__' == __name__:
    # 运行十折交叉验证
    dataset_path = 'D:/program/GitHub/protein_wang/data/output_Fold_regression'
    hidden_feats = 572
    CV_FOLDS = 10
    # best_model_path, _ = cross_validation_dhne(dataset_path, CV_FOLDS=CV_FOLDS, epochs=200, lr=1e-3, batch_size=8,
    #                                       hidden_feats=hidden_feats)
    # 存储所有折的指标
    all_aucs = []
    all_f1s = []
    all_precisions = []
    all_recalls = []
    all_accuracies = []

    test_dataset = load_graphpred_testDataset(dataset_path, bidirected=False, is_classification=True)  # 加载测试集
    feat_dim = 572
    for fold in range(CV_FOLDS):
        # best_model_path = f'best_dhne_model_fold_{fold}.pt'
        best_model_path = f'best_dhne_model_fold_{fold}.pt'
        auc, f1, precision, recall, accuracy = test_best_model(test_dataset, embed_size=feat_dim, model_path=best_model_path, batch_size=1)
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
