import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import dgl
import logging
import os
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from abc import ABCMeta
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from protein_wang.pyHGT.utils import load_graphpred_dataset, load_graphpred_testDataset
from dgl.nn.pytorch import GlobalAttentionPooling


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
            # hg = GlobalAttentionPooling(g)
            hg = dgl.mean_nodes(g, 'h')  # 也可以使用 sum 或 max 来进行全局池化

        output = self.output_layer.to(self.device)(hg)
        graph_embed = self.mlp.to(self.device)(hg)

        return decodeds + [output, graph_embed]


def train_dhne_model(train_dataset, valid_dataset, embed_size, epochs=50, lr=0.001, batch_size=8, fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize DHNE model
    dhne = DHNE(nums_type=572, dim_features=embed_size, embedding_sizes=64,
                hidden_size=128, device=device).to(device)

    # Optimizer
    optimizer = optim.Adam(dhne.parameters(), lr=lr)

    # Loss function
    regression_loss = nn.MSELoss()

    # Data loaders
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_pcc = 0
    step_pcc_early_stop = 10  # 早停器
    pcc_early_stop = step_pcc_early_stop
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
                output = dhne(input_ids)
                graph_embed = output[-1]

                # Compute loss
                loss = regression_loss(graph_embed.squeeze(), labels.float())
                total_loss += loss.item()

                # Backward propagation and optimization
                loss.backward()
                optimizer.step()

                # Update progress bar description
                pbar.set_postfix({'Loss': total_loss / (i + 1)})

        # Validation
        dhne.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in valid_loader:
                input_ids, labels, _ = batch
                input_ids, labels = input_ids.to(device), labels.to(device)

                output = dhne(input_ids)
                graph_embed = output[-1].squeeze(-1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(graph_embed.squeeze().cpu().numpy())

        # Compute validation loss and metrics
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        val_loss = np.mean((np.array(all_labels) - np.array(all_preds)) ** 2)  # Validation set loss
        # Compute MSE
        mse = mean_squared_error(all_labels, all_preds)

        # Compute R²
        r2 = r2_score(all_labels, all_preds)

        # Compute MAE
        mae = mean_absolute_error(all_labels, all_preds)

        # Compute PCC (Pearson correlation coefficient)
        pcc = np.corrcoef(all_labels, all_preds)[0, 1]

        # Save model with best validation performance
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(dhne.state_dict(), best_model_path)
        if pcc >= best_pcc:
            best_pcc = pcc
            pcc_early_stop = step_pcc_early_stop
        else:
            pcc_early_stop -= 1
        if pcc_early_stop == 0:
            print(f"Early stopping at epoch {epoch + 1}/{epochs}")
            break

        log_metrics(epoch + 1, loss, mse, r2, mae, pcc)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation MSE: {mse:.4f}, '
              f'R²: {r2:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}')
    print("Training completed.")

    return dhne, best_loss, mse, r2, mae, pcc, best_model_path


def test_best_model(test_dataset, embed_size, model_path, batch_size=8):
    """使用测试集评估最优模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dhne = DHNE(nums_type=572, dim_features=embed_size, embedding_sizes=64,
                hidden_size=128, device=device).to(device)
    dhne.load_state_dict(torch.load(model_path))
    dhne.eval()

    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    np_preds = []
    np_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels, _ = batch
            input_ids, labels = input_ids.to(device), labels.to(device)

            output = dhne(input_ids)
            graph_embed = output[-1].squeeze(-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(graph_embed.squeeze().cpu().numpy())
            np_labels.append(labels.cpu().numpy().flatten())
            np_preds.append(graph_embed.cpu().numpy().flatten())

    # 将预测值和实际值转换为 NumPy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    np_preds = np.concatenate(np_preds)
    np_labels = np.concatenate(np_labels)

    # 确保输出目录存在
    output_heatmap_dir = os.path.join('heatmaps')
    os.makedirs(output_heatmap_dir, exist_ok=True)
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.set(style='white')
    plt.hexbin(np_labels, np_preds, gridsize=50, cmap='viridis', bins='log')
    plt.colorbar(label='log10(count)')

    # 获取当前坐标轴范围
    ax = plt.gca()
    x_min, _ = ax.get_xlim()  # 自动获取 x 轴范围
    y_min, _ = ax.get_ylim()  # 自动获取 y 轴范围

    # 确保 y=x 直线覆盖整个视图范围（取 x 和 y 的并集）
    line_min = min(x_min, y_min)

    # 绘制 y=x 直线（红色虚线）
    plt.axline([line_min, line_min], slope=1, color='r', linestyle='--', linewidth=2, label='y = x')

    plt.xlabel('Actual Tm')
    plt.ylabel('Predicted Tm')
    plt.title('Actual vs Predicted Tm Heatmap')
    plt.savefig(os.path.join(output_heatmap_dir, 'heatmap_hexbin.png'))
    plt.show()

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


def cross_validation_dhne(dataset_path, CV_FOLDS=10, epochs=50, lr=0.001, batch_size=8, hidden_feats=128):
    all_losses = []
    feat_dim = None
    best_model_path = None

    for cv_select in range(CV_FOLDS):
        # if cv_select == 0:
        #     continue
        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # Load dataset and select train and validation sets for the current fold
        train_dataset, valid_dataset, feat_dim, relations = load_graphpred_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select
        )

        # Train the model and record the loss on the validation set
        dhne_model, best_loss, mse, r2, mae, pcc, model_path = train_dhne_model(train_dataset, valid_dataset,
                                                                                embed_size=hidden_feats, epochs=epochs,
                                                                                lr=lr, batch_size=batch_size, fold=cv_select)
        all_losses.append(best_loss)

        # Record the best model path based on validation set performance
        if best_model_path is None or best_loss < min(all_losses):
            best_model_path = model_path

    # Print the results of 10-fold cross-validation
    mean_loss = sum(all_losses) / CV_FOLDS
    logging.info(f"\n10-Fold Cross-Validation Mean Loss: {mean_loss:.4f}")
    print(f"\n10-Fold Cross-Validation Mean Loss: {mean_loss:.4f}")

    return best_model_path, mean_loss


def log_metrics(epoch, total_loss, mse, r2, mae, pcc):
    """记录每个epoch的结果"""
    logging.info(f"Epoch {epoch}: Loss = {total_loss:.4f}, MSE = {mse:.4f}, R2 = {r2:.4f}, "
                 f"MAE = {mae:.4f}, PCC = {pcc:.4f}")


if '__main__' == __name__:
    # 运行十折交叉验证
    dataset_path = 'D:/program/GitHub/protein_wang/data/output_Fold_regression'
    hidden_feats = 572
    CV_FOLDS = 10
    best_model_path, _ = cross_validation_dhne(dataset_path, CV_FOLDS=CV_FOLDS, epochs=200, lr=1e-3, batch_size=8,
                                          hidden_feats=hidden_feats)
    # 存储所有折的指标
    all_mse = []
    all_r2 = []
    all_mae = []
    all_pcc = []

    # 加载测试集并评估最优模型
    test_dataset = load_graphpred_testDataset(dataset_path)  # 加载测试集
    feat_dim = 572
    for fold in range(CV_FOLDS):
        best_model_path = f'best_dhne_model_fold_{fold}_last.pt'
        mse, r2, mae, pcc = test_best_model(test_dataset, embed_size=feat_dim, model_path=best_model_path, batch_size=8)
        # 保存指标
        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)
        # if fold == 4:
        #     break

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
