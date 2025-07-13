import torch as th
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
import logging
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
from dgl.nn.pytorch import GraphConv
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from protein_wang.pyHGT.utils import load_graphpred_dataset, load_graphpred_testDataset
from abc import ABCMeta


# 配置日志
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GeneralLinear(nn.Module):
    r"""
    General Linear, combined with activation, normalization(batch and L2), dropout and so on.

    Parameters
    ------------
    in_features : int
        size of each input sample, which is fed into nn.Linear
    out_features : int
        size of each output sample, which is fed into nn.Linear
    act : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    has_l2norm : bool
        If True, applies torch.nn.functional.normalize to the node features at last of forward(). Default: ``True``
    has_bn : bool
        If True, applies torch.nn.BatchNorm1d to the node features after applying nn.Linear.

    """

    def __init__(self, in_features, out_features, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(GeneralLinear, self).__init__()
        self.has_l2norm = has_l2norm
        has_bn = has_bn
        self.layer = nn.Linear(in_features, out_features, bias=not has_bn)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(nn.BatchNorm1d(out_features))
        if dropout > 0:
            layer_wrapper.append(nn.Dropout(p=dropout))
        if act is not None:
            layer_wrapper.append(act)
        self.post_layer = nn.Sequential(*layer_wrapper)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer.weight)

    def forward(self, batch_h: th.Tensor) -> th.Tensor:
        r"""
        Apply Linear, BatchNorm1d, Dropout and normalize(if need).
        """
        batch_h = self.layer(batch_h)
        batch_h = self.post_layer(batch_h)
        if self.has_l2norm:
            batch_h = F.normalize(batch_h, p=2, dim=1)
        return batch_h


class HeteroLinearLayer(nn.Module):
    r"""
    Transform feature with nn.Linear. In general, heterogeneous feature has different dimension as input.
    Even though they may have same dimension, they may have different semantic in every dimension.
    So we use a linear layer for each node type to map all node features to a shared feature space.

    Parameters
    ----------
    linear_dict : dict
        Key of dict can be node type(node name), value of dict is a list contains input dimension and output dimension.
    """

    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, **kwargs):
        super(HeteroLinearLayer, self).__init__()

        self.layer = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            self.layer[name] = GeneralLinear(in_features=linear_dim[0], out_features=linear_dim[1], act=act,
                                             dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

    def forward(self, dict_h: dict) -> dict:
        r"""
        Parameters
        ----------
        dict_h : dict
            A dict of heterogeneous feature

        return dict_h
        """
        # note must set new_h dict, or overwrite dict_h
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layer[name](batch_h)
        return new_h


class HeteroMLPLayer(nn.Module):
    r"""
    HeteroMLPLayer contains multiple GeneralLinears, different with HeteroLinearLayer.
    The latter contains only one layer.

    Parameters
    ----------
    linear_dict : dict
        Key of dict can be node type(node name), value of dict is a list contains input, hidden and output dimension.

    """

    def __init__(self, linear_dict, act=None, dropout=0.0, has_l2norm=True, has_bn=True, final_act=False, **kwargs):
        super(HeteroMLPLayer, self).__init__()
        self.layers = nn.ModuleDict({})
        for name, linear_dim in linear_dict.items():
            nn_list = []
            n_layer = len(linear_dim) - 1
            for i in range(n_layer):
                in_dim = linear_dim[i]
                out_dim = linear_dim[i + 1]
                if i == n_layer - 1:
                    if final_act:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act,
                                              dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                    else:
                        layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=None,
                                              dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)
                else:
                    layer = GeneralLinear(in_features=in_dim, out_features=out_dim, act=act,
                                          dropot=dropout, has_l2norm=has_l2norm, has_bn=has_bn)

                nn_list.append(layer)
            self.layers[name] = nn.Sequential(*nn_list)

    def forward(self, dict_h):
        new_h = {}
        if isinstance(dict_h, dict):
            for name, batch_h in dict_h.items():
                new_h[name] = self.layers[name](batch_h)
        return new_h


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


class NSHE(BaseModel):
    r"""
    NSHE[IJCAI2020]
    Network Schema Preserving Heterogeneous Information Network Embedding
    `Paper Link <http://www.shichuan.org/doc/87.pdf>`
    `Code Link https://github.com/Andy-Border/NSHE`

    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        return cls(hg, 'GCN', project_dim=args.dim_size['project'],
                   emd_dim=args.dim_size['emd'], context_dim=args.dim_size['context'],
                   num_heads=args.num_heads, dropout=args.dropout)

    def __init__(self, g, gnn_model, project_dim, emd_dim, context_dim, num_heads, dropout):
        super(NSHE, self).__init__()
        self.gnn_model = gnn_model
        self.norm_emb = True
        # dimension of transform: after projecting, after aggregation, after CE_encoder
        self.project_dim = g.nodes['node'].data['emb'].shape[1]
        self.emd_dim = emd_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.dropout = dropout

        # * ================== encoder config==================
        linear_dict1 = {}
        linear_dict2 = {}
        linear_dict3 = {}
        cla_dim = self.emd_dim + self.context_dim * (len(g.ntypes) - 1)
        for ntype in g.ntypes:
            in_dim = g.nodes[ntype].data['emb'].shape[1]
            linear_dict1[ntype] = (in_dim, self.project_dim)
            linear_dict2[ntype] = (self.emd_dim, self.context_dim)
            linear_dict3[ntype] = (cla_dim, 1)
        # * ================== Project feature Layer==================
        self.feature_proj = HeteroLinearLayer(linear_dict1, has_l2norm=False, has_bn=False)
        # * ================== Neighborhood Agg(gnn_model)==================
        if self.gnn_model == "GCN":
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, norm="none", activation=F.relu, allow_zero_in_degree=True)
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, norm="none", activation=None, allow_zero_in_degree=True)
        elif self.gnn_model == "GAT":
            self.gnn1 = GraphConv(self.project_dim, self.emd_dim, activation=F.relu, allow_zero_in_degree=True)
            self.gnn2 = GraphConv(self.emd_dim, self.emd_dim, activation=None, allow_zero_in_degree=True)

        # * ================== Context encoder(called CE in the paper)=================
        self.context_encoder = HeteroLinearLayer(linear_dict2, has_l2norm=False, has_bn=False)
        # * ================== NSI Classification================
        self.linear_classifier = HeteroMLPLayer(linear_dict3, has_l2norm=False, has_bn=False)
        self.classifier = nn.Linear(hidden_feats, 1)

    def forward(self, hg, h):
        with hg.local_scope():
            # * =============== Encode heterogeneous feature ================
            h_dict = h      # feature_proj(h)
            # hg.ndata['h_proj'] = h_dict
            # 如果只有一种节点类型，则直接赋值张量
            if len(hg.ntypes) == 1:
                hg.ndata['h_proj'] = list(h_dict.values())[0]
            else:
                hg.ndata['h_proj'] = h_dict
            # hg.ndata['h_proj'] = h_dict
            g_homo = dgl.to_homogeneous(hg, ndata=['h_proj'])
            # * =============== Node Embedding Generation ===================
            h = g_homo.ndata['h_proj']
            #h = self.gnn1(g_homo, h)
            h = self.gnn2(g_homo, h)
            if self.norm_emb:
                # Independently normalize each dimension
                h = F.normalize(h, p=2, dim=1)
            # Context embedding generation
            hg.ndata['h'] = h
            emb = self.h2dict(h, h_dict)

            # 使用全局池化生成图级表示
            graph_representation = dgl.mean_nodes(hg, 'h')
            logits = self.classifier(graph_representation).squeeze(-1)

        return logits, emb, h

    def h2dict(self, h, hdict):
        pre = 0
        for i, value in hdict.items():
            hdict[i] = h[pre:value.shape[0]+pre]
            pre += value.shape[0]
        return hdict


class NSHELayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_heads, rel_names, activation=None, dropout=0.0):
        super(NSHELayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # 定义异构图卷积
        self.conv = dgl.nn.HeteroGraphConv({
            rel: dgl.nn.pytorch.GATConv(in_feat, out_feat // num_heads, num_heads)
            for rel in rel_names
        })

        # 自环权重
        self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        # 偏置
        self.h_bias = nn.Parameter(th.Tensor(out_feat))
        nn.init.zeros_(self.h_bias)

    def forward(self, g, inputs):
        g = g.local_var()

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs_src)

        def _apply(ntype, h):
            h = h + th.matmul(inputs_dst[ntype], self.loop_weight)
            h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


def train_nshe_model(train_dataset, valid_dataset, embed_size, epochs=50, lr=0.001, batch_size=8, fold=None):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # 初始化 NSHE 模型，使用 train_dataset 中的异构图来初始化
    sample_graph = train_dataset[0][0]  # 使用训练集中的一个图来获取异构图结构
    nshe = NSHE(sample_graph, 'GCN', project_dim=embed_size, emd_dim=embed_size, context_dim=embed_size, num_heads=4, dropout=0.2).to(device)

    # 优化器
    optimizer = optim.Adam(nshe.parameters(), lr=lr)

    # 损失函数
    regression_loss = nn.MSELoss()

    # 数据加载器
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_pcc = 0
    step_pcc_early_stop = 10  # 早停器
    pcc_early_stop = step_pcc_early_stop
    best_model_path = f'best_model_fold_{fold}.pt' if fold is not None else 'best_model.pt'

    for epoch in range(epochs):
        nshe.train()
        total_loss = 0
        # 设置进度条
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for i, batch in enumerate(pbar):
                graphs, labels, _ = batch
                graphs, labels = graphs.to(device), labels.to(device)

                # ---------------------
                # 训练模型
                # ---------------------
                optimizer.zero_grad()

                # 前向传播
                h = {ntype: graphs.nodes[ntype].data['emb'].to(device) for ntype in graphs.ntypes}
                hg_embed, _, _ = nshe(graphs, h)

                # 计算损失
                loss = regression_loss(hg_embed, labels)
                total_loss += loss.item()

                # 反向传播和优化
                loss.backward()
                optimizer.step()
                # 更新进度条描述
                pbar.set_postfix({'Loss': total_loss / (i + 1)})
                loss = total_loss / (i + 1)

        # 验证模型
        nshe.eval()
        all_labels = []
        all_preds = []
        with th.no_grad():
            for batch in valid_loader:
                graphs, labels, _ = batch
                graphs, labels = graphs.to(device), labels.to(device)

                h = {ntype: graphs.nodes[ntype].data['emb'].to(device) for ntype in graphs.ntypes}
                hg_embed, _, _ = nshe(graphs, h)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(hg_embed.cpu().numpy())

        # 计算验证集上的损失和指标
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        val_loss = regression_loss(th.tensor(all_preds), th.tensor(all_labels))     # 计算验证集上的损失
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
            th.save(nshe.state_dict(), best_model_path)
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

    return nshe, best_loss, mse, r2, mae, pcc, best_model_path


def test_best_model(test_dataset, embed_size, model_path, batch_size=8):
    """使用测试集评估最优模型"""
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # 初始化 NSHE 模型，使用 train_dataset 中的异构图来初始化
    sample_graph = test_dataset[0][0]  # 使用训练集中的一个图来获取异构图结构
    nshe = NSHE(sample_graph, 'GCN', project_dim=embed_size, emd_dim=embed_size, context_dim=embed_size, num_heads=4, dropout=0.2).to(device)
    nshe.load_state_dict(th.load(model_path))
    nshe.eval()

    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    with th.no_grad():
        for batch in test_loader:
            graphs, labels, _ = batch
            graphs, labels = graphs.to(device), labels.to(device)

            h = {ntype: graphs.nodes[ntype].data['emb'].to(device) for ntype in graphs.ntypes}
            hg_embed, _, _ = nshe(graphs, h)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(hg_embed.cpu().numpy())

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


def cross_validation(dataset_path, CV_FOLDS=10, epochs=50, lr=0.1, batch_size=8, hidden_feats=512, bidirected=False):
    all_losses = []
    feat_dim = None
    relations = None
    best_model_path = None

    for cv_select in range(CV_FOLDS):
        if cv_select == 5:
            break
        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # 加载数据集，根据当前折数选择训练集和验证集
        train_dataset, valid_dataset, feat_dim, relations = load_graphpred_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, bidirected=bidirected
        )

        # 训练模型并记录验证集的损失
        nshe_model, best_loss, mse, r2, mae, pcc, model_path = train_nshe_model(train_dataset, valid_dataset,
                                                                                embed_size=hidden_feats, epochs=epochs,
                                                                                lr=lr, batch_size=batch_size, fold=cv_select)
        all_losses.append(best_loss)

        # 记录验证集中表现最好的模型路径
        if best_model_path is None or best_loss < min(all_losses):
            best_model_path = model_path

    # 打印十折交叉验证的结果
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
    best_model_path, _ = cross_validation(dataset_path, CV_FOLDS=CV_FOLDS, epochs=200, lr=1e-3, batch_size=8,
                                          hidden_feats=hidden_feats, bidirected=False)

    # 存储所有折的指标
    all_mse = []
    all_r2 = []
    all_mae = []
    all_pcc = []

    # 加载测试集并评估最优模型
    test_dataset = load_graphpred_testDataset(dataset_path, bidirected=False)  # 加载测试集
    feat_dim = 572
    for fold in range(CV_FOLDS):
        best_model_path = f'best_model_fold_{fold}.pt'
        mse, r2, mae, pcc= test_best_model(test_dataset, embed_size=feat_dim, model_path=best_model_path, batch_size=8)
        # 保存指标
        all_mse.append(mse)
        all_r2.append(r2)
        all_mae.append(mae)
        all_pcc.append(pcc)
        if fold==4:
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