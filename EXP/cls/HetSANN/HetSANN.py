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
from dgl.nn.pytorch import TypedLinear
from dgl.ops import edge_softmax
import dgl.function as Fn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from HGRIFN.utils import load_dataset, load_testDataset
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


def to_hetero_feat(h, type, name):
    """Feature convert API.

    It uses information about the type of the specified node
    to convert features ``h`` in homogeneous graph into a heteorgeneous
    feature dictionay ``h_dict``.

    Parameters
    ----------
    h: Tensor
        Input features of homogeneous graph
    type: Tensor
        Represent the type of each node or edge with a number.
        It should correspond to the parameter ``name``.
    name: list
        The node or edge types list.

    Return
    ------
    h_dict: dict
        output feature dictionary of heterogeneous graph
    """
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[th.where(type == index)]

    return h_dict


class HetSANN(BaseModel):
    r"""
    This is a model HetSANN from `An Attention-Based Graph Neural Network for Heterogeneous Structural Learning
    <https://arxiv.org/abs/1912.10832>`__

    It contains the following part:

    Apply a linear transformation:

    .. math::
       h^{(l+1, m)}_{\phi(j),i} = W^{(l+1, m)}_{\phi(j),\phi(i)} h^{(l)}_i \quad (1)

    And return the new embeddings.

    You may refer to the paper HetSANN-Section 2.1-Type-aware Attention Layer-(1)

    Aggregation of Neighborhood:

    Computing the attention coefficient:

    .. math::
       o^{(l+1,m)}_e = \sigma(f^{(l+1,m)}_r(h^{(l+1, m)}_{\phi(j),j}, h^{(l+1, m)}_{\phi(j),i})) \quad (2)

    .. math::
       f^{(l+1,m)}_r(e) = [h^{(l+1, m)^T}_{\phi(j),j}||h^{(l+1, m)^T}_{\phi(j),i}]a^{(l+1, m)}_r ] \quad (3)

    .. math::
       \alpha^{(l+1,m)}_e = exp(o^{(l+1,m)}_e) / \sum_{k\in \varepsilon_j} exp(o^{(l+1,m)}_k) \quad (4)

    Getting new embeddings with multi-head and residual

    .. math::
       h^{(l + 1, m)}_j = \sigma(\sum_{e = (i,j,r)\in \varepsilon_j} \alpha^{(l+1,m)}_e h^{(l+1, m)}_{\phi(j),i}) \quad (5)

    Multi-heads:

    .. math::
       h^{(l+1)}_j = \parallel^M_{m = 1}h^{(l + 1, m)}_j \quad (6)

    Residual:

    .. math::
       h^{(l+1)}_j = h^{(l)}_j + \parallel^M_{m = 1}h^{(l + 1, m)}_j \quad (7)

    Parameters
    ----------
    num_heads: int
        the number of heads in the attention computing
    num_layers: int
        the number of layers we used in the computing
    in_dim: int
        the input dimension
    num_classes: int
        the number of the output classes
    num_etypes: int
        the number of the edge types
    dropout: float
        the dropout rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    ntype: list
        the list of node type
    """

    def __init__(self, num_heads, num_layers, in_dim, num_classes,
                 ntypes, num_etypes, dropout, negative_slope, residual):
        super(HetSANN, self).__init__()
        self.num_layers = num_layers
        self.ntypes = ntypes
        # self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.activation = F.elu
        self.het_layers = nn.ModuleList()

        # input projection
        self.het_layers.append(
            HetSANNConv(
                num_heads,
                in_dim,
                in_dim // num_heads,
                num_etypes,
                dropout,
                negative_slope,
                False,
                self.activation,
            )
        )

        # hidden layer
        for i in range(1, num_layers - 1):
            self.het_layers.append(
                HetSANNConv(
                    num_heads,
                    in_dim,
                    in_dim // num_heads,
                    num_etypes,
                    dropout,
                    negative_slope,
                    residual,
                    self.activation
                )
            )

        # output projection
        self.het_layers.append(
            HetSANNConv(
                1,
                in_dim,
                num_classes,
                num_etypes,
                dropout,
                negative_slope,
                residual,
                None,
            )
        )

        # linear layer to aggregate graph features to 1 dimension
        # self.classifier = nn.Linear(hidden_feats, num_classes)

    def forward(self, hg, h_dict):
        """
        The forward part of the HetSANN.

        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types

        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        if hasattr(hg, 'ntypes'):
            with hg.local_scope():
                # input layer and hidden layers
                # hg.ndata['h'] = h_dict
                # 如果只有一种节点类型，则直接赋值张量
                if len(hg.ntypes) == 1:
                    hg.ndata['h'] = list(h_dict.values())[0]
                else:
                    hg.ndata['h'] = h_dict
                g = dgl.to_homogeneous(hg, ndata='h')
                h = g.ndata['h']
                for i in range(self.num_layers - 1):
                    h = self.het_layers[i](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)

                # output layer
                h = self.het_layers[-1](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True)

                # 使用全局池化生成图级表示
                hg.ndata['h'] = h
                graph_representation = dgl.mean_nodes(hg, 'h')

                # h_dict = to_hetero_feat(h, g.ndata['_TYPE'], self.ntypes)

        else:
            # for minibatch training, input h_dict is a tensor
            h = h_dict
            for layer, block in zip(self.het_layers, hg):
                h = layer(block, h, block.ndata['_TYPE']['_N'], block.edata['_TYPE'], presorted=False)
            h_dict = to_hetero_feat(h, block.ndata['_TYPE']['_N'][:block.num_dst_nodes()], self.ntypes)

        # logits = self.classifier(graph_representation).squeeze(-1)
        return graph_representation


class HetSANNConv(nn.Module):
    """
    The HetSANN convolution layer.

    Parameters
    ----------
    num_heads: int
        the number of heads in the attention computing
    in_dim: int
        the input dimension of the features
    hidden_dim: int
        the hidden dimension of the features
    num_etypes: int
        the number of the edge types
    dropout: float
        the dropout rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    """

    def __init__(self, num_heads, in_dim, hidden_dim, num_etypes,
                 dropout, negative_slope, residual, activation):
        super(HetSANNConv, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.W = TypedLinear(self.in_dim, self.hidden_dim * self.num_heads, num_etypes)
        # self.W_out = TypedLinear(hidden_dim * num_heads, num_classes, num_etypes)

        # self.W_hidden = nn.ModuleDict()
        # self.W_out = nn.ModuleDict()

        # for etype in etypes:
        #     self.W_hidden[etype] = nn.Linear(in_dim, hidden_dim * num_heads)

        # for etype in etypes:
        #     self.W_out[etype] = nn.Linear(hidden_dim * num_heads, num_classes)

        self.a_l = TypedLinear(self.hidden_dim * self.num_heads, self.hidden_dim * self.num_heads, num_etypes)
        self.a_r = TypedLinear(self.hidden_dim * self.num_heads, self.hidden_dim * self.num_heads, num_etypes)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        if residual:
            self.residual = nn.Linear(in_dim, self.hidden_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.activation = activation

    def forward(self, g, x, ntype, etype, presorted=False):
        """
        The forward part of the HetSANNConv.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        x: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``

        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        # formula (1)
        g.srcdata['h'] = x
        g.apply_edges(Fn.copy_u('h', 'm'))
        h = g.edata['m']
        feat = self.W(h, etype, presorted)
        h = self.dropout(feat)
        g.edata['m'] = h
        h = h.view(-1, self.num_heads, self.hidden_dim)

        # formula (2) (3) (4)
        h_l = self.a_l(h.view(-1, self.num_heads * self.hidden_dim), etype, presorted) \
            .view(-1, self.num_heads, self.hidden_dim).sum(dim=-1)

        h_r = self.a_r(h.view(-1, self.num_heads * self.hidden_dim), etype, presorted) \
            .view(-1, self.num_heads, self.hidden_dim).sum(dim=-1)

        attention = self.leakyrelu(h_l + h_r)
        attention = edge_softmax(g, attention)

        # formula (5) (6)
        with g.local_scope():
            h = h.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = h @ attention.reshape(-1, self.num_heads, 1)

            g.update_all(Fn.copy_e('m', 'w'), Fn.sum('w', 'emb'))
            h_output = g.dstdata['emb']
            # h_prime = []
            # h = h.permute(1, 0, 2).contiguous()
            # for i in range(self.num_heads):
            #     g.edata['alpha'] = attention[:, i]
            #     g.srcdata.update({'emb': h[i]})
            #     g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
            #                  Fn.sum('m', 'emb'))
            #     h_prime.append(g.ndata['emb'])
            # h_output = torch.cat(h_prime, dim=1)

        # formula (7)
        if g.is_block:
            x = x[:g.num_dst_nodes()]
        if self.residual:
            res = self.residual(x)
            h_output += res

        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output


def train_hetsann_model(train_dataset, valid_dataset, embed_size, epochs=50, lr=0.001, batch_size=8, fold=None):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # Initialize HetSANN model using a heterogeneous graph from train_dataset for initialization
    sample_graph = train_dataset[0][0]  # Use a graph from the training set to obtain the heterogeneous structure
    hetsann = HetSANN(num_heads=4, num_layers=3, in_dim=embed_size, num_classes=1, ntypes=sample_graph.ntypes,
                      num_etypes=len(sample_graph.etypes), dropout=0.2, negative_slope=0.2, residual=True).to(device)

    # 优化器
    optimizer = optim.Adam(hetsann.parameters(), lr=lr)

    # 损失函数
    classification_loss = nn.BCEWithLogitsLoss()

    # 数据加载器
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_accuracy = 0
    best_model_path = f'best_model_fold_{fold}.pt' if fold is not None else 'best_model.pt'

    for epoch in range(epochs):
        hetsann.train()
        total_loss = 0
        # Set progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for i, batch in enumerate(pbar):
                graphs, labels, _ = batch
                graphs, labels = graphs.to(device), labels.to(device)

                # Train the model
                optimizer.zero_grad()

                # Forward propagation
                h_dict = {ntype: graphs.nodes[ntype].data['emb'].to(device) for ntype in graphs.ntypes}
                graph_embed = hetsann(graphs, h_dict)

                # 计算损失
                loss = classification_loss(graph_embed.squeeze(), labels.float())
                total_loss += loss.item()

                # Backward propagation and optimization
                loss.backward()
                optimizer.step()

                # 更新进度条描述
                pbar.set_postfix({'Loss': total_loss / (i + 1)})
                loss = total_loss / (i + 1)

        # Validation
        hetsann.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        with th.no_grad():
            for batch in valid_loader:
                graphs, labels, _ = batch
                graphs, labels = graphs.to(device), labels.to(device)

                h_dict = {ntype: graphs.nodes[ntype].data['emb'].to(device) for ntype in graphs.ntypes}
                graph_embed = hetsann(graphs, h_dict)
                predicted = (graph_embed > 0.5).int()  # 预测为1的阈值

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(graph_embed.cpu().numpy())

        # 计算各项指标
        auc, f1, precision, recall, accuracy = evaluate_performance(all_labels, all_preds, all_probs)

        # 保存验证集上表现最好的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            th.save(hetsann.state_dict(), best_model_path)

        log_metrics(epoch + 1, total_loss, auc, f1, precision, recall, accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Validation AUC: {auc:.4f}, '
              f'F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
    print("Training completed.")

    return hetsann, accuracy, auc, f1, precision, recall, best_model_path


def test_best_hetsann_model(test_dataset, embed_size, model_path, batch_size=8):
    """Using the test set to evaluate the best HetSANN model"""
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    # Initialize HetSANN model using a heterogeneous graph from test_dataset for initialization
    sample_graph = test_dataset[0][0]  # Use a graph from the test set to obtain the heterogeneous structure
    hetsann = HetSANN(num_heads=4, num_layers=3, in_dim=embed_size, num_classes=1, ntypes=sample_graph.ntypes,
                      num_etypes=len(sample_graph.etypes), dropout=0.2, negative_slope=0.2, residual=True).to(device)
    hetsann.load_state_dict(th.load(model_path))
    hetsann.eval()

    test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_labels = []
    all_preds = []
    all_probs = []
    with th.no_grad():
        for batch in test_loader:
            graphs, labels, _ = batch
            graphs, labels = graphs.to(device), labels.to(device)

            h_dict = {ntype: graphs.nodes[ntype].data['emb'].to(device) for ntype in graphs.ntypes}
            graph_embed = hetsann(graphs, h_dict)
            predicted = (graph_embed > 0.5).int()  # 预测为1的阈值

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(graph_embed.cpu().numpy())

        # 计算各项指标
        auc, f1, precision, recall, accuracy = evaluate_performance(all_labels, all_preds, all_probs)
        print(
            f'Used {model_path[:-3]}. Test AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')
        logging.info(
            f'Used {model_path[:-3]}. Test Results - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Accuracy: {accuracy:.4f}')

    return auc, f1, precision, recall, accuracy


def cross_validation_hetsann(dataset_path, CV_FOLDS=10, epochs=50, lr=0.1, batch_size=8, hidden_feats=512, bidirected=False):
    all_losses = []
    feat_dim = None
    relations = None
    best_model_path = None

    for cv_select in range(CV_FOLDS):
        if cv_select == 5:
            break
        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # Load dataset and select train and validation sets for the current fold
        train_dataset, valid_dataset, feat_dim, relations = load_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, bidirected=bidirected, is_classification=True
        )

        # Train the model and record the loss on the validation set
        hetsann_model, accuracy, auc, f1, precision, recall, model_path = train_hetsann_model(train_dataset, valid_dataset,
                                                                                embed_size=hidden_feats, epochs=epochs,
                                                                                lr=lr, batch_size=batch_size, fold=cv_select)
        all_accuracies.append(accuracy)

        # 记录验证集中表现最好的模型路径
        if best_model_path is None or auc > max([auc for auc in all_accuracies]):
            best_model_path = model_path

    # Print the results of 10-fold cross-validation
    mean_loss = sum(all_losses) / 5     # CV_FOLDS
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
    # 存储所有折的指标
    all_aucs = []
    all_f1s = []
    all_precisions = []
    all_recalls = []
    all_accuracies = []

    best_model_path, _ = cross_validation_hetsann(dataset_path, CV_FOLDS=CV_FOLDS, epochs=60, lr=1e-3, batch_size=8,
                                          hidden_feats=hidden_feats, bidirected=False)

    # 加载测试集并评估最优模型
    test_dataset = load_testDataset(dataset_path, bidirected=False, is_classification=True)  # 加载测试集
    feat_dim = 572
    for fold in range(CV_FOLDS):
        best_model_path = f'best_model_fold_{fold}.pt'
        auc, f1, precision, recall, accuracy = test_best_hetsann_model(test_dataset, embed_size=feat_dim, model_path=best_model_path, batch_size=8)
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
