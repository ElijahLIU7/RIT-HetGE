import logging
import os
import csv
import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats
from scipy import stats
import seaborn as sns

from time import time
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GATConv, GINConv, GraphConv, HGTConv, SAGEConv, GlobalAttentionPooling
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from protein.pyHGT.conv import HGTConv as HGT
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


class GraphRegressor(nn.Module):
    def __init__(
            self,
            gnn_type,
            num_gnn_layers,
            num_coder_layers,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            dropout=0.,
            activation=None
    ):
        super(GraphRegressor, self).__init__()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.num_coder_layers = num_coder_layers
        self.relations = relations
        self.num_relations = len(relations)

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a

        self.dropout = dropout
        self.activation = activation.casefold()

        self.autoEncoder = AutoEncoder(num_layers=self.num_coder_layers, feat_dim=self.feat_dim,
                                       embed_dim=self.embed_dim, activation=self.activation)

        self.embedder = MuxGNNGraph(
            gnn_type=self.gnn_type,
            num_gnn_layers=self.num_gnn_layers,
            relations=self.relations,
            embed_dim=self.embed_dim,
            dim_a=self.dim_a,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.regressor = Regressor(self.dim_a, self.activation, top_k=20)

    def forward(self, graph):
        feat = graph.ndata['emb'].float()  # message passing only supports float dtypes
        # feat, embed, decoded, (top_k_nodes, top_k_relations, top_k_attention_weights) = self.autoEncoder(graph, feat, self.embedder)
        feat, embed, decoded = self.autoEncoder(feat)
        embeded, attention_weights = self.embedder(graph, embed)
        # Call regressor to get output and top K nodes with their attention weights
        reg_Tm, top_k_weights, top_k_indices = self.regressor(graph, embeded)
        # Use top_k_indices to gather the corresponding attention_weights
        top_k_attention_weights = attention_weights[top_k_indices]
        return reg_Tm, (top_k_indices, top_k_attention_weights, top_k_weights), feat, decoded

    def train_model(
            self,
            train_dataset,
            eval_dataset,
            batch_size=16,
            EPOCHS=50,
            lr=1e-3,
            weight_decay=0.01,
            accum_steps=1,
            num_workers=2,
            loss_delta=10,
            Lambda1=0.1,
            Lambda2=0.1,
            device='cpu',
            model_dir='.model_save/model',
    ):
        self.to(device)

        os.makedirs(model_dir, exist_ok=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        start_train = time()
        train_losses = []
        valid_losses = []
        valid_r2, valid_mae, valid_pcc = [], [], []
        best_loss = float('inf')
        Valid_MSE = 0.
        pcc = 0.
        best_loss_stop = float('inf')
        step_pcc_early_stop = 30  # 早停器
        pcc_early_stop = step_pcc_early_stop
        HuberLoss = nn.HuberLoss(reduction='mean', delta=loss_delta)
        # mseLoss = nn.MSELoss(reduction='mean')

        for epoch in range(EPOCHS):
            self.train()
            self.to(device)

            data_iter = tqdm(
                train_loader,
                desc=f'Epoch {epoch:02}',
                total=len(train_loader),
                position=0
            )

            loss, avg_loss = None, 0.
            for i, (batch_graph, labels, _) in enumerate(data_iter):
                batch_graph = batch_graph.to(device)
                labels = labels.squeeze().to(device)

                preds, _, encoded, decoded = self(batch_graph)#, encoded, decoded
                preds = preds.squeeze()
                loss = Lambda1 * HuberLoss(preds, labels) + model_loss(encoded, decoded, Lambda2)
                # Normalize for batch accumulation
                loss /= accum_steps

                loss.backward()
                avg_loss += loss.item() * accum_steps  # Multiply back to get the actual loss
                data_iter.set_postfix({
                    'Avg_train_loss': avg_loss / (i + 1),
                    'Valid_loss': Valid_MSE,
                    'Valid PCC': pcc
                })

                if ((i + 1) % accum_steps == 0) or ((i + 1) == len(data_iter)):
                    optimizer.step()
                    optimizer.zero_grad()
                    # test the model, to plus 'break'

            train_loss = avg_loss / len(data_iter)

            if epoch % 1 == 0:
                Valid_MSE, r2, mae, pcc, name_protein, _ = self.eval_model(
                    eval_dataset,
                    batch_size=1,
                    num_workers=num_workers,
                    device=device)

            logging.info(
                f'{epoch:02}: Train Loss: {train_loss:.4f} | Valid MSE: {Valid_MSE:.4f} | Valid R2: {r2} | Valid MAE: {mae} | Valid PCC: {pcc}')

            train_losses.append(train_loss)
            valid_losses.append(Valid_MSE)
            valid_r2.append(r2)
            valid_mae.append(mae)
            valid_pcc.append(pcc)

            if Valid_MSE < best_loss:
                best_loss = Valid_MSE
                best_metrics = (r2, mae, pcc)
                torch.save({
                    'epoch': epoch,
                    'loss': best_loss,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': best_metrics
                }, f'{model_dir}/bestACC_checkpoint.pt')
                pcc_early_stop = step_pcc_early_stop
            else:
                pcc_early_stop -= 1
            # if train_loss <= best_loss_stop:
            #     best_loss_stop = train_loss
            #     pcc_early_stop = step_pcc_early_stop
            # else:
            #     pcc_early_stop -= 1
            if pcc_early_stop == 0:
                print(f"Early stopping at epoch {epoch + 1}/{EPOCHS}")
                break

            torch.save({
                'epoch': epoch,
                'loss': Valid_MSE,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': (r2, mae, pcc)
            }, f'{model_dir}/checkpoint.pt')

            torch.cuda.empty_cache()  # 清理显存

        end_train = time()
        logging.info(f'Total training time... {end_train - start_train:.2f}s')

        return train_losses, valid_losses, valid_r2, valid_mae, valid_pcc

    def eval_model(
            self,
            eval_dataset,
            batch_size=16,
            num_workers=2,
            device='cpu',
            Is_test=False,
            model_load='.model_save/model',
            Is_Best_test=False
    ):
        if Is_test:
            if Is_Best_test == True:
                checkpoint = torch.load(f'{model_load}/bestACC_checkpoint.pt')
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                checkpoint = torch.load(f'{model_load}/checkpoint.pt')
                self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()
        self.to(device)

        eval_loader = GraphDataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        preds, labels, name_protein, (top_k_nodes, top_k_relations, top_k_attention_weights) = self.predict(eval_loader,
                                                                                                            device=device)
        r2, mae, pcc = compute_metrics(preds, labels)

        # Convert lists to tensors
        preds_tensor = torch.tensor(preds, device=device)
        labels_tensor = torch.tensor(labels, device=device)

        mseLoss = nn.MSELoss(reduction='mean')
        MSE = mseLoss(preds_tensor, labels_tensor).item()
        # HuberLoss = nn.HuberLoss(reduction='mean', delta=1.0)
        # eval_loss = HuberLoss(preds_tensor, labels_tensor).item()

        if Is_test:
            # 输出每个蛋白质的真实标签和预测标签
            protein_results = list(zip(name_protein, labels, preds))

            # 打开一个 CSV 文件并设置写入模式
            output_dir = model_load
            os.makedirs(output_dir, exist_ok=True)
            # 收集预测值和真实值
            all_preds = []
            all_labels = []

            with open('protein_results.csv', mode='w', newline='') as file:
                writer = csv.writer(file)

                # 写入CSV文件的表头
                writer.writerow(['Protein', 'True Tm', 'Predicted Tm'])

                # 遍历 protein_results，将数据写入 CSV
                for name, true_label, pred_label in protein_results:
                    # 检查是否有结束符 \x00，并去除
                    if name[-1] == '\x00':
                        name = name[:-4]

                    # 打印输出
                    print(f'Protein: {name}, True Tm: {true_label}, Predicted Tm: {pred_label}')

                    all_preds.append(pred_label.cpu().numpy().flatten())
                    all_labels.append(true_label.numpy().flatten())

                    # 将数据写入CSV文件
                    writer.writerow([name, true_label, pred_label])

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            # 确保输出目录存在
            output_heatmap_dir = os.path.join(model_load, 'heatmaps')
            os.makedirs(output_heatmap_dir, exist_ok=True)
            # 绘制热力图
            plt.figure(figsize=(10, 8))
            sns.set(style='white')
            plt.hexbin(all_labels, all_preds, gridsize=50, cmap='viridis', bins='log')
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

            # 使用seaborn的联合分布图
            g = sns.jointplot(x=all_labels, y=all_preds, kind='hex', color='blue', height=8)
            g.set_axis_labels('Actual Tm', 'Predicted Tm', fontsize=12)
            plt.savefig(os.path.join(output_heatmap_dir, 'heatmap_joint.png'))
            plt.show()

            # 另外可以添加散点图以显示趋势线
            plt.figure(figsize=(8, 6))
            sns.regplot(x=all_labels, y=all_preds, scatter_kws={'alpha': 0.3})
            plt.xlabel('Actual Tm')
            plt.ylabel('Predicted Tm')
            plt.title('Actual vs Predicted Tm with Regression Line')
            plt.savefig(os.path.join(output_heatmap_dir, 'scatter_regression.png'))
            plt.show()
            # 计算残差
            residuals = labels_tensor - preds_tensor

            # 绘制残差图
            plt.figure(figsize=(10, 6))
            plt.scatter(preds_tensor.cpu().numpy(), residuals.cpu().numpy(), alpha=0.5)
            plt.hlines(y=0, xmin=np.min(preds_tensor.cpu().numpy()), xmax=np.max(preds_tensor.cpu().numpy()),
                       colors='r')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Predicted Values')
            plt.show()

            # 绘制Q-Q图
            plt.figure(figsize=(10, 6))
            stats.probplot(residuals.cpu().numpy(), dist="norm", plot=plt)
            plt.title('Q-Q Plot of Residuals')
            plt.show()

        return MSE, r2, mae, pcc, name_protein, (top_k_nodes, top_k_relations, top_k_attention_weights)

    def predict(self, graph_loader, device='cpu'):
        # self.eval()
        # self.to(device)

        data_iter = tqdm(
            graph_loader,
            desc=f'Predict: ',
            total=len(graph_loader),
            position=0
        )

        with torch.no_grad():
            preds, labels, name = [], [], []
            all_top_k_nodes = []
            all_top_k_relations = []
            all_top_k_weight = []

            for batch_graph, batch_labels, batch_name in data_iter:
                batch_graph = batch_graph.to(device)
                batch_labels = batch_labels.squeeze().to(device)

                batch_preds, (top_k_nodes, top_k_relations, top_k_attention_weights), _, _ = self(batch_graph)#, _, _

                all_top_k_nodes.append(top_k_nodes.cpu().tolist())
                all_top_k_relations.append(top_k_relations.cpu().tolist())
                all_top_k_weight.append(top_k_attention_weights.cpu().tolist())

                preds.extend(batch_preds.cpu())
                if batch_labels.dim() != 0:
                    labels.extend(batch_labels.unsqueeze(-1).cpu())
                else:
                    labels.append(batch_labels.unsqueeze(-1).cpu())
                name.extend(batch_name)

        return preds, labels, name, (all_top_k_nodes, all_top_k_relations, all_top_k_weight)


class AutoEncoder(nn.Module):
    def __init__(self, num_layers, feat_dim, embed_dim, activation=None, noise_std=1):
        super(AutoEncoder, self).__init__()
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.activation = activation
        self.noise_std = noise_std  # 保存噪声标准差
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # 自编码器初始化
        tmp = []
        tmp.append(self.feat_dim)
        if self.feat_dim < self.embed_dim:
            self.feat_dim = self.embed_dim
        for i in range(num_layers):
            reduction = round(self.feat_dim - (self.feat_dim - self.embed_dim) / num_layers * (i + 1))
            self.encoder.append(nn.Sequential(
                nn.Linear(tmp[i], reduction), self._get_activation_fn(activation)
            ))
            tmp.append(reduction)
        for j in range(num_layers):
            self.decoder.append(nn.Sequential(
                nn.Linear(tmp[-(j + 1)], tmp[-(j + 2)]), self._get_activation_fn(activation)
            ))
        # for i in range(num_layers):
        #     self.encoder.append(nn.Sequential(
        #         nn.Linear(tmp[i], self.embed_dim), self._get_activation_fn(activation)
        #     ))
        #     tmp.append(self.embed_dim)
        # for j in range(num_layers):
        #     self.decoder.append(nn.Sequential(
        #         nn.Linear(tmp[-(j + 1)], tmp[-(j + 2)]), self._get_activation_fn(activation)
        #     ))


    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'selu':
            act_fn = nn.SELU()
        elif activation == 'softmax':
            # Softmax requires specifying the dimension to apply it to
            act_fn = nn.Softmax(dim=-1)
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, feat):     # (self, graph, feat, embedder):
        h = feat
        # 在初始阶段对每个节点加入高斯白噪声
        noise = torch.randn_like(feat) * self.noise_std
        feat = feat + noise
        for i, layer in enumerate(self.encoder):
            feat = layer(feat)
        encode = feat
        # encode, (top_k_nodes, top_k_relations, top_k_attention_weights) = embedder(graph, feat)
        # feat = encode
        for i, layer in enumerate(self.decoder):
            feat = layer(feat)
        return h, encode, feat     # h, encode, feat, (top_k_nodes, top_k_relations, top_k_attention_weights)


class MuxGNNGraph(nn.Module):
    def __init__(
            self,
            gnn_type,
            num_gnn_layers,
            relations,
            embed_dim,
            dim_a,
            dropout=0.,
            activation=None,
            use_autoencoder=True  # 使用自编码器的标志
    ):
        super(MuxGNNGraph, self).__init__()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.relations = relations
        self.num_relations = len(self.relations)
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.activation = activation
        self.dropout = dropout
        self.use_autoencoder = use_autoencoder

        self.layers = nn.ModuleList(
                [MuxGNNLayer(gnn_type=gnn_type,
                            relations=self.relations,
                            in_dim=self.embed_dim,
                            out_dim=self.dim_a,
                            dim_a=self.dim_a,
                            dropout=self.dropout,
                            activation=self.activation,)]
        )
        for _ in range(1, self.num_gnn_layers):
            self.layers.append(
                MuxGNNLayer(
                    gnn_type=self.gnn_type,
                    relations=self.relations,
                    in_dim=self.dim_a,
                    out_dim=self.dim_a,
                    dim_a=self.dim_a,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )

        self.alpha = nn.Parameter(torch.ones(self.num_gnn_layers - 1))  # 残差连接中可学习的参数 alpha
        # self.readout_fn = AvgPooling()
        # self.gate_nn = nn.Linear(in_features=self.embed_dim, out_features=1)
        # self.readout_fn = GlobalAttentionPooling(self.gate_nn)

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, graph, feat):
        h = feat

        alpha = torch.sigmoid(self.alpha)  # 将 alpha 值约束在[0,1]之间
        for i, layer in enumerate(self.layers):
            if i == self.num_gnn_layers - 1:
                h_new, attention_weights = layer(graph, h, Is_attention=True, Is_last=True)
            else:
                h_new = layer(graph, h, Is_attention=True)
            if i == 0:
                h = h_new
            else:
                # h = h_new + h
                h = alpha[i - 1] * h_new + (1 - alpha[i - 1]) * h

        return h, attention_weights  # self.readout_fn(graph, h)


class MuxGNNLayer(nn.Module):
    def __init__(
            self,
            gnn_type,
            relations,
            in_dim,
            out_dim,
            dim_a,
            dropout=0.,
            activation=None,
            use_autoencoder=True,  # 使用自编码器的标志
    ):
        super(MuxGNNLayer, self).__init__()
        self.gnn_type = gnn_type
        self.relations = relations
        self.num_relations = len(self.relations)
        self.use_autoencoder = use_autoencoder
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.act_str = activation

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(self.act_str)

        if self.gnn_type == 'hgtconv':
            self.num_heads = 8
            self.head_size = self.out_dim / self.num_heads
            self.num_etypes = self.num_relations
            self.num_ntypes = 1
        if self.gnn_type == 'gcn':
            self.gnn = GraphConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                norm='both',
                weight=True,
                bias=True,
                activation=self.activation,
                allow_zero_in_degree=True
            )
        elif self.gnn_type == 'gat':
            self.gnn = GATConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                num_heads=4,
                feat_drop=dropout,
                residual=False,
                activation=self.activation,
                allow_zero_in_degree=True
            )
        elif self.gnn_type == 'gin':
            self.gnn = GINConv(
                apply_func=nn.Sequential(
                    nn.Linear(self.in_dim, self.out_dim),
                    self.dropout,
                    self.activation,
                    nn.Linear(self.out_dim, self.out_dim),
                    self.dropout,
                    self.activation,
                ),
                aggregator_type='mean',
            )
        elif self.gnn_type == 'sage':
            self.gnn = SAGEConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                aggregator_type='pool',
                feat_drop=dropout,
                activation=self.activation
            )
        elif self.gnn_type == 'hgt':
            self.gnn = HGT(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                num_types=1,
                num_relations=self.relations,
                n_heads=2,
                dropout=dropout,
                use_norm=True
            )
        elif self.gnn_type == 'hgtconv':
            self.gnn = HGTConv(
                in_size=self.in_dim,
                head_size=self.head_size,
                num_heads=self.num_heads,
                num_etypes=self.num_etypes,
                num_ntypes=self.num_ntypes,
                dropout=self.dropout,
                use_norm=True
            )

        self.attention = SemanticAttentionBatched(self.num_relations, self.out_dim, self.dim_a, dropout=dropout)

        # self.norm = None
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True)  # 层标准化
        # self.norm = nn.BatchNorm1d(self.num_relations)

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'selu':
            act_fn = nn.SELU()
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, graph, feat, Is_attention=False, Is_last=False):
        h = torch.zeros(self.num_relations, graph.num_nodes(), self.out_dim, device=graph.device)

        with graph.local_scope():
            if self.gnn_type == 'hgtconv':
                h = self.gnn(graph, feat)
            for i, graph_layer in enumerate(self.relations):
                rel_graph = graph['node', graph_layer, 'node']

                if self.gnn_type == 'hgt':
                    source_nodes, destination_nodes, edges = rel_graph.edges(form='all')
                    edge_index = torch.stack((source_nodes, destination_nodes))
                    edge_weights = rel_graph.edges[graph_layer].data['weight']
                    edge_index = torch.cat((edge_index, edge_weights.T), dim=0)
                    node_type = torch.zeros(feat.size(0)).to(feat.device)
                    edge_type = torch.full((edge_index[0].size(0),), i, dtype=torch.long, device=rel_graph.device)
                    edge_time = torch.zeros(edge_index[0].size(0), dtype=torch.long, device=rel_graph.device)
                    h_out = self.gnn(feat, node_type, edge_index, edge_type, edge_time)
                    h_out = self.layer_norm(h_out)
                else:
                    h_out = self.gnn(rel_graph, feat).squeeze()
                    if self.gnn_type == 'gat':
                        h_out = h_out.sum(dim=1)
                    # h_out = self.layer_norm(h_out)
                h[i] = h_out

        if self.norm:
            h = self.norm(h)

        h = self.attention(graph, h, Is_attention=Is_attention)
        if Is_last:
            attention_weights = self.attention.get_top_k_nodes()
            return h, attention_weights
        # h = self.attention(graph, h, Is_attention=Is_attention)

        return h


class Regressor(nn.Module):
    def __init__(self, embed_dim, activation, top_k):
        super(Regressor, self).__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.gate_nn = torch.nn.Linear(embed_dim, 1)
        self.gap = GlobalAttentionPooling(self.gate_nn)
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            self._get_activation_fn(activation),
            nn.Linear(self.embed_dim, 1)
        )

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        elif activation == 'selu':
            act_fn = nn.SELU()
        elif activation == 'softmax':
            # Softmax requires specifying the dimension to apply it to
            act_fn = nn.Softmax(dim=-1)
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, graph, x):
        # Get results and attention weights for Global Attention
        graph_x, attention_weights = self.gap(graph, x, get_attention=True)
        top_k_weights, top_k_indices = torch.topk(attention_weights.squeeze(), self.top_k)

        return self.regressor(graph_x), top_k_weights, top_k_indices


class SemanticAttentionBatched(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, out_dim=1, dropout=0.):
        super(SemanticAttentionBatched, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.out_dim)
        )

        self.reset_parameters()
        self.attention_weights = None  # Dynamic initialisation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('sigmoid')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, graph, h, Is_attention=False, batch_size=512, Is_last=False):
        # Shape of input h: (num_relations, num_nodes, dim)
        # Output shape: (num_nodes, dim)
        graph.ndata['h'] = torch.zeros(graph.num_nodes(), h.size(-1), device=graph.device)

        # Initialize the attention weights tensor
        num_nodes = graph.num_nodes()
        self.attention_weights = torch.zeros((num_nodes, self.num_relations), device=graph.device)

        node_loader = DataLoader(
            graph.nodes(),
            batch_size=batch_size,
            shuffle=False,
        )

        for node_batch in node_loader:
            h_batch = h[:, node_batch, :]
            if Is_attention == True:
                attention = F.softmax(
                    torch.matmul(torch.sigmoid(torch.matmul(h_batch, self.weights_s1)), self.weights_s2),
                    dim=0).squeeze()

                attention = self.dropout(attention)
            else:
                Len = len(node_batch)
                attention = torch.ones((self.num_relations, Len), device=graph.device)

            try:
                graph.ndata['h'][node_batch] = torch.einsum('rb,rbd->bd', attention, h_batch)
                # Store attention weight
                self.attention_weights[node_batch] = attention.transpose(0, 1)
            except RuntimeError:
                graph.ndata['h'][node_batch] = torch.einsum('rb,rbd->bd', attention.unsqueeze(1), h_batch)
                # Store attention weight
                self.attention_weights[node_batch] = attention

        return graph.ndata.pop('h')

    def get_top_k_nodes(self):
        return self.attention_weights


def model_loss(original, decoded, beta=1e-3):
    # Reconstruction error: calculating the mean square error between the original input and the decoder output
    reconstruction_loss = F.mse_loss(decoded, original)
    # Total Loss = Reconstruction Error + Loss of Similarity
    total_loss = beta * reconstruction_loss
    return total_loss


def compute_metrics(preds, labels):
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    preds = preds.squeeze().cpu().numpy()
    labels = labels.squeeze().cpu().numpy()
    r2 = r2_score(labels, preds)
    mae = mean_absolute_error(labels, preds)
    pcc, _ = scipy.stats.pearsonr(labels, preds)
    return r2, mae, pcc
