import csv
import logging
import os
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

from time import time
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import GraphConv, GlobalAttentionPooling
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class GraphClassifier(nn.Module):
    def __init__(
            self,
            num_gnn_layers=1,
            num_coder_layers=1,
            relations=6,
            feat_dim=572,
            embed_dim=32,
            dim_a=6,
            dropout=0.,
            activation=None
    ):
        super(GraphClassifier, self).__init__()
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
        self.classifier = BinaryClassifier(self.embed_dim, top_k=20)

    def forward(self, graph):
        feat = graph.ndata['emb'].float()  # message passing only supports float dtypes
        # embed, (top_k_nodes, top_k_relations) = self.embedder(graph, feat)
        feat, embed, decoded = self.autoEncoder(feat)
        embeded, attention_weights = self.embedder(graph, embed)
        class_Tm, top_k_weights, top_k_indices, graph_x = self.classifier(graph, embeded)
        top_k_attention_weights = attention_weights[top_k_indices]
        return class_Tm, (top_k_indices, top_k_attention_weights, top_k_weights), graph_x, feat, decoded

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
            Lambda1=0.1,
            Lambda2=0.9,
            device='cpu',
            model_dir='.model_save/model',
            Is_Best_test=False
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
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-8)

        start_train = time()
        test_acc, test_p, test_r, test_f1, test_auc = [], [], [], [], []
        valid_score = 0
        best_acc = 0
        best_loss_stop = float('inf')
        step_pcc_early_stop = 5  # 早停器
        pcc_early_stop = step_pcc_early_stop

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

                logits, _, _, encoded, decoded = self(batch_graph)
                logits = logits.squeeze()
                loss = Lambda1 * F.binary_cross_entropy_with_logits(logits, labels) + model_loss(encoded, decoded, Lambda2)

                # Normalize for batch accumulation
                loss /= accum_steps

                loss.backward()

                if ((i + 1) % accum_steps == 0) or ((i + 1) == len(data_iter)):
                    optimizer.step()
                    optimizer.zero_grad()

                avg_loss += loss.item()

                data_iter.set_postfix({
                    'valid_score': valid_score,
                    'avg_loss': avg_loss / (i + 1)
                })
            train_loss = avg_loss / len(data_iter)

            if epoch % 2 == 0:
                t_acc, t_p, t_r, t_f1, t_auc, name_protein, _ = self.eval_model(
                    eval_dataset, batch_size=1, num_workers=num_workers, device=device, Is_test=False, model_load=model_dir
                )
                valid_score = t_acc

                logging.info(
                    f'{epoch:02}: Valid Acc: {t_acc:.4f} | Valid Prec: {t_p:.4f} | Valid Recall: {t_r:.4f} | Valid F1: {t_f1:.4f} | Valid AUC: {t_auc:.4f}')

                test_acc.append(t_acc)
                test_p.append(t_p)
                test_r.append(t_r)
                test_f1.append(t_f1)
                test_auc.append(t_auc)

                if t_acc >= best_acc:
                    best_acc = t_acc
                    best_metrics = (t_acc, t_p, t_r, t_f1, t_auc)
                    torch.save({
                        'epoch': epoch,
                        'loss': loss,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': best_metrics
                    }, f'{model_dir}/bestACC_checkpoint.pt')
                    pcc_early_stop = step_pcc_early_stop
                else:
                    pcc_early_stop -= 1
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f'{model_dir}/checkpoint.pt')
                if train_loss <= best_loss_stop:
                    best_loss_stop = train_loss
                if pcc_early_stop == 0:
                    print(f"Early stopping at epoch {epoch + 1}/{EPOCHS}")
                    break

                torch.cuda.empty_cache()  # 清理显存

        end_train = time()
        logging.info(f'Total training time... {end_train - start_train:.2f}s')

        return test_acc, test_p, test_r, test_f1

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
                checkpoint = torch.load(f'{model_load}/BestACC_checkpoint.pt')
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

        pred_probs, y_true, name_protein, (top_k_nodes, top_k_relations, top_k_attention_weights), embeddings = self.predict(eval_loader, device=device)

        y_pred = [
            1 if pred > 0.5 else 0
            for pred in pred_probs
        ]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, pred_probs)
        # Assuming y_true and y_pred are defined
        cm = confusion_matrix(y_true, y_pred)
        print(cm)

        self.plot_tsne(embeddings, y_true, y_pred)

        if Is_test:
            # 输出每个蛋白质的真实标签和预测标签
            protein_results = list(zip(name_protein, y_true, y_pred))

            # 打开一个 CSV 文件并设置写入模式
            output_dir = model_load
            os.makedirs(output_dir, exist_ok=True)
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

                    # 将数据写入CSV文件
                    writer.writerow([name, true_label, pred_label])

            # Tm = pd.read_csv(f'F:/dataset/protein/test2_dataset.csv')
            # Tm['Protein_ID'] = Tm['Protein_ID'].apply(lambda x: re.split('_|-', x)[0])
            # Tm_dict = pd.Series(Tm.Tm.values, index=Tm.Protein_ID).to_dict()
            #
            # total_errors = 0
            # Tm_55_65_errors = 0
            # Tm_55_65_correct = 0
            # all = 0
            #
            # # 用于绘制图像的列表
            # correct_predictions = []
            # incorrect_predictions = []
            #
            # for name, true_label, pred_label in protein_results:
            #     if name[-1] == '\x00':
            #         name = name[:-4]
            #     print(f'Protein: {name}, True Label: {true_label}, Predicted Label: {pred_label}, Tm: {Tm_dict[name]}')
            #     if 50 <= Tm_dict[name] < 70:
            #         all += 1
            #     if pred_label == 1:
            #         correct_predictions.append((name, Tm_dict.get(name, 0)))
            #     else:
            #         incorrect_predictions.append((name, Tm_dict.get(name, 0)))
            #     if true_label != pred_label:
            #         total_errors += 1
            #         if 50 <= Tm_dict[name] <= 70:
            #             Tm_55_65_errors += 1
            #     else:
            #         if 50 <= Tm_dict[name] <= 70:
            #             Tm_55_65_correct += 1
            #
            # # 绘制图像
            # plt.figure(figsize=(12, 6))
            # plt.title('Predictions vs. True Labels')
            # plt.xlabel('Protein')
            # plt.ylabel('Temperature (Tm)')
            #
            # # 将正确和错误预测的节点合并成一个列表
            # all_points = correct_predictions + incorrect_predictions
            #
            # # 随机选取节点进行绘制
            # if all_points:
            #     tmp = random.sample(all_points, len(all_points))
            #     names, temps = zip(*tmp)
            #     colors = ['red' if point in correct_predictions else 'blue' for point in tmp]
            #     plt.scatter(names, temps, color=colors, marker='o')
            #
            # # 绘制温度虚线
            # plt.axhline(y=60, color='gray', linestyle='--', label='Tm = 60°C')
            # plt.axhline(y=70, color='green', linestyle='--', label='Tm = 70°C')
            # plt.axhline(y=50, color='green', linestyle='--', label='Tm = 50°C')
            #
            # # plt.xticks(protation=45)
            # plt.xticks([])
            # plt.legend()
            # plt.tight_layout()
            # plt.show()
            #
            # if total_errors > 0:
            #     error_ratio = Tm_55_65_errors / total_errors
            #     correct_ratio = Tm_55_65_correct / all
            #     print(f'Ratio of errors/allErrors with Tm in [50, 70]: {error_ratio:.2f}')
            #     print(f'Ratio of correct/allInSection with Tm in [50, 70]: {correct_ratio:.2f}')
            # else:
            #     print('No errors found.')

        return accuracy, precision, recall, f1, auc, name_protein, (top_k_nodes, top_k_relations, top_k_attention_weights)

    def predict(self, graph_loader, device='cpu'):
        self.eval()
        self.to(device)

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
            all_embeddings = []  # To store graph embeddings

            for batch_graph, batch_labels, batch_name in data_iter:
                batch_graph = batch_graph.to(device)

                batch_preds, (top_k_nodes, top_k_relations, top_k_attention_weights), embed, _, _ = self(batch_graph)
                batch_preds = torch.sigmoid(batch_preds)

                all_top_k_nodes.append(top_k_nodes.cpu().tolist())
                all_top_k_relations.append(top_k_relations.cpu().tolist())
                all_top_k_weight.append(top_k_attention_weights.cpu().tolist())
                all_embeddings.append(embed.cpu().numpy())

                preds.extend(batch_preds.cpu())
                if batch_labels.dim() != 0:
                    labels.extend(batch_labels.unsqueeze(-1).cpu())
                else:
                    labels.append(batch_labels.unsqueeze(-1).cpu())
                name.extend(batch_name)
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        return preds, labels, name, (all_top_k_nodes, all_top_k_relations, all_top_k_weight), all_embeddings

    def plot_tsne(self, embeddings, true_labels, pred_labels):
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

        # # Plot true labels
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        # sns.scatterplot(
        #     x='Dimension 1', y='Dimension 2',
        #     hue='True Label', palette='viridis',
        #     data=df,
        #     legend='full',
        #     alpha=0.7
        # )
        # plt.title('t-SNE Visualization (True Labels)')

        # # Plot predicted labels
        # plt.subplot(1, 2, 2)
        sns.scatterplot(
            x='Dimension 1', y='Dimension 2',
            hue='Predicted Label', palette='viridis',
            data=df,
            legend='full',
            alpha=0.7
        )
        # plt.title('t-SNE Visualization (Predicted Labels)')

        plt.tight_layout()
        plt.show()


class AutoEncoder(nn.Module):
    def __init__(self, num_layers, feat_dim, embed_dim, activation=None, noise_std=0.1):
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
        for i in range(num_layers):
            self.encoder.append(nn.Sequential(
                nn.Linear(tmp[i], self.embed_dim // (num_layers - i)), self._get_activation_fn(activation)
            ))
            tmp.append(self.embed_dim // (num_layers - i))
        for j in range(num_layers):
            self.decoder.append(nn.Sequential(
                nn.Linear(tmp[-(j + 1)], tmp[-(j + 2)]), self._get_activation_fn(activation)
            ))

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
            activation=None
    ):
        super(MuxGNNGraph, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.relations = relations
        self.num_relations = len(self.relations)
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.activation = activation
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [MuxGNNLayer(relations=self.relations, in_dim=self.embed_dim, out_dim=self.dim_a, dim_a=self.dim_a,
                         dropout=self.dropout, activation=self.activation)]
        )
        for _ in range(1, self.num_gnn_layers):
            self.layers.append(
                MuxGNNLayer(relations=self.relations, in_dim=self.dim_a, out_dim=self.dim_a, dim_a=self.dim_a,
                            dropout=self.dropout, activation=self.activation)
            )

        self.alpha = nn.Parameter(torch.ones(self.num_gnn_layers - 1))        # 残差连接中可学习的参数 alpha

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
        elif activation == 'softmax':
            # Softmax requires specifying the dimension to apply it to
            act_fn = nn.Softmax(dim=-1)
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, graph, feat):
        h = feat

        alpha = torch.sigmoid(self.alpha)  # 将 alpha 值约束在[0,1]之间
        for i, layer in enumerate(self.layers):
            if i == self.num_gnn_layers - 1:
                h_new, attention_weights = layer(graph, h, Is_attention=True,  Is_last=True)
            else:
                h_new = layer(graph, h, Is_attention=False)
            if i == 0:
                h = h_new
            else:
                h = alpha[i - 1] * h_new + (1 - alpha[i - 1]) * h

        return h, attention_weights


class MuxGNNLayer(nn.Module):
    def __init__(
            self,
            relations,
            in_dim,
            out_dim,
            dim_a,
            dropout=0.,
            activation=None,
            use_autoencoder=True,  # 使用自编码器的标志
    ):
        super(MuxGNNLayer, self).__init__()
        self.relations = relations
        self.num_relations = len(self.relations)
        self.use_autoencoder = use_autoencoder
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.act_str = activation

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(self.act_str)

        self.gnn = GraphConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                norm='both',
                weight=True,
                bias=True,
                activation=self.activation,
                allow_zero_in_degree=True
            )

        self.attention = SemanticAttentionBatched(self.num_relations, self.out_dim, self.dim_a, dropout=dropout)

        # self.norm = None
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True)
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
        elif activation == 'softmax':
            # Softmax requires specifying the dimension to apply it to
            act_fn = nn.Softmax(dim=-1)
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, graph, feat, Is_attention=False, Is_last=False):
        h = torch.zeros(self.num_relations, graph.num_nodes(), self.out_dim, device=graph.device)

        with graph.local_scope():
            for i, graph_layer in enumerate(self.relations):
                rel_graph = graph['node', graph_layer, 'node']
                h_out = self.gnn(rel_graph, feat).squeeze()
                h[i] = h_out

        if self.norm:
            h = self.norm(h)

        h = self.attention(graph, h, Is_attention=Is_attention)
        if Is_last:
            attention_weights = self.attention.get_top_k_nodes()
            return h, attention_weights

        return h


class BinaryClassifier(nn.Module):
    def __init__(self, embed_dim, top_k):
        super(BinaryClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k

        self.gate_nn = torch.nn.Linear(embed_dim, 1)
        self.gap = GlobalAttentionPooling(self.gate_nn)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
        self.relu_features = None
        self._register_hooks()

    def _register_hooks(self):

        def hook_function(module, input, output):
            self.relu_features = output.detach().cpu()

        relu_layer = self.classifier[1]
        relu_layer.register_forward_hook(hook_function)

    def forward(self,graph, x):
        graph_x, attention_weights = self.gap(graph, x, get_attention=True)
        top_k_weights, top_k_indices = torch.topk(attention_weights.squeeze(), self.top_k)
        return self.classifier(graph_x), top_k_weights, top_k_indices, self.relu_features


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
        self.attention_weights = None

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
    reconstruction_loss = F.mse_loss(decoded, original)
    total_loss = 1e-2 * beta * reconstruction_loss
    return total_loss
