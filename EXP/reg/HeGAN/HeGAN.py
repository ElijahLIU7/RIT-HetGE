import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import numpy as np
import logging
from dgl.dataloading import GraphDataLoader
from collections import OrderedDict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from protein_wang.pyHGT.utils import load_graphpred_dataset, load_graphpred_testDataset
from abc import ABCMeta
import random
from tqdm import tqdm


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
            out_dict[i] = h[pre:value.shape[0] + pre]
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


# 定义生成器
class HeGAN(BaseModel):
    r"""
    HeGAN was introduced in `Adversarial Learning on Heterogeneous Information Networks <https://dl.acm.org/doi/10.1145/3292500.3330970>`_

    It included a **Discriminator** and a **Generator**. For more details please read docs of both.

    Parameters
    ----------
    emb_size: int
        embedding size
    hg: dgl.heteroGraph
        hetorogeneous graph
    """

    def __init__(self, emb_size):
        super().__init__()

        self.generator = Generator(emb_size)
        self.discriminator = Discriminator(emb_size)

    def forward(self, hg):
        """
                Receives a heterogeneous graph `hg`, initializes node and relation embeddings if necessary,
                and performs the adversarial process to update the embeddings.
                """
        self.generator.init_embeddings(hg)
        self.discriminator.init_embeddings(hg)

        # Generate fake samples and discriminate
        # Here you would proceed with the HeGAN adversarial training steps using the dynamically initialized embeddings.

        # The generator and discriminator produce node embeddings that can be aggregated for graph-level classification.
        node_emb = self.generator.get_graph_embedding(hg)  # Aggregate node embeddings for graph-level representation
        return node_emb  # This can be used as the input to a downstream classifier for graph classification.

    def extra_loss(self):
        pass


class Generator(nn.Module):
    r"""
     A Discriminator :math:`D` eveluates the connectivity between the pair of nodes :math:`u` and :math:`v` w.r.t. a relation :math:`r`. It is formulated as follow:

    .. math::
        D(\mathbf{e}_v|\mathbf{u},\mathbf{r};\mathbf{\theta}^D) = \frac{1}{1+\exp(-\mathbf{e}_u^{D^T}) \mathbf{M}_r^D \mathbf{e}_v}

    where :math:`e_v \in \mathbb{R}^{d\times 1}` is the input embeddings of the sample :math:`v`,
    :math:`e_u^D \in \mathbb{R}^{d \times 1}` is the learnable embedding of node :math:`u`,
    :math:`M_r^D \in \mathbb{R}^{d \times d}` is a learnable relation matrix for relation :math:`r`.

    There are also a two-layer MLP integrated into the generator for enhancing the expression of the fake samples:

    .. math::
        G(\mathbf{u}, \mathbf{r}; \mathbf{\theta}^G) = f(\mathbf{W_2}f(\mathbf{W}_1 \mathbf{e} + \mathbf{b}_1) + \mathbf{b}_2)

    where :math:`e` is drawn from Gaussian distribution. :math:`\{W_i, b_i}` denote the weight matrix and bias vector for :math:`i`-th layer.

    The discriminator Loss is :

    .. math::
        L_G = \mathbb{E}_{\langle u,v\rangle \sim P_G, e'_v \sim G(u,r;\theta^G)} = -\log -D(e'_v|u,r)) +\lambda^G || \theta^G ||_2^2

    where :math:`\theta^G` denote all the learnable parameters in Generator.

    Parameters
    -----------
    emb_size: int
        embeddings size.
    hg: dgl.heteroGraph
        heterogenous graph.

    """

    def __init__(self, emb_size):
        super().__init__()
        self.n_relation = 6
        self.node_emb_dim = emb_size
        self.etypes = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
        self.nodes_embedding = nn.ParameterDict()
        # for nodes_type, nodes_emb in zip(hg.ntypes, hg.ndata['emb']):
        #     self.nodes_embedding[nodes_type] = nn.Parameter(nodes_emb, requires_grad=True)

        self.relation_matrix = nn.ParameterDict()
        for et in self.etypes:
            rm = torch.empty(self.node_emb_dim, self.node_emb_dim)
            rm = nn.init.xavier_normal_(rm)
            self.relation_matrix[et] = nn.Parameter(rm, requires_grad=True)

        self.fc = nn.Sequential(
            OrderedDict([
                ("w_1", nn.Linear(in_features=self.node_emb_dim, out_features=self.node_emb_dim, bias=True)),
                ("a_1", nn.LeakyReLU()),
                ("w_2", nn.Linear(in_features=self.node_emb_dim, out_features=self.node_emb_dim)),
                ("a_2", nn.LeakyReLU())
            ])
        )

    def init_embeddings(self, hg):
        # Initialize nodes_embedding and relation_matrix based on the input graph structure
        self.nodes_embedding = nn.ParameterDict({
            nt: nn.Parameter(torch.randn(hg.num_nodes(nt), self.node_emb_dim), requires_grad=True)
            for nt in hg.ntypes
        }).to(hg.device)

        self.relation_matrix = nn.ParameterDict({
            et: nn.Parameter(torch.empty(self.node_emb_dim, self.node_emb_dim).normal_(), requires_grad=True)
            for et in hg.etypes
        }).to(hg.device)

    def get_graph_embedding(self, hg):
        # Aggregate node embeddings to produce a graph-level embedding for classification
        node_embeddings = [self.nodes_embedding[nt] for nt in hg.ntypes]
        graph_embedding = torch.cat(node_embeddings).mean(dim=0)
        return graph_embedding

    def forward(self, gen_hg, dis_node_emb, dis_relation_matrix, noise_emb):
        r"""
        Parameters
        -----------
        gen_hg: dgl.heterograph
            sampled graph for generator.
        dis_node_emb: dict[str: Tensor]
            discriminator node embedding.
        dis_relation_matrix: dict[str: Tensor]
            discriminator relation embedding.
        noise_emb: dict[str: Tensor]
            noise embedding.
        """
        score_list = []
        with gen_hg.local_scope():
            self.assign_node_data(gen_hg, dis_node_emb)
            self.assign_edge_data(gen_hg, dis_relation_matrix)
            self.generate_neighbor_emb(gen_hg, noise_emb)
            for et in gen_hg.canonical_etypes:
                gen_hg.apply_edges(lambda edges: {'s': edges.src['dh'].unsqueeze(1).matmul(edges.data['de']).squeeze(1)},
                                   etype=et)
                gen_hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.data['g'])}, etype=et)

                score = torch.sum(gen_hg.edata['score'].pop(et), dim=1)
                score_list.append(score)

        return torch.cat(score_list)

    def generate_neighbor_emb(self, hg, noise_emb):
        for et in hg.canonical_etypes:
            hg.apply_edges(lambda edges: {'g': edges.src['h'].unsqueeze(1).matmul(edges.data['e']).squeeze(1)},
                           etype=et)
            hg.apply_edges(lambda edges: {'g': edges.data['g'] + noise_emb[et]}, etype=et)
            hg.apply_edges(lambda edges: {'g': self.fc(edges.data['g'])}, etype=et)

        return {et: hg.edata['g'][et] for et in hg.canonical_etypes}

    def assign_edge_data(self, hg, dis_relation_matrix=None):
        for et in hg.canonical_etypes:
            n = hg.num_edges(et)
            e = self.relation_matrix[et[1]]
            # 如果只有一种边类型，直接赋值张量
            if len(hg.etypes) == 1:
                hg.edata['e'] = e.expand(n, -1, -1)
            else:
                hg.edata['e'] = {et: e.expand(n, -1, -1)}
            if dis_relation_matrix:
                if len(hg.etypes) == 1:
                    de = dis_relation_matrix[et[1]]
                    hg.edata['de'] = de.expand(n, -1, -1)
                else:
                    de = dis_relation_matrix[et[1]]
                    hg.edata['de'] = {et: de.expand(n, -1, -1)}

    def assign_node_data(self, hg, dis_node_emb=None):
        for nt in hg.ntypes:
            # num_nodes = hg.num_nodes(nt)
            # if nt not in self.nodes_embedding or self.nodes_embedding[nt].size(0) != num_nodes:
            #     self.nodes_embedding[nt] = nn.Parameter(torch.randn(num_nodes, self.node_emb_dim),
            #                                             requires_grad=True).to(hg.device)
            hg.nodes[nt].data['h'] = self.nodes_embedding[nt]
        # if dis_node_emb:
        #     hg.ndata['dh'] = dis_node_emb
        if dis_node_emb:
            # 确认是否有多个节点类型
            if len(hg.ntypes) == 1:
                hg.ndata['dh'] = dis_node_emb[nt]  # 直接传入张量
            else:
                hg.ndata['dh'] = dis_node_emb


class Discriminator(nn.Module):
    r"""
    A generator :math:`G` samples fake node embeddings from a continuous distribution. The distribution is Gaussian distribution:

    .. math::
        \mathcal{N}(\mathbf{e}_u^{G^T} \mathbf{M}_r^G, \mathbf{\sigma}^2 \mathbf{I})

    where :math:`e_u^G \in \mathbb{R}^{d \times 1}` and :math:`M_r^G \in \mathbb{R}^{d \times d}` denote the node embedding of :math:`u \in \mathcal{V}` and the relation matrix of :math:`r \in \mathcal{R}` for the generator.

    There are also a two-layer MLP integrated into the generator for enhancing the expression of the fake samples:

    .. math::
        G(\mathbf{u}, \mathbf{r}; \mathbf{\theta}^G) = f(\mathbf{W_2}f(\mathbf{W}_1 \mathbf{e} + \mathbf{b}_1) + \mathbf{b}_2)

    where :math:`e` is drawn from Gaussian distribution. :math:`\{W_i, b_i}` denote the weight matrix and bias vector for :math:`i`-th layer.

    The discriminator Loss is:

    .. math::
        L_1^D = \mathbb{E}_{\langle u,v,r\rangle \sim P_G} = -\log D(e_v^u|u,r))

        L_2^D = \mathbb{E}_{\langle u,v\rangle \sim P_G, r' \sim P_{R'}} = -\log (1-D(e_v^u|u,r')))

        L_3^D = \mathbb{E}_{\langle u,v\rangle \sim P_G, e'_v \sim G(u,r;\theta^G)} = -\log (1-D(e_v'|u,r)))

        L_G = L_1^D + L_2^D + L_2^D + \lambda^D || \theta^D ||_2^2

    where :math:`\theta^D` denote all the learnable parameters in Discriminator.

    Parameters
    -----------
    emb_size: int
        embeddings size.
    hg: dgl.heteroGraph
        heterogenous graph.

    """
    def __init__(self, emb_size):
        super().__init__()
        self.n_relation = 6
        self.node_emb_dim = emb_size
        self.etypes = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
        self.nodes_embedding = nn.ParameterDict()

        self.relation_matrix = nn.ParameterDict()
        for et in self.etypes:
            rm = torch.empty(self.node_emb_dim, self.node_emb_dim)
            rm = nn.init.xavier_normal_(rm)
            self.relation_matrix[et] = nn.Parameter(rm, requires_grad=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.node_emb_dim, self.node_emb_dim // 2),
            nn.ReLU(),
            nn.Linear(self.node_emb_dim // 2, 1)
        )

    def init_embeddings(self, hg):
        # Similar to Generator, initialize node and relation embeddings dynamically
        self.nodes_embedding = nn.ParameterDict({
            nt: nn.Parameter(torch.randn(hg.num_nodes(nt), self.node_emb_dim), requires_grad=True)
            for nt in hg.ntypes
        }).to(hg.device)

        self.relation_matrix = nn.ParameterDict({
            et: nn.Parameter(torch.empty(self.node_emb_dim, self.node_emb_dim).normal_(), requires_grad=True)
            for et in hg.etypes
        }).to(hg.device)

    def forward(self, g, pos_hg, neg_hg1, neg_hg2, generate_neighbor_emb):
        r"""
        Parameters
        ----------
        pos_hg:
            sampled postive graph.
        neg_hg1:
            sampled negative graph with wrong relation.
        neg_hg2:
            sampled negative graph wtih wrong node.
        generate_neighbor_emb:
            generator node embeddings.
        """
        self.assign_node_data(pos_hg)
        self.assign_node_data(neg_hg1)
        self.assign_node_data(neg_hg2, generate_neighbor_emb)
        self.assign_edge_data(pos_hg)
        self.assign_edge_data(neg_hg1)
        self.assign_edge_data(neg_hg2)

        # with pos_hg.local_scope():
        #     pos_g = dgl.mean_nodes(pos_hg, 'h')  # 也可以使用 sum 或 max 来进行全局池化
        #     # 整图池化
        #     graph_embd_pos_g = self.mlp(pos_g)
        #
        # with neg_hg1.local_scope():
        #     neg_g1 = dgl.mean_nodes(neg_hg1, 'h')  # 也可以使用 sum 或 max 来进行全局池化
        #     # 整图池化
        #     graph_embd_neg_g1 = self.mlp(neg_g1)
        #
        # with neg_hg2.local_scope():
        #     neg_g2 = dgl.mean_nodes(neg_hg2, 'h')  # 也可以使用 sum 或 max 来进行全局池化
        #     # 整图池化
        #     graph_embd_neg_g2 = self.mlp(neg_g2)
        with g.local_scope():
            g.ndata['h'] = self.nodes_embedding['node']
            hg = dgl.mean_nodes(g, 'h')
        graph_embd = self.mlp(hg)

        pos_score = self.score_pred(pos_hg)
        neg_score1 = self.score_pred(neg_hg1)
        neg_score2 = self.score_pred(neg_hg2)

        # embed = self.nodes_embedding['node']
        # neg_g1 = dgl.mean_nodes(embed, 'node')  # 也可以使用 sum 或 max 来进行全局池化
        # # 整图池化
        # graph_embd = self.mlp(neg_g1)

        return pos_score, neg_score1, neg_score2, graph_embd      # graph_embd_pos_g, graph_embd_neg_g1, graph_embd_neg_g2

    def get_parameters(self):
        r"""
        return discriminator node embeddings and relation embeddings.
        """
        return {k: self.nodes_embedding[k] for k in self.nodes_embedding.keys()}, \
               {k: self.relation_matrix[k] for k in self.relation_matrix.keys()}

    def score_pred(self, hg):
        r"""
        predict the discriminator score for sampled heterogeneous graph.
        """
        score_list = []
        # with hg.local_scope():
        for et in hg.canonical_etypes:
            hg.apply_edges(lambda edges: {
                's': edges.src['h'].unsqueeze(1).matmul(edges.data['e']).reshape(hg.num_edges(et), 572)},
                           etype=et)
            if len(hg.edata['f']) == 0:
                hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.dst['h'])}, etype=et)
            else:
                hg.apply_edges(lambda edges: {'score': edges.data['s'].multiply(edges.data['f'])}, etype=et)
            score = torch.sum(hg.edata['score'].pop(et), dim=1)
            score_list.append(score)
        return torch.cat(score_list)

    def assign_edge_data(self, hg):
        d = {}
        for et in hg.canonical_etypes:
            e = self.relation_matrix[et[1]]
            n = hg.num_edges(et)
            d[et] = e.expand(n, -1, -1)
        hg.edata['e'] = d

    def assign_node_data(self, hg, generate_neighbor_emb=None):
        for nt in hg.ntypes:
            hg.nodes[nt].data['h'] = self.nodes_embedding[nt]
        if generate_neighbor_emb:
            hg.edata['f'] = generate_neighbor_emb


# 训练 HeGAN 模型
def train_hegan_model(train_dataset, valid_dataset, embed_size, epochs=50, lr=0.001, batch_size=8, fold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化 HeGAN 模型，使用 train_dataset 中的异构图来初始化
    hegan = HeGAN(emb_size=embed_size).to(device)
    generator = hegan.generator.to(device)
    discriminator = hegan.discriminator.to(device)

    # 优化器
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # 损失函数
    criterion = nn.HuberLoss(reduction='mean', delta=1.0)

    # 数据加载器
    train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    train_dis_loss = 0
    best_model_path = f'best_model_fold_{fold}.pt' if fold is not None else 'best_att_hgcn_model.pt'

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        accumulation_steps = 10
        disc_optimizer.zero_grad()
        gen_optimizer.zero_grad()
        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for i, batch in enumerate(epoch_iter):
            # discriminator step
            batch_graph, labels, _ = batch
            labels = labels.to(device)
            _ = hegan(batch_graph.to(device))
            sampler = GraphSampler(batch_graph, 2)
            pos_hg, pos_hg1, pos_hg2 = sampler.sample_graph_for_dis()
            pos_hg, pos_hg1, pos_hg2 = pos_hg.to(device), pos_hg1.to(device), pos_hg2.to(device)
            noise_emb = {
                et: torch.tensor(
                    np.random.normal(0.0, 1, (pos_hg2.num_edges(et), embed_size)).astype(
                        'float32')).to(device)
                for et in pos_hg2.canonical_etypes
            }

            generator.assign_node_data(pos_hg2, None)
            generator.assign_edge_data(pos_hg2, None)
            generate_neighbor_emb = generator.generate_neighbor_emb(pos_hg2, noise_emb)
            pos_score, neg_score1, neg_score2, graph_embd = discriminator(batch_graph.to(device), pos_hg, pos_hg1, pos_hg2, generate_neighbor_emb)

            # batch_graph_d = batch_graph.to(device)
            # with batch_graph_d.local_scope():
            #     batch_graph_d.ndata['h'] = discriminator.nodes_embedding['node']
            #     hg = dgl.mean_nodes(batch_graph_d, 'h')
            #     graph_embd = discriminator.mlp(hg)

            loss = criterion(graph_embd.squeeze(), labels)

            pos_loss = -torch.mean(F.logsigmoid(pos_score))
            neg_loss1 = -torch.mean(F.logsigmoid(-neg_score1 + 1e-5))
            neg_loss2 = -torch.mean(F.logsigmoid(-neg_score2 + 1e-5))
            dis_loss = pos_loss + neg_loss1 + neg_loss2 + loss
            train_dis_loss += dis_loss

            dis_loss.backward()
            if (i+1) % accumulation_steps == 0:
                disc_optimizer.zero_grad()
                disc_optimizer.step()

            # generator step
            dis_node_emb, dis_relation_matrix = discriminator.get_parameters()
            sampler = GraphSampler(batch_graph, 2)
            gen_hg = sampler.sample_graph_for_gen()
            noise_emb = {
                et: torch.tensor(
                    np.random.normal(0.0, 1, (gen_hg.num_edges(et), embed_size)).astype(
                        'float32')).to(device)
                for et in gen_hg.canonical_etypes
            }
            gen_hg = gen_hg.to(device)
            score = generator(gen_hg, dis_node_emb, dis_relation_matrix, noise_emb)
            gen_loss = -torch.mean(F.logsigmoid(score)) * (1 - labels) + \
                       -torch.mean(F.logsigmoid(1 - score + 1e-5)) * labels
            gen_loss = gen_loss.sum()

            gen_loss.backward()
            if (i + 1) % accumulation_steps == 0:
                gen_optimizer.zero_grad()
                gen_optimizer.step()

            if i % 5 == 0:
                torch.cuda.empty_cache()  # 清理显存中的缓存

        dis_loss, gen_loss = dis_loss.item(), gen_loss.item()

        hegan.eval()
        generator.eval()
        discriminator.eval()
        valid_dis_loss = 0.0
        valid_gen_loss = 0.0
        num_batches = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in valid_loader:
                batch_graph, labels, _ = batch
                labels = labels.to(device)
                _ = hegan(batch_graph.to(device))
                sampler = GraphSampler(batch_graph, 10)
                pos_hg, pos_hg1, pos_hg2 = sampler.sample_graph_for_dis()
                pos_hg, pos_hg1, pos_hg2 = pos_hg.to(device), pos_hg1.to(device), pos_hg2.to(device)
                noise_emb = {
                    et: torch.tensor(
                        np.random.normal(0.0, 1, (pos_hg2.num_edges(et), embed_size)).astype(
                            'float32')).to(device)
                    for et in pos_hg2.canonical_etypes
                }

                generator.assign_node_data(pos_hg2, None)
                generator.assign_edge_data(pos_hg2, None)
                generate_neighbor_emb = generator.generate_neighbor_emb(pos_hg2, noise_emb)
                pos_score, neg_score1, neg_score2 = discriminator(pos_hg, pos_hg1, pos_hg2, generate_neighbor_emb)

                batch_graph_d = batch_graph.to(device)
                with batch_graph_d.local_scope():
                    batch_graph_d.ndata['h'] = discriminator.nodes_embedding['node']
                    hg = dgl.mean_nodes(batch_graph_d, 'h')
                    graph_embd = discriminator.mlp(hg)

                loss = criterion(graph_embd.squeeze(), labels)
                all_preds.extend(graph_embd.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pos_loss = -torch.mean(F.logsigmoid(pos_score))
                neg_loss1 = -torch.mean(F.logsigmoid(-neg_score1 + 1e-5))
                neg_loss2 = -torch.mean(F.logsigmoid(-neg_score2 + 1e-5))
                dis_loss = pos_loss + neg_loss1 + neg_loss2 + loss
                valid_dis_loss += dis_loss.item()

                # 生成器损失
                dis_node_emb, dis_relation_matrix = hegan.discriminator.get_parameters()
                gen_hg = sampler.sample_graph_for_gen()
                noise_emb = {
                    et: torch.tensor(
                        np.random.normal(0.0, 1, (gen_hg.num_edges(et), embed_size)).astype(
                            'float32')).to(device)
                    for et in gen_hg.canonical_etypes
                }
                gen_hg = gen_hg.to(device)
                score = hegan.generator(gen_hg, dis_node_emb, dis_relation_matrix, noise_emb)
                gen_loss = -torch.mean(F.logsigmoid(score)) * (1 - labels) + \
                           -torch.mean(F.logsigmoid(1 - score + 1e-5)) * labels
                valid_gen_loss += gen_loss.item()

                num_batches += 1
        # 计算平均损失
        avg_dis_loss = valid_dis_loss / num_batches
        avg_gen_loss = valid_gen_loss / num_batches

        print(epoch)
        print("discriminator:\n\tloss:{:.4f}".format(avg_dis_loss))
        print("generator:\n\tloss:{:.4f}".format(avg_gen_loss))

        # 计算验证集上的损失和指标
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        val_loss = np.mean((all_labels - all_preds) ** 2)
        mse = mean_squared_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        pcc = np.corrcoef(all_labels, all_preds)[0, 1]

        # 保存验证集上表现最好的模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict()
            }, best_model_path)

    print("Training completed.")
    return generator, best_loss, mse, r2, mae, pcc, best_model_path


class GraphSampler:
    r"""
    First load graph data to self.hg_dict, then interate.
    """

    def __init__(self, hg, k):
        self.k = k
        self.ets = hg.canonical_etypes
        self.nt_et = {}
        for et in hg.canonical_etypes:
            if et[0] not in self.nt_et:
                self.nt_et[et[0]] = [et]
            else:
                self.nt_et[et[0]].append(et)

        self.hg_dict = {key: {} for key in hg.ntypes}
        for nt in hg.ntypes:
            for nid in range(hg.num_nodes(nt)):
                if nid not in self.hg_dict[nt]:
                    self.hg_dict[nt][nid] = {}
                for et in self.nt_et[nt]:
                    self.hg_dict[nt][nid][et] = hg.successors(nid, et)

    def sample_graph_for_dis(self):
        r"""
        sample three graphs from original graph.

        Note
        ------------
        pos_hg:
            Sampled graph from true graph distribution, that is from the original graph with real node and real relation.
        neg_hg1:
            Sampled graph with true nodes pair but wrong realtion.
        neg_hg2:
            Sampled graph with true scr nodes and realtion but wrong node embedding.
            Embedding are generated by Generator, so we can use `pos_hg` as adjacency matrix.
        """
        pos_dict = {}
        neg_dict1 = {}

        for nt in self.hg_dict.keys():
            for src in self.hg_dict[nt].keys():
                for i in range(self.k):
                    et = random.choice(self.nt_et[nt])
                    if len(self.hg_dict[nt][src][et]) == 0:
                        continue
                    dst = random.choice(self.hg_dict[nt][src][et])
                    if et not in pos_dict:
                        pos_dict[et] = ([src], [dst])
                    else:
                        pos_dict[et][0].append(src)
                        pos_dict[et][1].append(dst)

                    wrong_et = random.choice(self.ets)
                    while wrong_et == et:
                        wrong_et = random.choice(self.ets)
                    wrong_et = (et[0], wrong_et[1], et[2])

                    if wrong_et not in neg_dict1:
                        neg_dict1[wrong_et] = ([src], [dst])
                    else:
                        neg_dict1[wrong_et][0].append(src)
                        neg_dict1[wrong_et][1].append(dst)

        pos_hg = dgl.heterograph(pos_dict, {nt: len(self.hg_dict[nt].keys()) for nt in self.hg_dict.keys()})
        neg_hg1 = dgl.heterograph(neg_dict1, {nt: len(self.hg_dict[nt].keys()) for nt in self.hg_dict.keys()})
        neg_hg2 = dgl.heterograph(pos_dict, {nt: len(self.hg_dict[nt].keys()) for nt in self.hg_dict.keys()})

        return pos_hg, neg_hg1, neg_hg2

    def sample_graph_for_gen(self):
        d = {}
        for nt in self.hg_dict.keys():
            for src in self.hg_dict[nt].keys():
                for i in range(self.k):
                    et = self.nt_et[nt][random.randint(0, len(self.nt_et[nt]) - 1)]
                    if len(self.hg_dict[nt][src][et]) == 0:
                        continue
                    dst = self.hg_dict[nt][src][et][random.randint(0, len(self.hg_dict[nt][src][et]) - 1)]
                    if et not in d:
                        d[et] = ([src], [dst])
                    else:
                        d[et][0].append(src)
                        d[et][1].append(dst)

        return dgl.heterograph(d, {nt: len(self.hg_dict[nt].keys()) for nt in self.hg_dict.keys()})


def log_metrics(epoch, total_loss, mse, r2, mae, pcc):
    """记录每个epoch的结果"""
    logging.info(f"Epoch {epoch}: Loss = {total_loss:.4f}, MSE = {mse:.4f}, R2 = {r2:.4f}, "
                 f"MAE = {mae:.4f}, PCC = {pcc:.4f}")


def log_metrics(epoch, total_loss, auc, f1, precision, recall, accuracy):
    """记录每个epoch的结果"""
    logging.info(f"Epoch {epoch}: Loss = {total_loss:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, "
                 f"Precision = {precision:.4f}, Recall = {recall:.4f}, Accuracy = {accuracy:.4f}")


def cross_validation(dataset_path, CV_FOLDS=10, epochs=50, lr=0.1, batch_size=8, emb_feats=512, bidirected=False):
    all_accuracies = []
    feat_dim = None
    best_model_path = None
    node_dict = {'node': 0}
    edge_dict = {'HBOND': 0, 'IONIC': 1, 'PICATION': 2, 'PIPISTACK': 3, 'SSBOND': 4, 'VDW': 5}

    for cv_select in range(CV_FOLDS):
        print(f"Running fold {cv_select + 1}/{CV_FOLDS}...")

        # 加载数据集，根据当前折数选择训练集和验证集
        train_dataset, valid_dataset, feat_dim, relations = load_graphpred_dataset(
            dataset_path, CV_FOLDS=CV_FOLDS, cv_select=cv_select, bidirected=bidirected
        )

        # 训练模型并记录验证集的准确率
        _, best_loss, mse, r2, mae, pcc, model_path = train_hegan_model(train_dataset, valid_dataset,
                                                                        embed_size=feat_dim,
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
emb_feats = 512
CV_FOLDS = 10
best_model_path, _ = cross_validation(dataset_path, CV_FOLDS=CV_FOLDS, epochs=50, lr=0.0001, batch_size=2,
                                      emb_feats=emb_feats, bidirected=False)

# 存储所有折的指标
all_mse = []
all_r2 = []
all_mae = []
all_pcc = []
