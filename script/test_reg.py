import os
import argparse
import logging
import re
import torch
import optuna
import csv
import joblib
import dgl
import pandas as pd

from dgl.data import DGLDataset
from HGRIFN.HG_RIFN_reg import GraphRegressor
from HGRIFN.utils import load_dataset, load_testDataset


def load_graphpred_dataset(dataset):
    data_path = f'D:/program/GitHub/protein_wang/data/output_Fold_regression'
    graph_path = f'{data_path}/graphs_with_labels_train_fold0_stand_last.bin'
    graphs, graph_attr = dgl.load_graphs(graph_path)

    relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
    feat_dim = graphs[0].ndata['emb'].size(-1)

    test_graphs = []
    test_labels = []
    test_name_protein = []

    for g, cv, lbl, nap in zip(graphs, graph_attr['cv_folds'], graph_attr['labels'], graph_attr['name_protein']):
        # 解码字符串
        name_protein = "".join(map(chr, nap.numpy())).strip()
        if name_protein[-1] == '\x00':
            name_protein = name_protein[:-4]
        test_graphs.append(g)
        test_labels.append(lbl)
        test_name_protein.append(name_protein)

    test_dataset = Graphdataset(dataset, test_graphs, torch.FloatTensor(test_labels), test_name_protein)

    return test_dataset, feat_dim, relations


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


def objective(trial):
    """
    使用optuna优化模型参数
    """
    # 定于超参数搜索空间
    print('Version: HG-RIFN_regression_test')
    dataset = trial.suggest_categorical('dataset', ['train_autoencoder'])
    model_name = trial.suggest_categorical('model_name', [f'graphs_with_regression_{dataset}_skip_batch'])
    data_dir = trial.suggest_categorical('data_dir', ['D:/program/GitHub/protein_wang/data/output_test'])
    gnn = trial.suggest_categorical('gnn', ['gin'])  # 'gin', 'gcn', 'gat', 'hgt'
    num_gnn_layer = 2        # 3
    num_coders_layers = 2
    pos_class_weight = 1.0  # trial.suggest_float('pos_class_weight', 0.5, 2.0)
    # 先定义一个较大的范围以便 trial 进行选择
    embed_dim = 1024
    dim_a = 27        # 19
    dropout = 0.27759357302825227
    activation = trial.suggest_categorical('activation', ['elu'])
    batch_size = 1
    epochs = 600
    lr = 0.00035860335262601496 # 5.109265806384588e-05
    weight_decay = 3.5860335262601496e-06
    accum_steps = 1  # 在权重更新之前要进行的梯度累积步骤数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # out_model_dir = "E:/dataset/结果/protein_wang/交叉验证最优模型保存/0.85"
    # out_model_dir = "D:/program/GitHub/protein_wang/data/output_Fold_regression/graphs_with_regression_train_skip_LR8.825812779106674e-05"       # 7.091007092290207e-05"
    # out_model_dir = f'D:/program/GitHub/protein_wang/data/output_Fold_regression/graphs_with_regression_train_skip_batch_LR{lr}'
    # out_model_dir = f'E:/dataset/结果/protein_wang/交叉验证最优模型保存/0.85'
    out_model_dir = f'{args.input}/{model_name}_LR{lr}'
    os.makedirs(out_model_dir, exist_ok=True)

    # 输出所有参数定义值
    print("Parameters:")
    print("     gnn:", gnn)
    print("     num_gnn_layer:", num_gnn_layer)
    print("     pos_class_weight:", pos_class_weight)
    print("     embed_dim:", embed_dim)
    print("     dim_a:", dim_a)
    print("     dropout:", dropout)
    print("     activation:", activation)
    print("     batch_size:", batch_size)
    print("     epochs:", epochs)
    print("     lr:", lr)
    print("     weight_decay:", weight_decay)
    print("     accum_steps:", accum_steps)

    log_path = f'protein_wang/logs/{dataset}/{model_name}'
    log_fname = f'{log_path}/test_log_{gnn}_{dataset}_LastAttention.out'
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=log_fname,
        filemode='a'
    )
    logging.info(f'Starting trial with parameters: {trial.params}')

    (test_dataset,
     feat_dim,
     relations) = load_graphpred_dataset(dataset)

    model = GraphRegressor(
        gnn_type=gnn,
        num_gnn_layers=num_gnn_layer,
        num_coder_layers=num_coders_layers,
        relations=relations,
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        dim_a=dim_a,
        dropout=dropout,
        activation=activation
    )
    t_loss, t_r2, t_mae, t_pcc, name_protein, (t_top_k_nodes, t_top_k_relations, top_k_attention_weights) = model.eval_model(
        test_dataset, batch_size=batch_size, num_workers=args.num_workers, device=device, Is_test=True, model_load=out_model_dir, Is_Best_test=False
    )

    Tm = pd.read_csv(f'F:/dataset/protein/test2_dataset.csv')
    Tm['Protein_ID'] = Tm['Protein_ID'].apply(lambda x: re.split('_|-', x)[0])
    Tm_dict = pd.Series(Tm.Tm.values, index=Tm.Protein_ID).to_dict()
    # 定义氨基酸三字母简写和一字母简写的对应字典
    amino_acid_map = {
        'A': 'ALA',
        'R': 'ARG',
        'N': 'ASN',
        'D': 'ASP',
        'Q': 'GLN',
        'E': 'GLU',
        'G': 'GLY',
        'H': 'HIS',
        'L': 'LEU',
        'M': 'MET',
        'F': 'PHE',
        'P': 'PRO',
        'S': 'SER',
        'T': 'THR',
        'W': 'TRP',
        'Y': 'TYR',
        'V': 'VAL',
        'K': 'LYS',
        'I': 'ILE',
        'C': 'CYS'
    }

    # 将结果和超参数写入CSV文件
    with open(log_fname, 'a') as f:
        f.write(
            '\n'.join(
                ('-' * 25,
                 f'Test Loss: {t_loss}',
                 f'Test R2: {t_r2}',
                 f'Test MAE: {t_mae}',
                 f'Test PCC: {t_pcc}',
                 '-' * 25 + '\n')
            )
        )

    # 测试集K-top
    output_dir = f'F:/dataset/结果/protein_wang/regression/Test_Graph_top_k'
    os.makedirs(output_dir, exist_ok=True)
    for name, nodes, relations, weights in zip(name_protein, t_top_k_nodes, t_top_k_relations, top_k_attention_weights):
        if name[-1] == '\x00':
            name = name[:-4]
        # Read the uploaded FASTA file and calculate the length of the second line
        fasta_file_path = f'F:/dataset/protein/FASTA/test2_dataset_fasta/{name}.fasta'
        # Read the file and extract the second line
        with open(fasta_file_path, 'r') as file:
            lines = file.readlines()
            sequence = lines[1].strip()
        output_csv = os.path.join(output_dir, f'graph_{name}_nodes_relations_Tm-{Tm_dict[name]}.csv')
        with open(output_csv, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                ['Node', 'Residue', 'VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND',
                 'PICATION', 'weight'])  # Assuming 6 relations per node
            for node, relation, weight in zip(nodes, relations, weights):
                residue = amino_acid_map[sequence[node]]
                csvwriter.writerow([node + 1, residue] + relation + [weight])

    return t_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='graphs_with_labels',
                        help='Name to save the trained model.')
    parser.add_argument('--input', type=str, default='D:/program/GitHub/protein_wang/data/output_Fold_regression',
                        # output_Fold_balance_withLowTm',
                        help='The address of preprocessed graph.')
    parser.add_argument('--dataset', type=str, default='/real')
    parser.add_argument('--model_dir', type=str, default='./model_save',
                        help='The address for storing the models and optimization results.')
    parser.add_argument('--gnn', type=str, default='hgt',
                        help='GNN layer to use with muxGNN. "gcn", "gat", "hgt", or "gin". Default is "gin".')
    parser.add_argument('--num_gnn_layer', type=int, default=1,
                        help='Number of GNN layers in the embedding module.')
    parser.add_argument('--pos_class_weight', type=float, default=1.,
                        help='Additional weight to apply to loss contribution of positive class.')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Size of output embedding dimension.')
    parser.add_argument('--dim_a', type=int, default=16,
                        help='Dimension of attention.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate during training.')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function. Options are "relu", "elu", or "gelu".')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size during training and inference.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Maximum limit on training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='L2 regularization penalty.')
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Number of gradient accumulation steps to take before weight update.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes.')
    parser.add_argument('--bidirected', action='store_true', default=False,
                        help='Use a bidirectional version of the input graphs.')
    parser.add_argument('--n_trials', type=int, default=1,
                        help='Number of trial runs.')

    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    joblib.dump(study, 'study.pkl')

    print(f'Best trial: {study.best_trial.value}')
    print('Best hyperparameters: ', study.best_trial.params)
