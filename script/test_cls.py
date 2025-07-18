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
from HGRIFN.HG_RIFN_cls import HGraphClassifier


def load_graphpred_dataset(data):
    data_path = data
    graph_path = f'{data_path}/graphs_test_fold0.bin'
    graphs, graph_attr = dgl.load_graphs(graph_path)

    relations = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']
    feat_dim = graphs[0].ndata['emb'].size(-1)

    test_graphs = []
    test_labels = []
    test_name_protein = []

    graph_attr['labels'] = (graph_attr['labels'] >= 60).int()

    for g, cv, lbl, nap in zip(graphs, graph_attr['cv_folds'], graph_attr['labels'], graph_attr['name_protein']):
        name_protein = "".join(map(chr, nap.numpy())).strip()
        if name_protein[-1] == '\x00':
            name_protein = name_protein[:-4]
        test_graphs.append(g)
        test_labels.append(lbl)
        test_name_protein.append(name_protein)

    test_dataset = Graphdataset(data, test_graphs, torch.FloatTensor(test_labels), test_name_protein)

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
    Optimize the model parameters using optuna
    """
    print('Version: HG-RIFN_classification_test')
    data = args.data
    model_name = f'classification_{data}_test'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = f'{args.results}/Best_{model_name}_result'

    log_path = f'./logs/{model_name}'
    log_fname = f'{log_path}/test_log.out'
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
     relations) = load_graphpred_dataset(data)

    model = HGraphClassifier()
    t_acc, t_p, t_r, t_f1, t_auc, name_protein, (t_top_k_nodes, t_top_k_relations,
                                                 top_k_attention_weights) = model.eval_model(
        test_dataset, batch_size=1, num_workers=args.num_workers, device=device, Is_test=True,
        model_load=model_dir, Is_Best_test=False
    )

    Tm = pd.read_csv(f'{args.input}/test_dataset.csv')
    Tm['Protein_ID'] = Tm['Protein_ID'].apply(lambda x: re.split('_|-', x)[0])
    Tm_dict = pd.Series(Tm.Tm.values, index=Tm.Protein_ID).to_dict()
    # A dictionary defining the three-letter and one-letter abbreviations of amino acids.
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

    # Write the results and hyperparameters to the CSV file
    with open(log_fname, 'a') as f:
        f.write(
            '\n'.join(
                ('-' * 25,
                 f'Test Accuracies: {t_acc}',
                 f'Test Precisions: {t_p}',
                 f'Test Recalls: {t_r}',
                 f'Test F1s: {t_f1}',
                 f'Test AUCs:{t_auc}',
                 '-' * 25 + '\n')
            )
        )

    # Test set K-top
    output_dir = f'{args.results}/classification/Test_Graph_top_k'
    os.makedirs(output_dir, exist_ok=True)
    for name, nodes, relations, weights in zip(name_protein, t_top_k_nodes, t_top_k_relations, top_k_attention_weights):
        if name[-1] == '\x00':
            name = name[:-4]
        # Read the uploaded FASTA file and calculate the length of the second line
        fasta_file_path = f'{args.input}/FASTA/test_dataset_fasta/{name}.fasta'
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

    return t_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/HRIN-ProTstab/')
    parser.add_argument('--results', type=str, default='./results/HRIN-ProTstab/classification', )
    parser.add_argument('--data', type=str, default='HRIN-ProTstab')
    parser.add_argument('--cuda', type=bool, default=True, help='cuda or not.')

    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    joblib.dump(study, 'study.pkl')

    print(f'Best trial: {study.best_trial.value}')
    print('Best hyperparameters: ', study.best_trial.params)
