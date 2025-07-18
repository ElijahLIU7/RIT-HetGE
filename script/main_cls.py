import os
import argparse
import logging
import torch
import optuna
import csv
import re
import pandas as pd

from HGRIFN.HG_RIFN_cls import GraphClassifier
from HGRIFN.utils import load_dataset, load_testDataset


def objective(trial):
    """
    Use Optuna to optimize model parameters
    """

    print('Version: RIT-HetGE_classification')
    data = args.data
    model_name = f'classification_{data}'
    # Defined in the hyperparameter search space
    num_gnn_layer = trial.suggest_int('num_gnn_layer', 2, 5)
    num_coders_layers = trial.suggest_int('num_coders_layers', 2, 5)
    embed_dim = trial.suggest_categorical('embed_dim_base', [256, 512, 768])
    dim_a = trial.suggest_categorical('dim_a', [8, 12, 16])
    dropout = trial.suggest_float('dropout', 0.15, 0.30)
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'gelu', 'selu', 'softmax'])
    batch_size = trial.suggest_int('batch_size', [64, 128])
    epochs = 1000
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    accum_steps = trial.suggest_int('accum_steps', 1, 5)

    args.cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if args.cuda else 'cpu')
    out_model_dir = f'{args.output}/Best_{model_name}_result'
    os.makedirs(out_model_dir, exist_ok=True)
    # for Lambda1 in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    Lambda1 = 1
    Lambda2 = 2 - Lambda1

    # 输出所有参数定义值
    print("Parameters:")
    print("     num_gnn_layer:", num_gnn_layer)
    print("     num_coders_layers:", num_coders_layers)
    print("     embed_dim:", embed_dim)
    print("     dim_a:", dim_a)
    print("     dropout:", dropout)
    print("     activation:", activation)
    print("     batch_size:", batch_size)
    print("     epochs:", epochs)
    print("     lr:", lr)
    print("     weight_decay:", weight_decay)
    print("     accum_steps:", accum_steps)
    print("     Lambda1:", Lambda1)
    print("     Lambda2:", Lambda2)

    log_path = f'./logs/{model_name}'
    log_fname = f'{log_path}/log.out'
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=log_fname,
        filemode='a'
    )
    logging.info(f'Starting trial with parameters: {trial.params}')
    logging.info(f'Lambda1:{Lambda1};Lambda2:{Lambda2}')

    CV_FOLDS = 10
    best_ACCs = 0
    test_dataset = load_testDataset(args.input, is_classification=True)
    for cv_fold in range(CV_FOLDS):
        print(f'CV Fold: {cv_fold}')
        with open(log_fname, 'a') as f:
            f.write('\n'.join((
                '-' * 25, f'Number of fold: {cv_fold}' + '\n'
            )))
        (train_dataset,
         valid_dataset,
         feat_dim,
         relations) = load_dataset(args.input, CV_FOLDS, cv_select=cv_fold, is_classification=True)

        model = GraphClassifier(
            num_gnn_layers=num_gnn_layer,
            num_coder_layers=num_coders_layers,
            relations=relations,
            feat_dim=feat_dim,
            embed_dim=embed_dim,
            dim_a=dim_a,
            dropout=dropout,
            activation=activation
        )

        test_acc, test_p, test_r, test_f1 = model.train_model(
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            EPOCHS=epochs,
            lr=lr,
            weight_decay=weight_decay,
            accum_steps=accum_steps,
            num_workers=args.num_workers,
            Lambda1=Lambda1,
            Lambda2=Lambda2,
            device=device,
            model_dir=out_model_dir,
        )

        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                     f'Test Accuracies: {test_acc}',
                     f'Test Precisions: {test_p}',
                     f'Test Recalls: {test_r}',
                     f'Test F1s: {test_f1}',
                     '-' * 25 + '\n')
                )
            )

        best_ACC, position = max((test_acc, idx) for idx, test_acc in enumerate(test_acc))
        corr_p = test_p[position]
        corr_r = test_r[position]
        corr_f1 = test_f1[position]
        if best_ACC > best_ACCs:
            best_ACCs = best_ACC

        t_acc, t_p, t_r, t_f1, t_auc, name_protein, (t_top_k_nodes, t_top_k_relations, top_k_attention_weights) = model.eval_model(
            test_dataset, batch_size=1, num_workers=args.num_workers, device=device, Is_test=True,
            model_load=out_model_dir, Is_Best_test=False
        )

        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                     f'Best accuracy: {best_ACC}',
                     f'Precision: {corr_p}',
                     f'Recall: {corr_r}',
                     f'F1: {corr_f1}',
                     '-' * 25,
                     f'Test Accuracies: {t_acc}',
                     f'Test Precisions: {t_p}',
                     f'Test Recalls: {t_r}',
                     f'Test F1s: {t_f1}',
                     f'Test AUCs:{t_auc}',
                     '-' * 25 + '\n')
                )
            )

        # Test set K-top nodes
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
        output_dir = f'{args.results}/Test_Graph_top_k'
        os.makedirs(output_dir, exist_ok=True)
        Tm = pd.read_csv(f'data/HRIN-ProTstab/test_dataset.csv')
        # Extract the content before the "_" in the Protein_ID column
        Tm['Protein_ID'] = Tm['Protein_ID'].apply(lambda x: re.split('_|-', x)[0])
        Tm_dict = pd.Series(Tm.Tm.values, index=Tm.Protein_ID).to_dict()
        for name, nodes, relations, weights in zip(name_protein, t_top_k_nodes, t_top_k_relations,
                                                   top_k_attention_weights):
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

        csv_file = f'{args.results}/fold_optuna_results_regression.csv'
        os.makedirs(args.model_dir, exist_ok=True)
        csv_file = os.path.join(args.model_dir, csv_file)
        fieldnames = ['Fold', 'model_name', 'data_dir', 'dataset', 'gnn', 'num_gnn_layer', 'embed_dim', 'dim_a',
                      'dropout', 'activation', 'batch_size', 'epochs', 'lr', 'weight_decay', 'accum_steps',
                      'Best_accuracy', 'precision', 'recall', 'f1', 'Test_accuracy', 'Test_precision', 'Test_recall',
                      'Test_f1']
        write_header = not os.path.exists(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'Fold': cv_fold,
                'model_name': model_name,
                'data_dir': args.input,
                'num_gnn_layer': num_gnn_layer,
                'embed_dim': embed_dim,
                'dim_a': dim_a,
                'dropout': dropout,
                'activation': activation,
                'batch_size': batch_size,
                'epochs': epochs,
                'lr': lr,
                'weight_decay': weight_decay,
                'accum_steps': accum_steps,
                'Best_accuracy': best_ACC,
                'precision': corr_p,
                'recall': corr_r,
                'f1': corr_f1,
                'Test_accuracy': t_acc,
                'Test_precision': t_p,
                'Test_recall': t_r,
                'Test_f1': t_f1
            })
    return t_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/HRIN-ProTstab/')
    parser.add_argument('--results', type=str, default='./results/HRIN-ProTstab/classification/', )
    parser.add_argument('--data', type=str, default='HRIN-ProTstab')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trial runs.')
    parser.add_argument('--cuda', type=bool, default=True, help='cuda or not.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes.')

    args = parser.parse_args()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)

    print(f'Best trial: {study.best_trial.value}')
    print('Best hyperparameters: ', study.best_trial.params)
