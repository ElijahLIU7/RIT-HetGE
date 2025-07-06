import os
import argparse
import logging
import numpy as np
import torch
import optuna
import csv
import random

from protein_wang.pyHGT.regression_test import GraphRegressor
from protein_wang.pyHGT.utils import load_graphpred_dataset, load_graphpred_testDataset


def objective(trial):
    """
    使用optuna优化模型参数
    """
    # 定于超参数搜索空间
    print('Version: 0.0.3_HG-RIFN_regression')
    dataset = trial.suggest_categorical('dataset', ['train_autoencoder'])
    model_name = trial.suggest_categorical('model_name', [f'graphs_with_regression_{dataset}_skip_batch'])
    input = trial.suggest_categorical('input', [args.input])

    gnn = trial.suggest_categorical('gnn', ['gin'])  # 'gin', 'gcn', 'gat', 'hgt'
    num_gnn_layer = trial.suggest_int('num_gnn_layer', 2, 2)  # trial.suggest_int('num_gnn_layer', 2, 5)
    num_coders_layers = trial.suggest_int('num_coders_layers', 3, 5)
    # 先定义一个较大的范围以便 trial 进行选择
    embed_dim = trial.suggest_categorical('embed_dim_base', [512])  # trial.suggest_int('embed_dim_base', 64, 128)
    dim_a = trial.suggest_categorical('dim_a', [128])
    dropout = trial.suggest_float('dropout', 0.15, 0.30)
    activation = trial.suggest_categorical('activation', ['gelu'])  # 'relu', 'elu', 'gelu', 'selu'
    batch_size = 24  # trial.suggest_categorical('batch_size', [64, 72, 81])
    epochs = 1000
    loss_delta = trial.suggest_float('loss_delta', 10, 30)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    accum_steps = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_model_dir = f'{args.input}/{model_name}_LR{lr}'
    os.makedirs(out_model_dir, exist_ok=True)
    # for Lambda1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    Lambda1 = 1
    Lambda2 = 2 - Lambda1

    # 输出所有参数定义值
    print("Parameters:")
    print("     gnn:", gnn)
    print("     num_gnn_layer:", num_gnn_layer)
    print("     num_coders_layers:", num_coders_layers)
    print("     embed_dim:", embed_dim)
    print("     dim_a:", dim_a)
    print("     dropout:", dropout)
    print("     activation:", activation)
    print("     batch_size:", batch_size)
    print("     epochs:", epochs)
    print("     loss_delta:", loss_delta)
    print("     lr:", lr)
    print("     weight_decay:", weight_decay)
    print("     accum_steps:", accum_steps)
    print("     Lambda1:", Lambda1)
    print("     Lambda2:", Lambda2)

    log_path = f'./logs/{dataset}/{model_name}'
    log_fname = f'{log_path}/log_{gnn}_protein_wang_last.out'
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
    best_valid_losses = float('inf')
    for cv_fold in range(2, CV_FOLDS):
        print(f'CV Fold: {cv_fold}')
        with open(log_fname, 'a') as f:
            f.write('\n'.join((
                '-' * 25, f'Number of fold: {cv_fold}' + '\n'
            )))
        (train_dataset,
         valid_dataset,
         feat_dim,
         relations) = load_graphpred_dataset(args.input, CV_FOLDS, cv_select=cv_fold, bidirected=args.bidirected)

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

        train_loss, valid_loss, valid_r2, valid_mae, valid_pcc = model.train_model(
            train_dataset,
            valid_dataset,
            batch_size=batch_size,
            EPOCHS=epochs,
            lr=lr,
            weight_decay=weight_decay,
            accum_steps=accum_steps,
            num_workers=args.num_workers,
            loss_delta=loss_delta,
            Lambda1=Lambda1,
            Lambda2=Lambda2,
            device=device,
            model_dir=out_model_dir,
        )

        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                     f'Train metrics:',
                     f'Train loss: {train_loss}',
                     f'Validation metrics:',
                     f'Validation loss: {valid_loss}',
                     f'Validation R2: {valid_r2}',
                     f'Valid MAE: {valid_mae}',
                     f'Valid PCC: {valid_pcc}',
                     '-' * 25 + '\n')
                )
            )

        best_valid_loss, position = min((valid_loss, idx) for idx, valid_loss in enumerate(valid_loss))
        corr_r2 = valid_r2[position]
        corr_mae = valid_mae[position]
        corr_pcc = valid_pcc[position]

        if best_valid_loss < best_valid_losses:
            best_valid_losses = best_valid_loss

        test_dataset = load_graphpred_testDataset(args.input, bidirected=args.bidirected, is_CNN=True)
        t_loss, t_r2, t_mae, t_pcc, name_protein, (
        t_top_k_nodes, t_top_k_relations, top_k_attention_weights) = model.eval_model(
            test_dataset, batch_size=1, num_workers=args.num_workers, device=device, Is_test=True,
            model_load=out_model_dir, Is_Best_test=True
        )

        # 将结果和超参数写入CSV文件
        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                     f'Best Valid Loss: {best_valid_losses}',
                     f'Valid R2: {corr_r2}',
                     f'Valid MAE: {corr_mae}',
                     f'Valid PCC: {corr_pcc}',
                     '-' * 25,
                     f'Test Loss: {t_loss}',
                     f'Test R2: {t_r2}',
                     f'Test MAE: {t_mae}',
                     f'Test PCC: {t_pcc}'
                     '-' * 25 + '\n')
                )
            )

        # 测试集K-top
        output_dir = f'D:/program/GitHub/protein_wang/Graph_top_k'
        os.makedirs(output_dir, exist_ok=True)
        for name, nodes, relations, weights in zip(name_protein, t_top_k_nodes, t_top_k_relations,
                                                   top_k_attention_weights):
            if name[-1] == '\x00':
                name = name[:-4]
            output_csv = os.path.join(output_dir, f'graph_{name}_nodes_relations.csv')
            with open(output_csv, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(
                    ['Node', 'VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION',
                     'WEIGHT'])  # Assuming 6 relations per node
                for node, relation, weight in zip(nodes, relations, weights):
                    csvwriter.writerow([node] + relation + [weight])

        csv_file = f'fold_optuna_results_{gnn}_with_skip.csv'
        os.makedirs(args.model_dir, exist_ok=True)
        csv_file = os.path.join(args.model_dir, csv_file)
        fieldnames = ['Fold', 'model_name', 'data_dir', 'dataset', 'gnn', 'num_gnn_layer', 'embed_dim', 'dim_a',
                      'dropout', 'activation', 'batch_size', 'epochs', 'lr', 'weight_decay', 'accum_steps',
                      'Best_Valid_Loss(MSE)', 'Valid_R2', 'Valid_MAE', 'Valid_PCC', 'Test_Loss(MSE)', 'Test_R2',
                      'Test_MAE', 'Test_PCC']
        write_header = not os.path.exists(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                'Fold': cv_fold,
                'model_name': model_name,
                'data_dir': args.input,
                'dataset': dataset,
                'gnn': gnn,
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
                'Best_Valid_Loss(MSE)': best_valid_loss,
                'Valid_R2': corr_r2,
                'Valid_MAE': corr_mae,
                'Valid_PCC': corr_pcc,
                'Test_Loss(MSE)': t_loss,
                'Test_R2': t_r2,
                'Test_MAE': t_mae,
                'Test_PCC': t_pcc,
            })
        break

    return best_valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='graphs_with_labels',
                        help='Name to save the trained model.')
    parser.add_argument('--input', type=str, default='D:/program/GitHub/protein_wang/data/output_Fold_regression',
                        help='The address of preprocessed graph.')
    parser.add_argument('--dataset', type=str, default='/real')
    parser.add_argument('--model_dir', type=str, default='./model_save',
                        help='The address for storing the models and optimization results.')
    parser.add_argument('--gnn', type=str, default='gcn',
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
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of trial runs.')

    args = parser.parse_args()

    # study = optuna.create_study(
    #     storage='sqlite:///test_ProTstable2_norm_autoencoder9.19.sqlite3',
    #     study_name='muxGCN_skip',
    #     pruner=optuna.pruners.MedianPruner(
    #         n_startup_trials=0,  # 在进行前X次试验时不会进行剪枝
    #         n_warmup_steps=10  # 在每次试验的前X个步骤中不会进行剪枝
    #     ),
    #     direction='minimize'
    # )
    # study = optuna.study.load_study('muxGCN_skip', 'sqlite:///ProTstable2_norm_hessian9.9.sqlite3')
    # study = optuna.study.load_study('Fold10+1', 'sqlite:///db_7.4GCN.sqlite3')

    # 启动优化过程
    # study.optimize(objective, n_trials=args.n_trials)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)
    #
    # joblib.dump(study, 'study.pkl')

    print(f'Best trial: {study.best_trial.value}')
    print('Best hyperparameters: ', study.best_trial.params)
