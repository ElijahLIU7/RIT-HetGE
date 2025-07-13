import os
import argparse
import logging
import torch
import optuna
import csv

from HGRIFN.HG_RIFN_reg import GraphRegressor
from HGRIFN.utils import load_graphpred_dataset, load_graphpred_testDataset


def objective(trial):
    """
    Use Optuna to optimize model parameters
    """

    print('Version: HG-RIFN_regression')
    data = args.data
    model_name = f'regression_{data}'
    # Defined in the hyperparameter search space
    num_gcn_layer = trial.suggest_int('num_gcn_layer', 2, 5)
    num_coders_layers = trial.suggest_int('num_coders_layers', 2, 5)
    embed_dim = trial.suggest_categorical('embed_dim_base', [256, 512, 768])
    dim_a = trial.suggest_categorical('dim_a', [8, 12, 16])
    dropout = trial.suggest_float('dropout', 0.15, 0.30)
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'gelu', 'selu', 'softmax'])
    batch_size = trial.suggest_int('batch_size', [64, 128])
    epochs = 1000
    loss_delta = trial.suggest_float('loss_delta', 10, 30)
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    accum_steps = trial.suggest_int('accum_steps', 1, 5)

    args.cuda = torch.cuda.is_available() and args.cuda
    device = torch.device('cuda' if args.cuda else 'cpu')
    out_model_dir = f'{args.input}/{model_name}_LR{lr}'
    os.makedirs(out_model_dir, exist_ok=True)
    # for Lambda1 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    Lambda1 = 1
    Lambda2 = 2 - Lambda1

    # Output all parameter definition values
    print("Parameters:")
    print("     num_gcn_layer:", num_gcn_layer)
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

    log_path = f'./logs/{model_name}'
    log_fname = f'{log_path}/log_HG-RIFN.out'
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

        model = GraphRegressor(num_gnn_layers=num_gcn_layer, num_coder_layers=num_coders_layers, relations=relations,
                               feat_dim=feat_dim, embed_dim=embed_dim, dim_a=dim_a, dropout=dropout,
                               activation=activation)

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

        # Write the results and hyperparameters to a CSV file.
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

        # Test set K-top
        output_dir = f'{args.results}/{data}/Graph_top_k'
        os.makedirs(output_dir, exist_ok=True)
        for name, nodes, relations, weights in zip(name_protein, t_top_k_nodes, t_top_k_relations,
                                                   top_k_attention_weights):
            if name[-1] == '\x00':
                name = name[:-4]
            output_csv = os.path.join(output_dir, f'protein_{name}_nodes_relations.csv')
            with open(output_csv, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(
                    ['Node', 'VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION',
                     'WEIGHT'])  # Assuming 6 relations per node
                for node, relation, weight in zip(nodes, relations, weights):
                    csvwriter.writerow([node] + relation + [weight])

        csv_file = f'{data}/fold_optuna_results_regression.csv'
        os.makedirs(args.results, exist_ok=True)
        csv_file = os.path.join(args.model_dir, csv_file)
        fieldnames = ['Fold', 'model_name', 'data_dir', 'dataset', 'gnn', 'num_gcn_layer', 'embed_dim', 'dim_a',
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
                'num_gcn_layer': num_gcn_layer,
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

    return best_valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='./data/HGEProTstab',
                        help='The address of preprocessed graph.')
    parser.add_argument('--results', type=str, default='./results',)
    parser.add_argument('--data', type=str, default='HGEProTstab')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trial runs.')
    parser.add_argument('--cuda', type=bool, default=True, help='cuda or not.')

    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.n_trials)

    print(f'Best trial: {study.best_trial.value}')
    print('Best hyperparameters: ', study.best_trial.params)
