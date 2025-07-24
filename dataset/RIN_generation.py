import argparse
import csv
import os
import os.path
import re
import random
import numpy as np
import torch
import dgl
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool


def protein_made(args, fold_idx, folds):
    force_types = ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']

    # 定义氨基酸三字母简写和一字母简写的对应字典
    amino_acid_map = {
        'ALA': 'A',
        'ARG': 'R',
        'ASN': 'N',
        'ASP': 'D',
        'GLN': 'Q',
        'GLU': 'E',
        'GLY': 'G',
        'HIS': 'H',
        'LEU': 'L',
        'MET': 'M',
        'PHE': 'F',
        'PRO': 'P',
        'SER': 'S',
        'THR': 'T',
        'TRP': 'W',
        'TYR': 'Y',
        'VAL': 'V',
        'LYS': 'K',
        'ILE': 'I',
        'CYS': 'C'
    }

    """
    AAindex
    """
    # Create a dictionary to store the attribute list of each amino acid.
    amino_acid_properties = {aa: [] for aa in amino_acid_map.keys()}

    # Read AAindex file
    with open('data/HRIN-ProTstab/AAindex.txt', 'r') as file:
        content = file.read()

    # Divide according to attribute blocks
    blocks = content.strip().split('//')

    # Parse each attribute block
    for block in blocks:
        lines = block.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('I'):
                values = re.split(r'\s+', line.strip())[1:]
                index_map = {}
                for idx, pair in enumerate(values):
                    first, second = pair.split('/')
                    index_map[first] = (i + 1, idx)
                    index_map[second] = (i + 2, idx)
                next_line = lines[i + 1].strip()
                next_values = re.split(r'\s+', next_line)
                next2_line = lines[i + 2].strip()
                next2_values = re.split(r'\s+', next2_line)
                for aa in amino_acid_map.keys():
                    if amino_acid_map[aa] in index_map.keys():
                        line_idx = index_map[amino_acid_map[aa]]
                        if line_idx[0] == (i + 1):
                            amino_acid_properties[aa].append(float(next_values[line_idx[1]]))
                        elif line_idx[0] == (i + 2):
                            amino_acid_properties[aa].append(float(next2_values[line_idx[1]]))

    # Standardize the 20 amino acid values for each AAindex feature
    for feature_idx in range(len(amino_acid_properties[aa])):
        # Take out the same characteristic value for 20 amino acids
        feature_values = [amino_acid_properties[aa][feature_idx] for aa in amino_acid_map.keys()]

        mean = np.mean(feature_values)
        std = np.std(feature_values)

        if std == 0:
            standardized_values = [value - mean for value in feature_values]
        else:
            standardized_values = [(value - mean) / std for value in feature_values]

        for aa_idx, aa in enumerate(amino_acid_map.keys()):
            amino_acid_properties[aa][feature_idx] = standardized_values[aa_idx]

    graph_list = []
    labels = []
    name_protein = []

    cv_fold = []
    cv_fold.extend([fold_idx] * len(folds))
    data_iter = folds
    data_iter = tqdm(data_iter, desc=f'Preprocess {args.struction_type}_Fold{fold_idx} Data', total=len(data_iter), position=0)

    for i, (original_index, tmpTy, Tm) in enumerate(data_iter):
        _graph = {}
        weights = {}
        node_features = {}
        node_types = {}
        Label = Tm
        """
        Based on the interaction force dimension, the protein residue nodes are embedded, 
        and the heterogeneous edges are represented by the interaction force.
        """

        # Read the uploaded FASTA file and calculate the length of the second line
        fasta_file_path = f'data/HRIN-ProTstab/FASTA/{args.struction_type}_dataset_fasta/{tmpTy}.fasta'
        # Read the file and extract the second line
        with open(fasta_file_path, 'r') as file:
            lines = file.readlines()
            max_position = len(lines[1].strip())

        pssm_file_path = f'data/HRIN-ProTstab/PSSM/{args.struction_type}_pssm/{tmpTy}.pssm'
        pssm_matrix = read_pssm(pssm_file_path, max_position)
        position_encodings = {i: compute_position_encoding(i, 572) for i in range(max_position)}

        for force_type in force_types:
            filename = '%s.csv' % tmpTy
            filename = f'data/HRIN-ProTstab/HNet/{args.struction_type}/{force_type}/{filename}'
            # 获取节点嵌入
            with open(filename, 'r', newline='') as fin:
                reader = csv.reader(fin)
                lines = list(reader)
                if not lines:
                    # print(f'Protein {tmpTy} doesn\'t have {force_type} edge.')
                    type, re_type = 'node', 'node'
                    _graph[(type, force_type, re_type)] = (
                        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
                    weights[(type, force_type, re_type)] = torch.empty(0, dtype=torch.float32)
                fin.seek(0)
                reader = csv.reader(fin)

                for line in reader:
                    if line[-1] == 'nan':
                        continue
                    position = int(line[0][2:-6]) - 1
                    type = line[0][-3:]
                    aaindex = amino_acid_properties[type]
                    pssm = pssm_matrix[position - 1].tolist()

                    re_position = int(line[2][2:-6]) - 1
                    re_type = line[2][-3:]
                    re_aaindex = amino_acid_properties[re_type]
                    re_pssm = pssm_matrix[re_position - 1].tolist()

                    if position not in node_features:
                        pssm_array = np.array(pssm, dtype=np.float32)
                        pssm = (pssm_array - np.mean(pssm_array)) / (np.std(pssm_array) + 1e-7)
                        pssm = pssm.tolist()
                        # protein_index = aaindex     # non-pssm for ablation
                        # protein_index = pssm        # non-aaindex for ablation
                        aaindex_array = np.array(aaindex, dtype=np.float32)
                        aaindex = (aaindex_array - np.mean(aaindex_array)) / (np.std(aaindex_array) + 1e-7)
                        aaindex = aaindex.tolist()
                        protein_index = pssm
                        protein_index += aaindex
                        # non-position for ablation
                        protein_index = [x + y for x, y in zip(protein_index, position_encodings[position])]
                        node_features[position] = protein_index
                        node_types[position] = type
                    if re_position not in node_features:
                        re_pssm_array = np.array(re_pssm, dtype=np.float32)
                        re_pssm = (re_pssm_array - np.mean(re_pssm_array)) / (np.std(re_pssm_array) + 1e-7)
                        re_pssm = re_pssm.tolist()
                        # re_protein_index = re_aaindex  # non-pssm for ablation
                        # re_protein_index = re_pssm  # non-aaindex for ablation
                        re_aaindex_array = np.array(aaindex, dtype=np.float32)
                        re_aaindex = (re_aaindex_array - np.mean(re_aaindex_array)) / (np.std(re_aaindex_array) + 1e-7)
                        re_aaindex = re_aaindex.tolist()
                        re_protein_index = re_pssm
                        re_protein_index += re_aaindex
                        # non-position for ablation
                        re_protein_index = [x + y for x, y in zip(re_protein_index, position_encodings[re_position])]
                        node_features[re_position] = re_protein_index
                        node_types[re_position] = re_type

                    edge_distance = float(line[5])

                    if (type, force_type, re_type) not in _graph.keys():
                        _graph[(type, force_type, re_type)] = (torch.tensor([position]), torch.tensor([re_position]))
                        weights[(type, force_type, re_type)] = torch.tensor([edge_distance])
                    else:
                        _graph[(type, force_type, re_type)] = (
                            torch.cat([_graph[(type, force_type, re_type)][0], torch.tensor([position])]),
                            torch.cat([_graph[(type, force_type, re_type)][1], torch.tensor([re_position])]))
                        weights[(type, force_type, re_type)] = torch.cat(
                            [weights[(type, force_type, re_type)], torch.tensor([edge_distance])])

        graph = dgl.heterograph(_graph)
        for ntype in set(node_types.values()):
            ntype_nodes = [node for node, ntype_val in node_types.items() if ntype_val == ntype]
            ntype_features = torch.tensor([node_features[node] for node in ntype_nodes], dtype=torch.float32)
            ntype_ids = torch.tensor(ntype_nodes, dtype=torch.long)
            graph.nodes[ntype].data['emb'] = torch.zeros((graph.num_nodes(ntype), ntype_features.size(1)))
            graph.nodes[ntype].data['emb'][ntype_ids] = ntype_features
        for force_type in weights.keys():
            graph.edges[force_type].data['weight'] = weights[force_type].unsqueeze(-1)
        graph_list.append(graph)
        labels.append(Label)
        name_protein.append(tmpTy)

    encoded_name_protein = [list(map(ord, name)) for name in name_protein]
    max_len = max(len(name) for name in encoded_name_protein)
    padded_name_protein = [name + [0] * (max_len - len(name)) for name in encoded_name_protein]
    name_protein_tensor = torch.tensor(padded_name_protein)

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    file_path = os.path.join(args.output_dir, f"{args.struction_type}_fold{fold_idx}.bin")
    dgl.save_graphs(file_path, graph_list, {'labels': torch.tensor(labels), 'cv_folds': torch.tensor(cv_fold),
                                            'name_protein': name_protein_tensor})


# Read the PSSM file and extract the matrix
def read_pssm(file_path, max_length):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    count = 0
    for line in lines:
        if line.strip() and line.startswith(('#', ' ', '\n')):
            parts = line.split()
            if len(parts) > 22 and is_number(parts[0]):
                count += 1
                data.append([int(value) for value in parts[2:22]])
                if count >= max_length:
                    break

    matrix = np.array(data)
    return matrix


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def chose2get_charHNet(str_type, char='VDW'):
    """
    Select and obtain different interaction results in the network, default 'VDW'.
    :param char: Interaction types include 'VDW', 'PIPISTACK', 'IONIC', 'HBOND', 'SSBOND', and 'PICATION'.
    :return: Output in the corresponding directory.
    """
    if char == 'VDW' or 'PIPISTACK' or 'HBOND' or 'IONIC' or 'SSBOND' or 'PICATION':
        return get_hnet(str_type, char)
    else:
        raise Exception("unknown the interaction's name!")


def read_hnet(char):
    """
    Read the dataset
    :param char: File path
    :return: Return the data in the dataset.
    """
    with open(char, 'r') as f:
        lines = f.read().splitlines()
        f.close()
    return lines


def get_hnet(str_type, char):
    dataFold = f'data/HRIN-ProTstab/{str_type}_struction'
    dataNames = os.listdir(dataFold)
    save = f'data/HRIN-ProTstab/HNet/{str_type}/' + char

    for i in tqdm(dataNames):
        if i == '.DS_Store': continue
        lines = read_hnet(os.path.join(dataFold, i))
        atomVDW = []
        for k in lines:
            if k.find(char) != -1:  # 查找WAT开始行
                k = k.split()[0:6]
                atomVDW += [k]
        os.makedirs(os.path.join(save), exist_ok=True)
        with open(os.path.join(save, i[:-4]) + '.csv', 'w+', newline='') as g:
            writer = csv.writer(g)
            for row in atomVDW:
                writer.writerow(row)
            g.close()
    print(str_type, char, " Writing successful[OK]")


# Pre-calculate the constant part of max_position
def compute_position_encoding(position, dimension):
    position_encoding = torch.zeros(dimension)
    for i in range(dimension):
        if i % 2 == 0:
            position_encoding[i] = torch.sin(
                torch.tensor((position) / (10000 ** (2 * (i // 2) / dimension)), dtype=torch.float32))
        else:
            position_encoding[i] = torch.cos(
                torch.tensor((position) / (10000 ** (2 * (i // 2) / dimension)), dtype=torch.float32))
    return position_encoding


def main():

    for char in ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']:
        chose2get_charHNet(args.struction_type, char)

    """
    graph embedding
    """
    struction_idx = f'data/HRIN-ProTstab/{args.struction_type}_pssm'

    Type_PSSM = [type[:-5] for type in os.listdir(struction_idx)]

    Type = list(set(Type_PSSM))      # & set(Type_IDs))

    """
    Read Tm information
    """
    Tm = pd.read_csv(f'data/HRIN-ProTstab/{args.struction_type}_dataset.csv')
    # Extract the content before the "_" in the Protein_ID column
    Tm['Protein_ID'] = Tm['Protein_ID'].apply(lambda x: re.split('_|-', x)[0])
    Tm_dict = pd.Series(Tm.Tm.values, index=Tm.Protein_ID).to_dict()

    fold = args.fold_size
    plddt_data = pd.read_csv(f'data/HRIN-ProTstab/{args.struction_type}_plddt.csv')
    # Create a sample list
    all_samples = []
    for i, tmpTy in enumerate(Type):
        if tmpTy[-1] == '\x00':
            tmpTy_tmp = tmpTy[:-4]
        else:
            tmpTy_tmp = tmpTy
        plddt = plddt_data[plddt_data.iloc[:, 0] == tmpTy_tmp].iloc[0, 1]
        if plddt <= 70:
            continue
        all_samples.append((i, tmpTy, Tm_dict[tmpTy]))
    # Disrupt the sample order
    random.shuffle(all_samples)
    # Allocate the samples to each fold
    if args.struction_type == 'test':
        folds = all_samples

        protein_made(args, 0, folds)

    else:
        fold_size = len(all_samples) // fold
        folds = [all_samples[i * fold_size:(i + 1) * fold_size] for i in range(fold)]
        for i in range(args.fold_size // 2):
            with Pool(2) as pool:
                pool.starmap(protein_made,
                             [(args, fold_idx, folds[fold_idx]) for fold_idx in range(i*2, (i+1)*2)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess HRIN-ProTstab Data.')

    '''
        Dataset arguments
    '''
    parser.add_argument('--output_dir', type=str, default='data/HRIN-ProTstab/')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--num_cross', type=int, default=10,
                        help='Number of cross validation')

    args = parser.parse_args()

    main()
