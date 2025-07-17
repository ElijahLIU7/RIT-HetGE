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

    # for char in force_types:
    #     chose2get_charHNet(args.struction_type, char)

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
    氨基酸属性AAindex
    """
    # 创建一个字典来存储每个氨基酸的属性列表
    amino_acid_properties = {aa: [] for aa in amino_acid_map.keys()}

    # 读取AAindex文件
    with open('data/HRIN-ProTstab/AAindex.txt', 'r') as file:
        content = file.read()

    # 按照属性块进行分割
    blocks = content.strip().split('//')

    # 解析每个属性块
    for block in blocks:
        # 查找I行，包含氨基酸的属性值
        lines = block.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('I'):
                # 提取属性值
                values = re.split(r'\s+', line.strip())[1:]
                index_map = {}
                # 检查值的数量是否为20
                for idx, pair in enumerate(values):
                    first, second = pair.split('/')
                    index_map[first] = (i + 1, idx)
                    index_map[second] = (i + 2, idx)
                # 解析下一行和下两行的属性值
                next_line = lines[i + 1].strip()
                next_values = re.split(r'\s+', next_line)
                next2_line = lines[i + 2].strip()
                next2_values = re.split(r'\s+', next2_line)
                # 将属性值添加到对应的氨基酸列表中
                for aa in amino_acid_map.keys():
                    if amino_acid_map[aa] in index_map.keys():
                        line_idx = index_map[amino_acid_map[aa]]
                        if line_idx[0] == (i + 1):
                            amino_acid_properties[aa].append(float(next_values[line_idx[1]]))
                        elif line_idx[0] == (i + 2):
                            amino_acid_properties[aa].append(float(next2_values[line_idx[1]]))

    # 对每个AAindex特征的20个氨基酸值进行标准化
    for feature_idx in range(len(amino_acid_properties[aa])):  # 假设每个氨基酸的属性个数相同
        # 取出20个氨基酸的同一个特征值
        feature_values = [amino_acid_properties[aa][feature_idx] for aa in amino_acid_map.keys()]

        # 计算该特征值的均值和标准差
        mean = np.mean(feature_values)
        std = np.std(feature_values)

        # 对该特征的20个氨基酸值进行标准化
        if std == 0:
            standardized_values = [value - mean for value in feature_values]
        else:
            standardized_values = [(value - mean) / std for value in feature_values]

        # 将标准化后的值写回 amino_acid_properties
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
        # AA_AA = {}
        Label = Tm
        """
        基于相互作用力维，对蛋白质残基节点进行嵌入，通过相互作用力表示异构边
        """

        # Read the uploaded FASTA file and calculate the length of the second line
        fasta_file_path = f'F:/dataset/protein/FASTA/{args.struction_type}_dataset_fasta/{tmpTy}.fasta'
        # Read the file and extract the second line
        with open(fasta_file_path, 'r') as file:
            lines = file.readlines()
            max_position = len(lines[1].strip())

        # 读取 PSSM 文件并提取矩阵
        pssm_file_path = f'F:/dataset/protein/PSSM/{args.struction_type}_pssm/{tmpTy}.pssm'
        pssm_matrix = read_pssm(pssm_file_path, max_position)
        # 在循环外部预先计算所有的 position_encoding
        position_encodings = {i: compute_position_encoding(i, max_position, 572) for i in range(max_position)}

        for force_type in force_types:
            filename = '%s.csv' % tmpTy
            filename = f'D:/program/GitHub/protein_wang/data/HNet/{args.struction_type}/{force_type}/{filename}'
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
                # 重置文件指针到开头
                fin.seek(0)
                reader = csv.reader(fin)

                for line in reader:
                    if line[-1] == 'nan':
                        continue
                    # 边的一个节点
                    position = int(line[0][2:-6]) - 1
                    type = line[0][-3:]
                    aaindex = amino_acid_properties[type]
                    pssm = pssm_matrix[position - 1].tolist()

                    # 边的另一个节点
                    re_position = int(line[2][2:-6]) - 1
                    re_type = line[2][-3:]
                    re_aaindex = amino_acid_properties[re_type]
                    re_pssm = pssm_matrix[re_position - 1].tolist()

                    # 添加节点特征
                    type, re_type = 'node', 'node'
                    if position not in node_features:
                        # 处理 PSSM 张量，确保数据为浮点型
                        pssm_array = np.array(pssm, dtype=np.float32)

                        # 对 PSSM 向量的每一列进行标准化（减去均值，除以标准差）
                        pssm = (pssm_array - np.mean(pssm_array)) / (np.std(pssm_array) + 1e-7)
                        pssm = pssm.tolist()
                        # protein_index = aaindex     # non-pssm for ablation
                        # protein_index = pssm        # non-aaindex for ablation
                        protein_index = pssm
                        protein_index += aaindex
                        # 使用预先计算好的位置编码
                        # 将位置编码加到特征向量中
                        # non-position for ablation
                        protein_index = [x + y for x, y in zip(protein_index, position_encodings[position])]
                        node_features[position] = protein_index
                        node_types[position] = type
                    if re_position not in node_features:
                        # 处理 PSSM 张量，确保数据为浮点型
                        re_pssm_array = np.array(re_pssm, dtype=np.float32)

                        # 对 PSSM 向量的每一列进行标准化（减去均值，除以标准差）
                        re_pssm = (re_pssm_array - np.mean(re_pssm_array)) / (np.std(re_pssm_array) + 1e-7)
                        re_pssm = re_pssm.tolist()
                        # re_protein_index = re_aaindex  # non-pssm for ablation
                        # re_protein_index = re_pssm  # non-aaindex for ablation
                        re_protein_index = re_pssm
                        re_protein_index += re_aaindex
                        # 将位置编码加到特征向量中
                        # non-position for ablation
                        re_protein_index = [x + y for x, y in zip(re_protein_index, position_encodings[re_position])]
                        node_features[re_position] = re_protein_index
                        node_types[re_position] = re_type

                    edge_distance = float(line[5])  # 节点之间能量
                    edge_angle = float(line[4])  # 节点之间角度

                    # if (position, force_type, re_position) not in AA_AA.keys():
                    #     AA_AA[(position, force_type, re_position)] = 1
                    # else:
                    #     continue

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
        # 添加节点特征
        for ntype in set(node_types.values()):
            ntype_nodes = [node for node, ntype_val in node_types.items() if ntype_val == ntype]
            ntype_features = torch.tensor([node_features[node] for node in ntype_nodes], dtype=torch.float32)
            ntype_ids = torch.tensor(ntype_nodes, dtype=torch.long)
            graph.nodes[ntype].data['emb'] = torch.zeros((graph.num_nodes(ntype), ntype_features.size(1)))
            graph.nodes[ntype].data['emb'][ntype_ids] = ntype_features
        # 添加边特征
        for force_type in weights.keys():
            graph.edges[force_type].data['weight'] = weights[force_type].unsqueeze(-1)
        # graph.graph_labels = torch.tensor([Label])
        graph_list.append(graph)
        labels.append(Label)
        name_protein.append(tmpTy)

    # 将字符串转换为 ASCII 编码
    encoded_name_protein = [list(map(ord, name)) for name in name_protein]
    max_len = max(len(name) for name in encoded_name_protein)
    padded_name_protein = [name + [0] * (max_len - len(name)) for name in encoded_name_protein]
    # 将编码后的数据转换为 PyTorch 张量
    name_protein_tensor = torch.tensor(padded_name_protein)

    os.makedirs(os.path.join(args.output_dir), exist_ok=True)
    file_path = os.path.join(args.output_dir, f"plddt_graphs_with_labels_{args.struction_type}_fold{fold_idx}_stand_last.bin")        # graphs_with_labels_{args.struction_type}_fold{fold_idx}_stand.bin")        # Ablation_pssm&position_ "graphsData_{args.struction_type}_fold{fold_idx}.bin"
    dgl.save_graphs(file_path, graph_list, {'labels': torch.tensor(labels), 'cv_folds': torch.tensor(cv_fold),
                                            'name_protein': name_protein_tensor})


# 读取 PSSM 文件并提取矩阵
def read_pssm(file_path, max_length):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 跳过前面的标题和空行
    data = []
    count = 0
    for line in lines:
        if line.strip() and line.startswith(('#', ' ', '\n')):
            parts = line.split()
            if len(parts) > 22 and is_number(parts[0]):  # 检查行是否包含足够的数据列
                count += 1
                data.append([int(value) for value in parts[2:22]])
                if count >= max_length:
                    break

    # 转换为 NumPy 数组
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
    选择并获取网络中不同相互作用结果，默认"VDW"
    :param char: 相互作用类型包括"VDW"、"PIPISTACK"、"IONIC"和"HBOND"
    :return: 输出在对应目录中
    """
    if char == 'VDW' or 'PIPISTACK' or 'HBOND' or 'IONIC' or 'SSBOND' or 'PICATION':
        return get_hnet(str_type, char)
    else:
        raise Exception("unknown the interaction's name!")


def read_hnet(char):
    """
    读取数据集
    :param char: 文件路径
    :return: 返回数据集中的数据
    """
    with open(char, 'r') as f:
        lines = f.read().splitlines()
        f.close()
    return lines


def get_hnet(str_type, char):
    dataFold = f'D:/program/GitHub/protein_wang/data/HNet/{str_type}_struction' # f'F:/dataset/protein/Struction/{str_type}_struction'
    dataNames = os.listdir(dataFold)
    save = f'D:/program/GitHub/protein_wang/data/HNet/{str_type}/' + char     # f'F:/dataset/protein/HNet/{str_type}/' + char

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


# 预先计算 max_position 的常量部分
def compute_position_encoding(position, max_position, dimension):
    position_encoding = torch.zeros(dimension)
    for i in range(dimension):
        if i % 2 == 0:
            position_encoding[i] = torch.sin(
                torch.tensor((position + 1) / (10000 ** (2 * (i // 2) / max_position)), dtype=torch.float32))
        else:
            position_encoding[i] = torch.cos(
                torch.tensor((position + 1) / (10000 ** (2 * (i // 2) / max_position)), dtype=torch.float32))
    return position_encoding


def main():
    # for char in ['VDW', 'PIPISTACK', 'HBOND', 'IONIC', 'SSBOND', 'PICATION']:
    #     chose2get_charHNet(args.struction_type, char)

    """
    图嵌入
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
    parser.add_argument('--output_dir', type=str, default='data/HRIN-ProTstab/preprocess')
    parser.add_argument('--data_type', type=str, default='train')
    parser.add_argument('--num_cross', type=int, default=10,
                        help='Number of cross validation')

    args = parser.parse_args()

    main()
