import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
from Lipase.dataset.state import dataNet, dataNodeNoW, pathNet

sys.path.append('..')


def read_adj():
    for temp_data in os.listdir(os.path.join('data')):
        for temp_adj in os.listdir(os.path.join('data', temp_data)):
            data = torch.load(os.path.join('data', temp_data, temp_adj))
            print(data)
            yield data


def adj():
    for temp in os.listdir(os.path.join(dataNet, 'HBOND')):  # 由于每个特征的文件都是相同的，使用'HBOND'代替
        with tqdm(total=len(os.listdir(os.path.join(dataNet, 'HBOND', temp)))) as pbar:  # 设置进度条
            for time in os.listdir(os.path.join(dataNet, 'HBOND', temp)):
                # adj_matrix_all_features(temp, tim)
                count_f = 1
                for char in ['HBOND', 'IONIC', 'PIPISTACK', 'VDW']:
                    adj = adj_matrix(char, temp, time)
                    if count_f == 1:
                        adj_all = torch.from_numpy(adj.to_numpy()).type(torch.FloatTensor).unsqueeze(-1)  # 添加特征邻接矩阵
                        count_f += 1
                    else:
                        adj_all = torch.cat(
                            [adj_all, torch.from_numpy(adj.to_numpy()).type(torch.FloatTensor).unsqueeze(-1)],
                            dim=-1)
                count_f = 1
                if count_f == 1:
                    adj_alltime = adj_all.unsqueeze(-1)     # 增加一个时间纬度
                    count_f += 1
                else:
                    adj_alltime = torch.cat([adj_alltime, adj_all.unsqueeze(-1)], dim=-1)
                path = 'data/'
                os.makedirs(path, exist_ok=True)
                torch.save(adj_alltime, path + 'adj_' + temp + '.pth')
                pbar.set_description(temp)
                pbar.update(1)


def adj_matrix(char, temp, time):
    """
    除水的邻接矩阵
    :return: 返回一个方阵
    """
    contextDataNames = dataNet + '/' + char  # 不同相互原子作用力路径
    if char == 'VDW':
        with open(os.path.join(contextDataNames, temp, time)) as g:
            tmp = pd.read_csv(g)
            g.close()
    elif char == 'HBOND':
        with open(os.path.join(contextDataNames, temp, time)) as g:
            tmp = pd.read_csv(g)
            g.close()
    elif char == 'IONIC':
        with open(os.path.join(contextDataNames, temp, time)) as g:
            tmp = pd.read_csv(g)
            g.close()
    elif char == 'PIPISTACK':
        with open(os.path.join(contextDataNames, temp, time)) as g:
            tmp = pd.read_csv(g)
            g.close()
    else:
        raise Exception("unknown the interaction's name!")
    with open(os.path.join(dataNodeNoW, temp, time)) as file:
        dataNoWAT = pd.read_csv(file, header=None, names=['ATOM', 'serial', 'name', 'resName', 'resSeq', 'x',
                                                          'y', 'z', 'occupancy', 'tempFactor'])
        file.close()
    if char == 'WAT':
        adj = dataNoWAT[['serial', 'name', 'resName']]
    else:
        adj = create_adj(tmp, char, temp, time)
    return adj


def create_adj(mat, char, temp, time):
    """
    生成邻接矩阵
    :return:
    """
    mat.loc[mat['-1'] == -1, 'Angle'] = 0  # 将非该相互作用力的Angle赋值为0
    # 生成不对称邻接矩阵
    df = mat.pivot_table(values='Angle', index='NodeId1', columns='NodeId2', aggfunc='sum', fill_value=0)
    return df


def interaction():
    dataNames = os.listdir(pathNet)  # 去水后的原子空间位置信息路径
    dataNames.sort()
    for i in dataNames:
        if i == '.DS_Store': continue
        proteinID = os.listdir(os.path.join(pathNet, i))
        idx = proteinID[0]
        with open(os.path.join(pathNet, i, idx)) as file:
            dataNoWAT = pd.read_csv(file, header=None, names=['NodeId1', 'Interaction', 'NodeId2', 'Distance',
                                                              'Angle', 'Energy', 'Atom1', 'Atom2', 'Donor',
                                                              'Positive', 'Cation', 'Orientation'], skiprows=1,
                                    sep='\t')
            file.close()
        tmp1 = dataNoWAT['Interaction'][0][0:-6]
        print('\n' + tmp1)
        for tmp in dataNoWAT['Interaction'][1:]:
            if tmp1.find(tmp[0:-6]) == -1:
                print(tmp[0:-6])
                tmp1 += tmp[0:-6]
