import os

import numpy as np
import pandas as pd
from src.dataset.state import dataNodeNoW, dataNet


def chose2get_distance(char='WAT'):
    """
    选择并获取网络中不同相互作用结果
    :param char: 相互作用类型包括“WAT”
    :return: 输出在对应目录中
    """
    if char == 'WAT' or 'PIPISTACK' or 'HBOND':
        return get_distance(char)
    else:
        raise Exception("unknown the interaction's name!")


def read_node(char):
    """
    读取数据集
    :param char: 文件路径
    :return: 返回数据集中的数据
    """
    with open(char, 'r') as f:
        lines = f.read().splitlines()
        f.close()
    return lines


def get_distance():
    """
    所有蛋白质中的原子的相对位置矩阵
    :return: 
    """
    dataNames = os.listdir(dataNodeNoW)  # 去水后的原子空间位置信息路径
    contextDataNames = dataNet + '/distance'  # 不同相互原子作用力路径
    for i in dataNames:
        proteinID = os.listdir(os.path.join(dataNodeNoW, i))
        for idx in proteinID:
            with open(os.path.join(dataNodeNoW, i, idx)) as file:
                dataNoWAT = pd.read_csv(file, header=None, names=['ATOM', 'serial', 'name', 'resName', 'resSeq', 'x',
                                                                  'y', 'z', 'occupancy', 'tempFactor'])
                file.close()
            dataNW = dataNoWAT[['x', 'y', 'z']]
            dist_matrix = np.sqrt(np.square(dataNW.values[:, np.newaxis] - dataNW.values).sum(axis=2))
            dist_dataNW = pd.DataFrame(dist_matrix, columns=dataNW.index.values.tolist(),
                                       index=dataNW.index.values.tolist())
            print(dist_dataNW)
            # n = len(dataNW)
            # matrix = [[0 for _ in range(n)] for _ in range(n)]
            # # print(type(dataNW.iloc[1][1] - dataNW.iloc[2][1]))
            # # 遍历邻接表，更新矩阵的值
            # for data in range(1000):
            #     for j in range(data + 1, 1000):
            #         if abs(dataNW.iloc[data][1] - dataNW.iloc[j][1]) > 3: continue
            #         matrix[data][j] = np.square(sum(np.square(dataNW.iloc[data][1:] - dataNW.iloc[j][1:])))
            #         matrix[j][data] = matrix[data][j]
            # print(matrix)
