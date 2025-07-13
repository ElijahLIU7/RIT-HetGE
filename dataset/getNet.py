import os
import csv
from src.dataset.state import pathNet


def chose2get_charNet(char='VDW'):
    """
    选择并获取网络中不同相互作用结果，默认‘VDW’
    :param char: 相互作用类型包括“VDW”、”PIPISTACK“、“IONIC”和”HBOND“
    :return: 输出在对应目录中
    """
    if char == 'VDW' or 'PIPISTACK' or 'HBOND' or 'IONIC':
        return get_net(char)
    else:
        raise Exception("unknown the interaction's name!")


def read_net(char):
    """
    读取数据集
    :param char: 文件路径
    :return: 返回数据集中的数据
    """
    with open(char, 'r') as f:
        lines = f.read().splitlines()
        f.close()
    return lines


def get_net(char):
    dataNames = os.listdir(pathNet)
    save = '../dataset/net/' + char
    for i in dataNames:
        if i == '.DS_Store': continue
        proteinID = os.listdir(os.path.join(pathNet, i))
        proteinID.sort()
        for j in proteinID:
            lines = read_net(os.path.join(pathNet, i, j))
            atomVDW = []
            for k in lines:
                if k.find(char) != -1:  # 查找WAT开始行
                    k = k.split()[0:5]
                    k.append('1')
                    atomVDW += [k]
                else:
                    k = k.split()[0:5]
                    k.append('-1')
                    atomVDW += [k]
            os.makedirs(os.path.join(save, i), exist_ok=True)
            with open(os.path.join(save, i, j[:-4]) + '.csv', 'w+', newline='') as g:
                writer = csv.writer(g)
                for row in atomVDW:
                    writer.writerow(row)
                g.close()
        print(i, char, " Writing successful[OK]")
