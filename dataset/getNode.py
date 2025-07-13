import os
import csv
from src.dataset.state import pathNode


def chose2get_charNode(char='WAT'):
    """
    选择并获取网络中不同相互作用结果
    :param char: 相互作用类型包括“WAT”
    :return: 输出在对应目录中
    """
    if char == 'WAT' or 'PIPISTACK' or 'HBOND':
        return get_node(char)
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


def get_node(char):
    dataNames = os.listdir(pathNode)
    dataNames.sort()
    save = '../dataset/node/' + char
    for i in dataNames:
        # allAtom = []
        if i == '.DS_Store': continue
        proteinID = os.listdir(os.path.join(pathNode, i))
        proteinID.sort()
        for j in proteinID:
            if j == '.DS_Store': continue
            lines = read_node(os.path.join(pathNode, i, j))
            atomVDW = []
            for k in lines:
                if k.find(char) == -1:  # 查找WAT开始行
                    atomVDW += [k.split()]
            atomVDW[0], atomVDW[-1] = [], []
            atomVDW = [x for x in atomVDW if x]
            # allAtom += [atomVDW]
            os.makedirs(os.path.join(save, i), exist_ok=True)
            with open(os.path.join(save, i, j) + '.csv', 'w+', newline='') as g:
                writer = csv.writer(g)
                for row in atomVDW:
                    writer.writerow(row)
                g.close()
        print(i, " Writing successful[OK]")
