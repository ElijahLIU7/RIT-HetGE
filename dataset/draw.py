import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset.state import dataNodeNoW

sys.path.append('..')


def draw():
    dataNames = os.listdir(dataNodeNoW)  # 去水后的原子空间位置信息路径
    proteinID = os.listdir(os.path.join(dataNodeNoW, dataNames[0]))
    with open(os.path.join(dataNodeNoW, dataNames[0], proteinID[0])) as file:
        dataNoWAT = pd.read_csv(file, header=None, names=['ATOM', 'serial', 'name', 'resName', 'resSeq', 'x',
                                                          'y', 'z', 'occupancy', 'tempFactor'])
        file.close()

    xs = dataNoWAT[['x']].values.T[0]
    ys = dataNoWAT[['y']].values.T[0]
    zs = dataNoWAT[['z']].values.T[0]

    # 方式1：设置三维图形模式
    fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
    ax = plt.subplot(projection='3d')       # Axes3D(fig)    # 将画布作用于 Axes3D 对象上。
    ax.set_title('Protein Atom Graph')
    ax.scatter(xs, ys, zs, c='r')    # 画出(xs1,ys1,zs1)的散点图。
    # ax.scatter(xs2, ys2, zs2, c='r', marker='^')
    # ax.scatter(xs3, ys3, zs3, c='g', marker='*')

    ax.set_xlabel('X label')    # 画出坐标轴
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    plt.show()
