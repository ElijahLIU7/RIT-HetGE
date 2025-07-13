import argparse
import pandas as pd
import numpy as np
import re
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


# 定义非线性模型方程
def melting_curve(T, a, m, T_mid):
    return (1 - a) / (1 + np.exp(-m * (1 / T - 1 / T_mid))) + a


# 读取数据
parser = argparse.ArgumentParser(description='Preprocess Lipase Data')

'''
    Dataset arguments
'''
parser.add_argument('--output_dir', type=str, default='E:/dataset/protein/output_Fold_balance_withLowTm',
                    help='The address to output.csv the preprocessed graph.')
parser.add_argument('--struction_type', type=str, default='train')
parser.add_argument('--fold_size', type=int, default=10,
                    help='Number of cross validation')

args = parser.parse_args()

Tm = pd.read_csv(f'E:/dataset/protein/{args.struction_type}_dataset.csv')
# 截取 Protein_ID 列的 "_" 之前的内容
Tm['Protein_ID'] = Tm['Protein_ID'].apply(lambda x: re.split('_|-', x)[0])
Tm_dict = pd.Series(Tm.Tm.values, index=Tm.Protein_ID).to_dict()

# 存储拟合结果和R²得分的列表
fit_results = []

# 对每个蛋白质进行拟合
for protein_id, tm in Tm_dict.items():
    # 假设我们有温度和溶解分数的数据，这里需要根据实际数据进行调整
    temperatures = tm  # 请填入实际的温度数据
    solubility = 0.5  # 请填入实际的溶解分数数据

    # 拟合曲线并计算R²得分
    try:
        params, _ = curve_fit(melting_curve, temperatures, solubility)
        a, m, T_mid = params
        fitted_curve = melting_curve(temperatures, a, m, T_mid)
        r2 = r2_score(solubility, fitted_curve)

        # 保存拟合参数和R²得分
        fit_results.append({
            'Protein_ID': protein_id,
            'a': a,
            'm': m,
            'T_mid': T_mid,
            'R²': r2
        })
    except RuntimeError:
        # 拟合失败时处理
        fit_results.append({
            'Protein_ID': protein_id,
            'a': np.nan,
            'm': np.nan,
            'T_mid': np.nan,
            'R²': np.nan
        })

# 将结果保存到CSV文件中
fit_results_df = pd.DataFrame(fit_results)
fit_results_df.to_csv(f'E:/dataset/protein/{args.struction_type}_fit_results.csv', index=False)

print("拟合结果已保存到CSV文件中。")
