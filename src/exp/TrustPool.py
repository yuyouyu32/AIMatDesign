import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from BMGs import BMGs
from config import *
from env.environment import Environment


class TrustPool:
    def __init__(self):
        self.data = pd.read_excel(DataPath).drop(columns=DropColumns)
        # Normalize the data
        for col in TargetColumns:
            self.data[col] = (self.data[col] - self.data[col].min()) / (self.data[col].max() - self.data[col].min())
        self.data[CompositionColumns] = self.data[CompositionColumns].apply(lambda row: row / row.sum(), axis=1)
        self.env = Environment()
        self.models = self.env.models
        self.fill_na_with_models()


    def fill_na_with_models(self):
        best_models = self.env.best_models
        for target_column in TargetColumns:
            X_pred = self.data[self.data[target_column].isna()].drop(columns=TargetColumns)
            if X_pred.empty:
                continue

            y_preds = best_models[target_column].predict(X_pred.to_numpy())
            
            min_value = self.data[target_column].min()
            max_value = self.data[target_column].max()
            # adjust the predicted values to be within the range of the original data(fix bug with edRVFL predictions)
            y_preds_adjusted = [max(min(y, max_value * 1.25), min_value * 0.75) for y in y_preds]
            
            self.data.loc[self.data[target_column].isna(), target_column] = y_preds_adjusted
            
    def check_valid_a(self, row_i, row_j, threshold):
        non_zero_indices_i = np.nonzero(row_i)[0]
        non_zero_indices_j = np.nonzero(row_j)[0]
        
        if np.array_equal(non_zero_indices_i, non_zero_indices_j):
            if self.env.min_com_num <= len(non_zero_indices_i) <= self.env.max_com_num:
                diff = row_i - row_j
                abs_diff = np.abs(diff[non_zero_indices_i])
                if np.all(abs_diff <= threshold):
                    a = np.zeros(self.env.n_actions)
                    a[:len(non_zero_indices_i)] = diff[non_zero_indices_i]
                    return True, a
        return False, None


    def generate_experience_pool(self, threshold, jsonl_path, rewrite=False):
        trust_exp_pool = []
        if os.path.exists(jsonl_path):
            if not rewrite:
            # return all the data in the jsonl file
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        trust_exp_pool.append(json.loads(line))
                return trust_exp_pool
            else:
                os.remove(jsonl_path)
        # if the file does not exist or rewrite is True, generate the data and write it to the file

        data_features = self.data.drop(columns=self.models.target_columns).values

        for i, j in tqdm(combinations(range(len(data_features)), 2), total=len(data_features) * (len(data_features) - 1) // 2):
            row_i = data_features[i]
            row_j = data_features[j]
            
            keep_flag, diff = self.check_valid_a(row_i, row_j, threshold)
            if keep_flag:
                s, s_ , a = row_i.tolist(), row_j.tolist(), diff.tolist() 
                result = {col: self.data.at[i, col] for col in self.models.target_columns}
                result_ = {col: self.data.at[j, col] for col in self.models.target_columns}
                r, done = self.env.reward(np.array(s), np.array(a), np.array(s_), result, result_)
                diff_map = {
                    's': s.copy(),
                    's_': s_.copy(),
                    'a': a.copy(),
                    'r': r,
                    'done': done,
                    'result': result,
                    'result_': result_,
                    'BMG': BMGs(s).bmg_s,
                    'BMG_': BMGs(s_).bmg_s,
                }
                # Record the reverse experience
                s, s_ , a = row_j.tolist(), row_i.tolist(), (-diff).tolist()
                result = {col: self.data.at[j, col] for col in self.models.target_columns}
                result_ = {col: self.data.at[i, col] for col in self.models.target_columns}
                r, done = self.env.reward(np.array(s), np.array(a), np.array(s_), result, result_)
                diff_map_reverse = {
                    's': s.copy(),
                    's_': s_.copy(),
                    'a': a.copy(),
                    'r': r,
                    'done': done,
                    'result': result,
                    'result_': result_,
                    'BMG': BMGs(s).bmg_s,
                    'BMG_': BMGs(s_).bmg_s,
                }

                trust_exp_pool.append(diff_map)
                trust_exp_pool.append(diff_map_reverse)
                with open(jsonl_path, 'a') as f:
                    f.write(json.dumps(diff_map, ensure_ascii=False) + '\n')
                    f.write(json.dumps(diff_map_reverse, ensure_ascii=False) + '\n')

def statistic_trust_pool(path):
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]

    sorted_data = sorted(data, key=lambda x: x['r'])
    with open(path, 'w') as file:
        for item in sorted_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')
    r_values = [item['r'] for item in sorted_data]

    r_values = [item['r'] for item in sorted_data]

    plt.grid(False)
    # 平滑r的曲线
    r_values = gaussian_filter1d(r_values, sigma=20)
    # 绘制 'r' 的分布图
    plt.figure(figsize=(12, 8))
    sns.histplot(r_values, bins=20, kde=True, color='#53A6D9',edgecolor='None', alpha=0.3, linewidth=0.5)

    # 添加平均线
    mean_r = np.mean(r_values)
    plt.axvline(mean_r, color='#4F1D61', linestyle='dashed', linewidth=1)
    # plt.text(mean_r * 1.2, plt.ylim()[1] * 0.9, rf'$\mathcal{{R}}_{{\mathrm{{avg}}}}$ = {mean_r:.2f}', color='#4F1D61')

    # 设置标签和标题
    plt.xlabel('Reward Value', fontsize=20, fontweight='bold')
    plt.ylabel('Frequency', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.legend([],[], frameon=False)  # Remove the legend created by hue
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    # plt.title('Distribution of Reward Values', fontsize=16)
    plt.savefig('/data/home/yeyongyu/SHU/ReinforceMatDesign/exp_pool/trust_pool_r_distribution.pdf', dpi=800)
    
def main():
    # trust_pool = TrustPool()
    # trust_pool.generate_experience_pool(A_Scale, '/data/home/yeyongyu/SHU/ReinforceMatDesign/exp_pool/trust_pool.jsonl', rewrite=True)
    statistic_trust_pool('/data/home/yeyongyu/SHU/ReinforceMatDesign/exp_pool/trust_pool.jsonl')

# python -m exp.TrustPool
if __name__ == '__main__':
    main()