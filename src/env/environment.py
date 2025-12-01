import math
import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from BMGs import BMGs
from config import *
from env.ML_model import ML_Model
from env.Cls_model import Cls_Model
from config import logging, MaxStep
import json


logger = logging.getLogger(__name__)


class Environment:
    def __init__(self) -> None:
        self.n_actions = N_Action
        self.n_states = N_State
        # Thresholds
        self.matrix_thresholds, self.exist_bmgs = self.define_matrix_thresholds(DataPath, Percentile)
        # Models
        self.models = ML_Model(DataPath, DropColumns, TargetColumns, MLResultPath)
        self.best_models = self.models.get_best_models(norm_features=True, norm_target=True)
        self.cls_model = Cls_Model(ClsPath, ClsDropColumns, ClsTargetColumns, ClsResultpath).get_best_models(norm_features=True)
        # Record new BMGs
        self.new_bmgs = []
        # Initial Pool and constraints
        self.original_data = pd.read_excel(DataPath)
        bmgs_index = self.original_data[self.original_data['cls_label'] == 1].index 
        
        self.features = self.original_data.drop(columns=DropColumns).drop(columns=TargetColumns)
        # Normalize the init_data row-wise
        self.features = self.features.apply(lambda x: x / x.sum(), axis=1)
        self.init_pool = self.features.values[bmgs_index]
        
        # Init Base Matrix config
        self.init_base_matrix = self.get_base_matrix_bases(BaseMatrixPath)
        
        self.max_com_num = min(self.features.astype(bool).sum(axis=1).max(), N_Action)
        self.min_com_num = max(self.features.astype(bool).sum(axis=1).min(), 3)
        
        self.env_step = 0
        # Record the min and max values of the target columns
        self.minmax_record = {}
        for target_clomuns in TargetColumns:
            self.minmax_record[target_clomuns] = self.original_data[target_clomuns].min(), self.original_data[target_clomuns].max()
        # Record SR_Percentile
        self.sr_percentile = {'all': 0, 'all_regression': 0, 'all_classification': 0, 'cls': 0}
        del self.original_data
        del self.features
        del bmgs_index
        
    
    @staticmethod
    def define_matrix_thresholds(excel_file: str, percentile: float = 0.8, norm_target: bool = True) -> Tuple[Dict, Dict]:
        """
        Define the matrix thresholds for each target property.
        
        Parameters:
            excel_file (str): path to the Excel file
            percentile (float): percentile for the threshold
        
        Returns:
            matrix_thresholds (dict): matrix thresholds for each target property
            exist_bmgs (list): list of existing BMGs
        """
        df = pd.read_excel(excel_file)
        # Filter the BMGs
        # df = df[df['cls_label'] == 1]
        # Get the unique existing BMGs
        exist_bmgs = {composition: 0 for composition in df['Chemical composition'].unique().tolist()}
        # Identify the base matrix for each row
        df = df.drop(columns=DropColumns)
        df['Tg/Tl'] = df['Tg(K)'] / df['Tl(K)']
        # normalize the target_columns
        if norm_target:
            df[TargetColumns] = df[TargetColumns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            
        target_columns = TargetColumns + ['Tg/Tl']
        feature_df = df.drop(columns=target_columns)
        max_values = feature_df.max(axis=1)
        df['Base_Matrix'] = feature_df.apply(
        lambda row: row[row == max_values[row.name]].index.tolist(),
        axis=1
        )
        df = df.explode('Base_Matrix').reset_index(drop=True)
        # Use the Base_Matrix values to groupby, then use the percentile to find the percentile of each group, only for TargetColumns, to get a thresholds dict
        matrix_thresholds = {group: data[target_columns].quantile(percentile).to_dict() for group, data in df.groupby('Base_Matrix')}
        matrix_thresholds = pd.DataFrame(matrix_thresholds).T
        # Add a new row for new BMGs with a specified data type
        matrix_thresholds.loc['New'] = pd.Series(dtype=float)
        # Fill the NaN values with the median of the column
        matrix_thresholds_filled = matrix_thresholds.apply(lambda x: x.fillna(x.median()), axis=0)
        matrix_thresholds_filled = matrix_thresholds_filled.to_dict(orient='index')

        return matrix_thresholds_filled, exist_bmgs

    @staticmethod
    def get_base_matrix_bases(json_path) -> Dict[str, Tuple]:
        with open(json_path, 'r') as f:
            base_matrix_bases = json.load(f)
        return base_matrix_bases
    
    def cls_func(self, s: np.ndarray) -> int:
        """
        Get the classification label of a BMGs.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            int: classification label
        """
        X = s.reshape(1, -1)
        return self.cls_model.predict_proba(X)[0][1]
    
    def target_func(self, s: np.ndarray) -> Dict:
        """
        Get the target properties of a BMGs.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            dict: target properties
        """
        X = s.reshape(1, -1)
        results = {}
        for target, model in self.best_models.items():
            results[target] = max(model.predict(X)[0], 0)
        return results
    
    def step(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Take a step in the environment.
        
        Parameters:
            s (np.ndarray): state vector
            a (np.ndarray): action vector
            indexs (np.ndarray): indices of the non-zero elements in the state vector
        
        Returns:
            np.ndarray: new state vector
        """
        s_ = s.copy()
        indexs = np.where(s != 0)[0]
        assert len(a) >= len(indexs)
        s_[indexs] = s_[indexs] + a[:len(indexs)]
        self.env_step += 1
        reward, done = self.reward(s, a, s_)
        return s_, reward, done
    
    def get_random_action(self, s: np.ndarray) -> np.ndarray:
        """
        Get a random action vector.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            np.ndarray: action vector
        """
        if not self.judge_s(s):
            raise ValueError("Invalid state vector {s}, Please check the state vector.")
        k = np.count_nonzero(s)
        a = np.zeros(N_Action)
        action = np.random.rand(k)
        action = (action - action.mean()) * (A_Scale)
        a[:k] = action
        return a
    
    def get_random_legal_action(self, s: np.ndarray) -> np.ndarray:
        """
        Get a random legal action vector.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            np.ndarray: action vector
        """
        if not self.judge_s(s):
            raise ValueError("Invalid state vector {s}, Please check the state vector.")
        iter = 1
        k = np.count_nonzero(s)
        while True:
            a = np.zeros(N_Action)
            action = np.random.rand(k)
            action = (action - action.mean()) * (A_Scale /iter)
            a[:k] = action
            s_ = self.step(s, a, 0)[0]
            if self.judge_a(a) and self.judge_s(s_):
                break
            iter += 1
        return a
    
    def reset(self):
        """
        Reset the state vector randomly. (In the initial pool)
        
        Returns:
            np.ndarray: state vector
        """
        s = self.init_pool[np.random.randint(0, self.init_pool.shape[0])]
        indexs = np.where(s != 0)[0]
        if len(indexs) < self.min_com_num or len(indexs) > self.max_com_num:
            s = self.reset() # Reset if the number of elements is out of range
        # 50% chance to replace the element with the smallest value
        if np.random.rand() > 0.5:
            s = self.replace_element(s)
            
        self.env_step = 0
        return s
    
    def reset_by_random_state(self, random_state: np.random.RandomState) -> np.ndarray:
        """
        Reset the state vector randomly. (In the initial pool)
        
        Returns:
            np.ndarray: state vector
        """
        s = self.init_pool[random_state.randint(0, self.init_pool.shape[0])]
        indexs = np.where(s != 0)[0]
        if len(indexs) < self.min_com_num or len(indexs) > self.max_com_num:
            s = self.reset() # Reset if the number of elements is out of range
        # 50% chance to replace the element with the smallest value
        if random_state.rand() > 0.5:
            s = self.replace_element(s)
            
        self.env_step = 0
        return s
    
    def replace_element(self, s: np.ndarray) -> np.ndarray:
        """
        Replace the element with the smallest value in the state vector with an element from the optional rest elements.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            np.ndarray: state vector
        """
        indexs = np.where(s != 0)[0]
        min_index = np.argmin(s[indexs])
        temp_value = s[indexs[min_index]]
        s[indexs[min_index]] = 0
        # TypeError: only integer scalar arrays can be converted to a scalar index
        rest_elements = OptionalResetElement - set(np.array(CompositionColumns)[indexs])
        new_element = random.choice(list(rest_elements))
        s[CompositionColumns.index(new_element)] = temp_value
        return s
      
    def reset_by_constraint(self, mandatory_elements: Dict[str, Tuple[int]], optional_elements: Dict[str, Tuple[int]], k: int, replace_flag = True, min_optional_len: int = 0) -> np.ndarray:
        """
        Reset the state vector based on the constraints.
        
        Parameters:
            mandatory_elements (dict): mandatory elements with range
                {
                    'Zr': (40, 70),
                    'Cu': (10, 25),
                    'Ni': (5, 15),
                    'Al': (5, 15)
                }
            optional_elements (dict): optional elements with range
                {
                    'Ag': (0, 10),
                    'Ti': (0, 10),
                    'La': (0, 10),
                    'Ce': (0, 10),
                    'Gd': (0, 10),
                    'Y': (0, 10)
                }
            k (int): number of optional elements with range
        
        Returns:
            np.ndarray: state vector
        """
        if len(mandatory_elements) > self.max_com_num:
            raise ValueError("The number of mandatory elements is out of range.")
        k = min(k, self.max_com_num)
        # Step 0: Randomly select optional elements
        optional_len = np.random.randint(max(self.min_com_num - len(mandatory_elements), 0), k - len(mandatory_elements) + 1)
        optional_len = max(optional_len, min_optional_len)
        # Step 1: Randomly select optional elements
        selected_optional_elements = random.sample(list(optional_elements.keys()), optional_len)
        
        # Step 2: Generate random values for mandatory and selected optional elements
        state = {}
        
        for element, (low, high) in mandatory_elements.items():
            state[element] = np.round(np.random.uniform(low, high), 4)
        
        for element in selected_optional_elements:
            low, high = optional_elements[element]
            state[element] = np.round(np.random.uniform(low, high), 4)
        
        # Step 3: Normalize the values to sum to 1
        total = sum(state.values())
        for element in state:
            state[element] = np.round((state[element] / total), 4)
        assert len(state) >= self.min_com_num and len(state) <= self.max_com_num
        # Step 4: Create the final state vector
        state_vector = np.zeros(len(CompositionColumns))
        for element, value in state.items():
            index = CompositionColumns.index(element)
            state_vector[index] = value
        if replace_flag and np.random.rand() > 0.5:
            state_vector = self.replace_element(state_vector)
        self.env_step = 0
        return state_vector
    
    def set_s_by_dict(self, s_dict: Dict[str, float]) -> np.ndarray:
        """
        Set the state vector based on the given dictionary.
        
        Parameters:
            s_dict (dict): dictionary of elements and their fractions
        
        Returns:
            np.ndarray: state vector
        """
        s = np.zeros(len(CompositionColumns))
        # 对s_dict中的元素进行归一化
        total = sum(s_dict.values())
        for element in s_dict:
            s_dict[element] = s_dict[element] / total
        for element, value in s_dict.items():
            if element in CompositionColumns:
                index = CompositionColumns.index(element)
                s[index] = value
        if not self.judge_s(s):
            raise ValueError("Invalid state vector {s}, Please check the state vector.")
        self.env_step = 0
        return s

    def judge_s(self, s: np.ndarray) -> bool:
        """
        Judge the state vector.
        
        Parameters:
            s (np.ndarray): state vector
        
        Returns:
            bool: True if the state vector is valid, False otherwise
        """
        if abs(sum(s) - 1) > 0.025 or s.min() < 0 or s.max() > 1:
            return False
        return True
    
    def judge_a(self, a: np.ndarray) -> bool:
        """
        Judge the action vector.
        
        Parameters:
            s (np.ndarray): state vector
            a (np.ndarray): action vector
        
        Returns:
            bool: True if the action vector is valid, False otherwise
        """
        if a.min() < -A_Scale or a.max() > A_Scale:
            return False
        return True

    def inverse_normal_targets(self, target_name, target_value):
        """
        Inverse the normalization of the target value.
        
        Parameters:
            target_name (str): target name
            target_value (float): target value
        
        Returns:
            float: inverse normalized target value
        """
        # df[target_columns] = df[target_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        min_value, max_value = self.minmax_record[target_name]
        return target_value * (max_value - min_value) + min_value
    
    def reward(self, s: np.ndarray, a: np.ndarray, s_: np.ndarray, result: Optional[Dict[str, float]]= None, result_: Optional[Dict[str, float]]= None) -> Tuple[float, bool]:
        """
        Calculate the reward based on the performance improvement and threshold achievement.

        Parameters:
            s (np.ndarray): state vector
            a (np.ndarray): action vector
            s_ (np.ndarray): new state vector
        
        Returns:
            float: reward
            bool: done
        """
        self.sr_percentile['all'] += 1
        ################## Illegal action or state vector ##################
        if not self.judge_s(s_):  # Illegal action or state vector
            return -1 + math.log(self.env_step + 1) / (2 * math.log(MaxStep)), True
        
        self.sr_percentile['all_classification'] += 1
        ################## BMG classification ##################
        bmg_prob = self.cls_func(s_)
        if bmg_prob < 0.5:
            return bmg_prob - 0.5, True
        self.sr_percentile['cls'] += 1
        ################## BMG target properties ##################
        reward = bmg_prob - 0.5
        if not result: result = self.target_func(s)
        if not result_: result_ = self.target_func(s_)
        result['Tg/Tl'] = self.inverse_normal_targets('Tg(K)', result['Tg(K)']) / self.inverse_normal_targets('Tl(K)', result['Tl(K)'])
        result_['Tg/Tl'] = self.inverse_normal_targets('Tg(K)', result_['Tg(K)']) / self.inverse_normal_targets('Tl(K)', result_['Tl(K)'])
        # Phase 1 (ensure legal action and state vector)
        bmg_ = BMGs(s_, result_)
        base_matrix_ = bmg_.get_base_matrix()
        
        # Calculate the reward based on performance improvement and threshold achievement
        thresholds = self.matrix_thresholds.get(base_matrix_, self.matrix_thresholds['New'])
        thresholds['Tg/Tl'] = min(0.6, thresholds['Tg/Tl'])
        done_count = 0
        self.sr_percentile['all_regression'] += 1
        for target, weight in RewardWeight.items():
            threshold = thresholds[target]
            current_value = result_[target]
            previous_value = result[target]
            if weight > 0:
                # Calculate improvement ratio
                improvement_ratio = (current_value - previous_value) / max(threshold, previous_value)
                smooth_improvement_ratio = np.tanh(improvement_ratio)  # Smooth the improvement ratio

                # Check if the current value meets or exceeds the threshold
                meets_threshold = current_value >= threshold
                
                if target in DoneTargets and current_value >= threshold * DoneRatio:
                    done_count += 1
                # Compute component of reward
                if meets_threshold:
                    reward = reward + 0.2 * weight # Reward for meeting the threshold

                reward += (weight * smooth_improvement_ratio * 0.1)
            if current_value >= threshold:
                if target not in self.sr_percentile:
                    self.sr_percentile[target] = 0
                self.sr_percentile[target] += 1
        if done_count == len(DoneTargets):
            done = True
        else:
            done = False
        if done:
            if self.exist_bmgs.get(bmg_.bmg_s, 0) == 0:
                self.exist_bmgs[bmg_.bmg_s] = 1
                reward += 0.2
                result_['Chemical composition'] = bmg_.bmg_s
                # Invserse the normalization of the target values
                for target in result_:
                    if target in self.minmax_record:
                        result_[target] = self.inverse_normal_targets(target, result_[target])
                self.new_bmgs.append(result_)
                # logger.info('Find new BMGs: %s, reward: %s', bmg_.bmg_s, reward)
            else:
                self.exist_bmgs[bmg_.bmg_s] += 1
                ubc_reward = Alpha * math.sqrt((2 * math.log(MaxStep))/self.exist_bmgs[bmg_.bmg_s])
                reward += ubc_reward / 10
                # logger.info('Find existing BMGs: %s, reward: %s', bmg_.bmg_s, ubc_reward)
                done = False
        return reward, done

    def save_bmgs(self, path):
        """
        Save the new BMGs to an Excel file.
        """
        if self.new_bmgs:
            df = pd.DataFrame(self.new_bmgs)
            df.to_excel(path, index=False)
    
    def log_sr_percentile(self):
        logger.info(f"SR for leggal action and state vector: {self.sr_percentile['all_classification']}/{self.sr_percentile['all']}({round(100 * self.sr_percentile['all_classification'] / self.sr_percentile['all'], 1)}%)")
        for target, count in self.sr_percentile.items():
            if target.startswith('all'):
                continue
            if target == 'cls':
                logger.info(f"SR for {target}: {count}/{self.sr_percentile['all_classification']}({round(100 * count / self.sr_percentile['all_classification'], 1)}%)")
            else:
                logger.info(f"SR_{Percentile * 100}% for {target}: {count}/{self.sr_percentile['all_regression']}({round(100 * count / self.sr_percentile['all_regression'], 1)}%)")
        return self.sr_percentile['all']


def unit_test():
    env = Environment()
    for s in env.init_pool:
        assert env.judge_s(s)
    print('len(env.init_pool):', len(env.init_pool))
    print(env.matrix_thresholds)
    for epoch in range(100):
        random_base_matrix = np.random.choice(list(env.init_base_matrix.keys()))
        s = env.reset_by_constraint(*env.init_base_matrix[random_base_matrix])
        for step in range(MaxStep):
            indexs = np.where(s != 0)[0]
            a = np.random.rand(len(indexs)) * A_Scale
            a = a - a.mean()
            s_, r, done = env.step(s, a)
            if done:
                print("step:", step, "r:", r, "done:", done)
                break
            s = s_


# python -m env.environment
if __name__ == '__main__':
    unit_test()
