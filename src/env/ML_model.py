import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import logging
from dataloader.my_dataloader import CustomDataLoader
from MLs.edRVFL import EnsembleDeepRVFL

logger = logging.getLogger(__name__)

class ML_Model:
    def __init__(self, data_path, drop_columns, target_columns, results_path):
        self.data = CustomDataLoader(data_path, drop_columns, target_columns)
        self.target_columns = target_columns
        self.results_path = results_path
        self.minmax_record = {}
        for target_clomuns in target_columns:
            self.minmax_record[target_clomuns] = self.data.data[target_clomuns].min(), self.data.data[target_clomuns].max()

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
    
    
    def get_best_models(self, norm_features=True, norm_target=True):
        xlsx_paths = glob.glob(self.results_path + '/*.xlsx')
        best_models = {}
        for target_column in self.target_columns:
            # Find the best model for each target column
            for xlsx_path in xlsx_paths:
                if target_column in xlsx_path:
                    best_models[target_column] = self._get_best_model(xlsx_path)
                    x, y = self.data.get_features_for_target(target_column)
                    x = x.to_numpy()
                    if norm_features:
                        x_sums = x.sum(axis=1, keepdims=True)
                        x = x / x_sums
                    y = y.to_numpy()
                    if norm_target:
                        y = (y - y.min()) / (y.max() - y.min())
                    best_models[target_column].fit(x, y)
                    logger.info(f"Best model for {target_column} is {best_models[target_column].__class__.__name__}")
        return best_models

    def _get_best_model(self, model_path):
        df = pd.read_excel(model_path, index_col=0)
        df = df.T
        # Find the best model based on the highest R2 score
        model_name = df['r2'].astype(float).idxmax()
        df_dict =  df.to_dict()
        best_param = df_dict['best_params'][model_name]

        if model_name == 'Ridge':
            model = Ridge(**eval(best_param))
        elif model_name == 'Lasso':
            model = Lasso(**eval(best_param))
        elif model_name == 'ElasticNet':
            model = ElasticNet(**eval(best_param))
        elif model_name == 'SVR':
            model = SVR(**eval(best_param))
        elif model_name == 'RandomForestRegressor':
            model = RandomForestRegressor(**eval(best_param))
        elif model_name == 'GradientBoostingRegressor':
            model = GradientBoostingRegressor(**eval(best_param))
        elif model_name == 'AdaBoostRegressor':
            model = AdaBoostRegressor(**eval(best_param))
        elif model_name == 'KNeighborsRegressor':
            model = KNeighborsRegressor(**eval(best_param))
        elif model_name == 'XGBRegressor':
            model = XGBRegressor(**eval(best_param))
        elif model_name == 'edRVFL':
            model = EnsembleDeepRVFL(**eval(best_param))
        else:
            raise ValueError("Unknown model name.")
        
        return model
    
    def get_pred_result(self, norm_features=True, norm_target=True, save_path="../results/pred_results"):
        xlsx_paths = glob.glob(self.results_path + '/*.xlsx')
        best_models = {}
        for target_column in self.target_columns:
            # Find the best model for each target column
            for xlsx_path in xlsx_paths:
                if target_column in xlsx_path:
                    best_models[target_column] = self._get_best_model(xlsx_path)
                    x, y = self.data.get_features_for_target(target_column)
                    x = x.to_numpy()
                    if norm_features:
                        x_sums = x.sum(axis=1, keepdims=True)
                        x = x / x_sums
                    y = y.to_numpy()
                    if norm_target:
                        y = (y - y.min()) / (y.max() - y.min())
                    y_pred, y_test = self.eval_model(best_models[target_column], x, y, target_column)
                    save_df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
                    save_df.to_excel(f"{save_path}/{target_column}.xlsx", index=False)
                    logger.info(f"Prediction results for {target_column} saved to {save_path}/{target_column}.xlsx")     
                            
    def eval_model(self, model, features, target, target_name):
        r2_scores = []
        rmse_scores = []
        mape_scores = []
        y_preds = []
        y_tests = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        print(f"Target {target_name} Model: {model.__class__.__name__} len(features): {len(features)}")
        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = target[train_index], target[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))
            for i, y in enumerate(y_test):
                y_test[i] = self.inverse_normal_targets(target_name, y)
            y_tests.extend(y_test)
            for i, y in enumerate(y_pred):
                y_pred[i] = self.inverse_normal_targets(target_name, y)
            y_preds.extend(y_pred)
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            # Handling zero values in target for MAPE calculation
            y_test_without_zeros = np.where(y_test != 0, y_test, 1e-8)
            y_pred_without_zeros = np.where(y_test != 0, y_pred, 1e-8)
            mape = np.mean(np.abs((y_test_without_zeros - y_pred_without_zeros) / y_test_without_zeros)) * 100
            mape_scores.append(mape)
        mean_r2_score = np.mean(r2_scores)
        mean_rmse_score = np.mean(rmse_scores)
        mean_mape_score = np.mean(mape_scores)
        logger.info(f"Target {target_name} Model: {model.__class__.__name__} R2 Score: {mean_r2_score}, RMSE: {mean_rmse_score}, MAPE: {mean_mape_score}")
        return y_preds, y_tests

def unit_test():
    data_path = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
    drop_columns = ['BMGs', "Chemical composition"]
    target_columns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
    results_path = '../results/ML_All'

    ml_model = ML_Model(data_path, drop_columns, target_columns, results_path)
    ml_model.get_pred_result()

# python -m env.ML_model
if __name__ == '__main__':
    unit_test()