import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import logging
from dataloader.my_dataloader import CustomDataLoader


logger = logging.getLogger(__name__)


def eval_none(best_param):
    best_param = eval(best_param)
    for key, value in best_param.items():
        if value == 'none':
            best_param[key] = None
    return best_param

class Cls_Model:
    def __init__(self, data_path, drop_columns, target_columns, results_path):
        self.data = CustomDataLoader(data_path, drop_columns, target_columns)
        self.target_columns = target_columns
        self.results_path = results_path

    def get_best_models(self, norm_features=True):
        cls_model = self._get_best_model(self.results_path + '/cls_label_ml.xlsx')
        x, y = self.data.get_features_for_target(self.target_columns[0])
        x = x.to_numpy()
        smote = SMOTE(random_state=42)
        x, y = smote.fit_resample(x, y)
        if norm_features:
            x_sums = x.sum(axis=1, keepdims=True)
            x = x / x_sums
        
        cls_model.fit(x, y)
        logger.info(f"Best model for {self.target_columns[0]} is {cls_model.__class__.__name__}")
        return cls_model

    def _get_best_model(self, model_path):
        df = pd.read_excel(model_path, index_col=0)
        df = df.T
        # Find the best model based on the highest R2 score
        model_name = df['auc'].astype(float).idxmax()
        df_dict =  df.to_dict()
        best_param = df_dict['best_params'][model_name]
        if model_name == 'LogisticRegression':
            model = LogisticRegression(**eval_none(best_param))
        elif model_name == 'SVC':
            model = SVC(**eval_none(best_param))
        elif model_name == 'RandomForestClassifier':
            model = RandomForestClassifier(**eval_none(best_param))
        elif model_name == 'GradientBoostingClassifier':
            model = GradientBoostingClassifier(**eval_none(best_param))
        elif model_name == 'AdaBoostClassifier':
            model = AdaBoostClassifier(**eval_none(best_param))
        elif model_name == 'KNeighborsClassifier':
            model = KNeighborsClassifier(**eval_none(best_param))
        elif model_name == 'XGBClassifier':
            model = XGBClassifier(**eval_none(best_param))
        elif model_name == 'DecisionTreeClassifier':
            model = DecisionTreeClassifier(**eval_none(best_param))
        elif model_name == 'GaussianNB':
            model = GaussianNB(**eval_none(best_param))
        elif model_name == 'MultinomialNB':
            model = MultinomialNB(**eval_none(best_param))
        elif model_name == 'BernoulliNB':
            model = BernoulliNB(**eval_none(best_param))
        elif model_name == 'QuadraticDiscriminantAnalysis':
            model = QuadraticDiscriminantAnalysis(**eval_none(best_param))
        elif model_name == 'CatBoostClassifier':
            best_param = eval_none(best_param)
            best_param['verbose'] = False
            model = CatBoostClassifier(**best_param)
        else:
            raise ValueError(f"Model {model_name} is not recognized.")

        return model

    
    def get_all_models(self, model_path):
        df = pd.read_excel(model_path, index_col=0).T
        df_dict = df.to_dict()
        all_models = {}
        for model_name in df_dict['auc'].keys():
            best_param = df_dict['best_params'][model_name]

            if model_name == 'LogisticRegression':
                all_models[model_name] = LogisticRegression(**eval_none(best_param))
            elif model_name == 'SVC':
                best_param = eval_none(best_param)
                best_param['probability'] = True
                all_models[model_name] = SVC(**best_param)     
            elif model_name == 'RandomForestClassifier':
                all_models[model_name] = RandomForestClassifier(**eval_none(best_param))
            elif model_name == 'GradientBoostingClassifier':
                all_models[model_name] = GradientBoostingClassifier(**eval_none(best_param))
            elif model_name == 'AdaBoostClassifier':
                all_models[model_name] = AdaBoostClassifier(**eval_none(best_param))
            elif model_name == 'KNeighborsClassifier':
                all_models[model_name] = KNeighborsClassifier(**eval_none(best_param))
            elif model_name == 'XGBClassifier':
                all_models[model_name] = XGBClassifier(**eval_none(best_param))
            elif model_name == 'DecisionTreeClassifier':
                all_models[model_name] = DecisionTreeClassifier(**eval_none(best_param))
            elif model_name == 'GaussianNB':
                all_models[model_name] = GaussianNB(**eval_none(best_param))
            elif model_name == 'MultinomialNB':
                all_models[model_name] = MultinomialNB(**eval_none(best_param))
            elif model_name == 'BernoulliNB':
                all_models[model_name] = BernoulliNB(**eval_none(best_param))
            elif model_name == 'LinearDiscriminantAnalysis':
                all_models[model_name] = LinearDiscriminantAnalysis(**eval_none(best_param))
            elif model_name == 'QuadraticDiscriminantAnalysis':
                all_models[model_name] = QuadraticDiscriminantAnalysis(**eval_none(best_param))
            elif model_name == 'CatBoostClassifier':
                best_param = eval_none(best_param)
                best_param['verbose'] = False
                all_models[model_name] = CatBoostClassifier(**best_param)
            else:
                raise ValueError(f"Model {model_name} is not recognized.")
        return all_models
            
    
    def eval_model(self, model, features, target):
        y_tests = []
        y_pred_probas = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        smote = SMOTE(random_state=42)
        
        for train_index, test_index in kf.split(features, target):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = target[train_index], target[test_index]
            X_train, y_train = smote.fit_resample(X_train, y_train)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_tests.extend(y_test)
            y_pred_probas.extend(y_pred_proba)
        return y_tests, y_pred_probas

    def get_cls_results(self, norm_features=True, save_path="../results/cls_pred_results"):
        x, y = self.data.get_features_for_target(self.target_columns[0])
        x = x.to_numpy()
        if norm_features:
            x_sums = x.sum(axis=1, keepdims=True)
            x = x / x_sums
        
        all_models = self.get_all_models(self.results_path + '/cls_label_ml.xlsx')
        save_df = pd.DataFrame()
        for model_name, model in all_models.items():
            logger.info(f"Evaluating {model_name} with K-fold cross-validation...")
            y_tests, y_pred_probas = self.eval_model(model, x, y)
            save_df[model_name] = y_pred_probas
            if 'y_test' not in save_df.columns:
                save_df['y_test'] = y_tests
            else:
                for i, y_label in enumerate(y_tests):
                    assert y_label == save_df['y_test'][i]
            logger.info(f"Model {model_name} AUC: {roc_auc_score(y_tests, y_pred_probas)}")
        save_df.to_excel(f"{save_path}/cls_y_pred_proba.xlsx", index=False)
            
            
def unit_test():
    data_path = '../data/ALL_data_cls.xlsx'  # Replace with your file path
    drop_columns = ['Class', 'GFA', "Chemical composition"]
    target_columns = ['cls_label']
    results_path = '../results/Cls'

    ml_model = Cls_Model(data_path, drop_columns, target_columns, results_path)
    ml_model.get_cls_results()

# python -m env.Cls_model
if __name__ == '__main__':
    unit_test()