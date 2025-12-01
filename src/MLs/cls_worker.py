import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from .cls_config import *
from config import logging

logger = logging.getLogger(__name__)


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affects subprocesses


class ModelEvaluatorKFold:
    def __init__(self, n_splits=5):
        self.models = models
        self.params = param_grid
        self.best_models = {}
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def evaluate_models(self, features, target, norm_features=False):
        results = {}
        features = features.to_numpy()
        target = target.to_numpy()
        if norm_features:
            # 横向归一化到0-1
            row_sums = features.sum(axis=1, keepdims=True)
            features = features / row_sums
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name} with K-fold cross-validation...")
            param_grid = self.params[model_name]

            grid_search = GridSearchCV(model, param_grid, cv=self.kf, scoring='roc_auc', verbose=0, n_jobs=12)
            logger.info("{:=^40}".format(f" {model_name} Grid Search"))
            grid_search.fit(features, target)

            best_model = grid_search.best_estimator_
            self.best_models[model_name] = best_model

            best_params = grid_search.best_params_
            model.set_params(**best_params)
            auc, precision, recall, f1 = self.eval_model(model, features, target)
            results[model_name] = {'best_params': best_params, 'auc': auc, 'precision': precision, 'recall': recall, 'f1_score': f1}
            logger.info(f"{model_name} AUC: {auc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")
            logger.info(f"{model_name} Best Parameters: {best_params}\n", results[model_name])
            logger.info("{:=^40}".format(f" Best {model_name} Parameters Done"))

        return results

    def eval_model(self, model, features, target):
        auc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        smote = SMOTE(random_state=42)
        for train_index, test_index in self.kf.split(features, target):
            X_train, X_test = features[train_index], features[test_index]
            # Over-sampling
            y_train, y_test = target[train_index], target[test_index]
            X_train, y_train = smote.fit_resample(X_train, y_train)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            auc_scores.append(roc_auc_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))
        mean_auc_score = np.mean(auc_scores)
        mean_precision_score = np.mean(precision_scores)
        mean_recall_score = np.mean(recall_scores)
        mean_f1_score = np.mean(f1_scores)
        return mean_auc_score, mean_precision_score, mean_recall_score, mean_f1_score

def unit_test():
    features = pd.DataFrame(np.random.randn(1000, 5) + 1, columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    target = pd.Series(np.random.choice([0, 1], size=(1000,), p=[0.2, 0.8]))
    evaluator = ModelEvaluatorKFold(n_splits=5)
    evaluation_results = evaluator.evaluate_models(features, target, norm_features=True)
    for model_name, result in evaluation_results.items():
        logger.info(f"Model: {model_name}")
        logger.info(f"Best Parameters: {result['best_params']}")
        logger.info(f"Best ROC AUC Score: {result['auc']}")
        logger.info(f"Precision: {result['precision']}")
        logger.info(f"Recall: {result['recall']}")
        logger.info(f"F1 Score: {result['f1_score']}\n")

if __name__ == "__main__":
    unit_test()
