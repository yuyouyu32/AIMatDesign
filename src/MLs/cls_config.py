from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

param_grid = {
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'max_iter': [100, 200, 500, 1000, 5000],
        'solver': ['lbfgs', 'liblinear', 'sag', 'saga'],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'class_weight': ['balanced']
    },
    'SVC': {
        'C': [0.01, 0.1, 1, 5, 10, 50, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'degree': [2, 3, 4, 5],
        'class_weight': ['balanced']
    },
    'RandomForestClassifier': {
        'n_estimators': [10, 20, 50, 100, 200, 500],
        'max_depth': [None, 3, 5, 10, 20, 30],
        'max_features': ['auto', 'sqrt', 'log2', 0.25, 0.5, 0.75],
        'class_weight': ['balanced']
    },
    'GradientBoostingClassifier': {
        'n_estimators': [10, 20, 50, 100, 200, 500],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1],
        'max_depth': [3, 5, 10, 15, 20]
    },
    'AdaBoostClassifier': {
        'n_estimators': [10, 20, 50, 100, 200, 500],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1]
    },
    'KNeighborsClassifier': {
        'n_neighbors': [3, 5, 7, 10, 20, 30, 50],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    'XGBClassifier': {
        'n_estimators': [10, 20, 50, 100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 10, 15, 20],
        'subsample': [0.5, 0.75, 1],
        'colsample_bytree': [0.5, 0.75, 1]
    },
    'DecisionTreeClassifier': {
        'max_depth': [None, 3, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'class_weight': ['balanced']
    },
    # 'MLPClassifier': {
    #     'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (200, 200)],
    #     'activation': ['relu', 'tanh', 'logistic'],
    #     'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    #     'max_iter': [200, 500, 1000, 2000]
    # },
    'GaussianNB': {
        'var_smoothing': [1e-10, 1e-09, 1e-08, 1e-07, 1e-06]
    },
    'MultinomialNB': {
        'alpha': [0.001, 0.01, 0.1, 1, 10]
    },
    'BernoulliNB': {
        'alpha': [0.001, 0.01, 0.1, 1, 10]
    },
    'LinearDiscriminantAnalysis': {
        'solver': ['svd', 'lsqr', 'eigen'],
        'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
    },
    'QuadraticDiscriminantAnalysis': {
        'reg_param': [0, 0.001, 0.01, 0.1]
    },
    # 'LGBMClassifier': {
    #     'n_estimators': [50, 100, 200, 500, 1000],
    #     'learning_rate': [0.005, 0.01, 0.1, 0.2, 0.3],
    #     'num_leaves': [31, 62, 127, 255],
    #     'max_depth': [3, 5, 7, 10, 20]
    # },
    'CatBoostClassifier': {
        'iterations': [50, 200, 500, 1000],
        'learning_rate': [0.01, 0.1, 0.3],
        'depth': [4, 8, 12]
    }
}

models = {
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(),
    'RandomForestClassifier': RandomForestClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'XGBClassifier': XGBClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    # 'MLPClassifier': MLPClassifier(),
    'GaussianNB': GaussianNB(),
    'MultinomialNB': MultinomialNB(),
    'BernoulliNB': BernoulliNB(),
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    # 'LGBMClassifier': LGBMClassifier(),
    'CatBoostClassifier': CatBoostClassifier(verbose=0)
}