from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from .edRVFL import EnsembleDeepRVFL
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern
import random

param_grid = {
    'Ridge': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [1000, 10000, 50000]},
    'Lasso': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [1000, 10000, 50000]},
    'ElasticNet': {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0.25, 0.5, 0.75], 'max_iter': [1000, 10000, 50000]},
    'SVR': {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]},
    'RandomForestRegressor': {'n_estimators': [10, 50, 100, 200, 500], 'max_depth': [None, 5, 10, 20, 50]},
    'GradientBoostingRegressor': {'n_estimators': [10, 50, 100, 200], 'max_depth': [1, 3, 5, 10, 50], 'learning_rate': [0.001, 0.01, 0.1, 1]},
    'AdaBoostRegressor': {'n_estimators': [10, 50, 100, 200], 'learning_rate': [0.001, 0.01, 0.1, 1]},
    'KNeighborsRegressor': {'n_neighbors': [3, 5, 7, 10, 20, 50]},
    'XGBRegressor': {'n_estimators': [10, 50, 100, 200, 500], 'learning_rate': [0.001, 0.01, 0.1, 1], 'max_depth': [3, 5, 10, 20, 50]},
    # 'GaussianProcessRegressor': {'kernel': [1.0 * RBF(1.0), 1.0 * RationalQuadratic(), 1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)), ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2), 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=1.5)]},
    'edRVFL': {
    'n_nodes': [128, 256], 
    'lam': [1, 2],
    'w_random_vec_range': [[-10, 10]],
    'b_random_vec_range': [[0, 10]],
    'n_layer': [16, 32, 64],
    'random_seed': [random.randint(0, 1000) for _ in range(32)],
    'same_feature': [True],
    'activation': ['leaky_relu']
    }
}


models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'ElasticNet': ElasticNet(),
    'SVR': SVR(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'XGBRegressor': XGBRegressor(),
    # 'GaussianProcessRegressor': GaussianProcessRegressor(), 
    'edRVFL': EnsembleDeepRVFL()
}