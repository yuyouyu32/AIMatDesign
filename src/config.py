import math
# Data config
DataPath = '../data/ALL_data_grouped_processed.xlsx'  # Replace with your file path
BaseMatrixPath = '../data/base_matrix.json'
DropColumns = ['BMGs', "Chemical composition", "cls_label"]
TargetColumns = ['Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)','yield(MPa)', 'Modulus (GPa)', 'Ε(%)']
CompositionColumns = ['Ni', 'Cr', 'Nb', 'P', 'B', 'Si', 'Fe', 'C', 'Mo', 'Y', 'Co', 'Au', 'Ge', 'Pd', 'Cu', 'Zr', 'Ti', 'Al', 'Mg', 'Ag', 'Gd', 'La', 'Ga', 'Hf', 'Sn', 'In', 'Ca', 'Zn', 'Nd', 'Er', 'Dy', 'Pr', 'Ho', 'Ce', 'Sc', 'Ta', 'Mn', 'Tm', 'Pt', 'V', 'W', 'Tb', 'Li', 'Sm', 'Lu', 'Yb', 'Pb', 'Sr', 'Ru', 'Be', 'Rh', 'S']
# Regression config
MLResultPath = '../results/ML_All'

# Classificaiton config
ClsPath = '../data/ALL_data_cls.xlsx'  # Replace with your file path
ClsDropColumns = ['Class', 'GFA', "Chemical composition"]
ClsTargetColumns = ['cls_label']
ClsResultpath = '../results/Cls'

# Seed config
Seed = 21

# ENV config
N_State = len(CompositionColumns)
N_Action = 7
A_Scale = 5 / 100
Percentile = 0.8
DoneRatio = 1.2
MaxStep = 128
Alpha = 2 / (math.sqrt((2 * math.log(MaxStep))/1)) # UBC 1 config
OptionalResetElement = {'Ag', 'Ti', 'La', 'Ce', 'Gd', 'Y', 'Nb', 'Sc'}


# Exp Config
TrustPoolPath = '../exp_pool/trust_pool.jsonl'
PoolSize = 250000

# Reward cnofig
RewardWeight = {
    'Dmax(mm)': 0.2,
    'Tg/Tl': 0.2,
    'Tg(K)': 0,
    'Tx(K)': 0,
    'Tl(K)': 0,
    'yield(MPa)': 0.0,
    'Modulus (GPa)': 0.3,
    'Ε(%)': 0.3
}
DoneTargets = {'Modulus (GPa)', 'Ε(%)'}

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

ExploreBases = [(
    {
        'Zr': (0.4, 0.7),
        'Cu': (0.1, 0.25),
        'Ni': (0.05, 0.15),
        'Al': (0.05, 0.15)
    },
    {
        'Ag': (0, 0.1),
        'Ti': (0, 0.1),
        'La': (0, 0.1),
        'Ce': (0, 0.1),
        'Gd': (0, 0.1),
        'Y': (0, 0.1)
    },
    5)]