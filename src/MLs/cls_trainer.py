import pandas as pd
from dataloader.my_dataloader import CustomDataLoader
from .cls_worker import ModelEvaluatorKFold
import os
from config import logging
logger = logging.getLogger(__name__)


output_path = '../results/'  # Replace with your output path

file_path = '/data/home/yeyongyu/SHU/ReinforceMatDesign/data/ALL_data_cls.xlsx'  # Replace with your file path
drop_columns = ['Class', 'GFA', "Chemical composition"]
target_columns = ['cls_label']
Save_path = output_path + 'Cls/'

def process_target(target_name, file_path, drop_columns, Save_path, target_columns):
    dataloader = CustomDataLoader(file_path, drop_columns, target_columns)
    features, target = dataloader.get_features_for_target(target_name)
    logger.info("{:=^80}".format(f" {target_name} Start"))
    evaluator = ModelEvaluatorKFold(n_splits=5)
    evaluation_results = evaluator.evaluate_models(features, target, norm_features=True)
    results = pd.DataFrame(evaluation_results)
    logger.info(f"{target_name}:\n")
    print(results)
    target_name = target_name.replace('/', '_')
    try:
        results.to_excel(Save_path + f"{target_name}_ml.xlsx")
    except FileNotFoundError:
        import os
        os.makedirs(Save_path, exist_ok=True)
        results.to_excel(Save_path + f"{target_name}_ml.xlsx")
    logger.info("{:=^80}".format(f" {target_name} Done"))

# nohup python -u -m MLs.cls_trainer > ../logs/cls_trainer.log 2>&1 & 
if __name__ == "__main__":
    if not os.path.exists(Save_path):
        os.makedirs(Save_path)

    for target_name in target_columns:
        process_target(target_name, file_path, drop_columns, Save_path, target_columns)
