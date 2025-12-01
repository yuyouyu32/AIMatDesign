import json
from env.environment import *
import numpy as np

data_path = "/data/home/yeyongyu/SHU/mattergen/results/mattergen_results.jsonl"

env = Environment()

with open(data_path, "r") as f:
    lines = f.readlines()
for line in lines:
    s = json.loads(line)
    s = env.set_s_by_dict(s)
    # 全0的a
    a = np.zeros(7)

    env.step(s, a)

env.log_sr_percentile()
print(len(env.new_bmgs) / len(lines))
print(env.new_bmgs)