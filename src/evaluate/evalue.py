import ast
import json
import os
import random
import re
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from config import logging

from evaluate.prompts import *

logger = logging.getLogger(__name__)

MaxRetries = 10
Delay = 15

def retry_on_failure(max_retries: int = 3, delay: int = 1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    logger.error(f"Attempt {retries}/{max_retries} failed: {e}")
                    time.sleep(delay)
            raise Exception(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator


def calculate_token_price(prompt_tokens, completion_tokens):
    return 7.13 * (0.01 * prompt_tokens / 1000 + 0.03 * completion_tokens / 1000)

@retry_on_failure(max_retries= MaxRetries, delay=Delay)
def get_rsp_from_GPT(sys_prompt: str, user_prompt: str, n: int = 1):
    """
    Description: 
        Get copywriter from GPT-4
    Args:
        sys_prompt (str): System prompt
        user_prompt (str): User prompt
        n (int, optional): Number of rsp to generate. Defaults to 1.
    Returns:
        rsp (list): List of rsp
        cost (float): Cost price(￥) of generating rsp
    """
    client = OpenAI(base_url=os.environ.get('OPENAI_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY'))
    messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_prompt}]
    chat_completion = client.chat.completions.create(messages=messages, model='gpt-4-turbo', n=n)
    rsps = []
    for i in range(n):
        response = chat_completion.choices[i].message.content
        rsps.append(response)
    cost = calculate_token_price(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)
    return rsps, cost

def parse_result(rsp):
    try:
        score_match = re.search(r'"score": (\d+\.\d+)', rsp)
        score_value = float(score_match.group(1)) if score_match else None
        return score_value
    except:
        return None

def main():
    all_cost = 0
    data = pd.read_excel('../results/random_search_result_filtered_similar.xlsx')
    data = data[['BMGs', 'Tg(K)', 'Tx(K)', 'Tl(K)', 'Dmax(mm)', 'yield(MPa)', 'Modulus (GPa)', 'Ε(%)',  'similar_index']]
    data = data.sort_values(by='Ε(%)', ascending=False)
    data.reset_index(drop=True, inplace=True)
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        similarities = row['similar_index']
        data_prompt = str(row.drop('similar_index').to_dict())
        task_prompt = EvalueUser.format(rule=Rules, data=data_prompt, sim=similarities)
        rsps, cost = get_rsp_from_GPT(sys_prompt=EvalueSystem, user_prompt=task_prompt, n=3)
        all_cost += cost
        scores = []
        for i, rsp in enumerate(rsps):
            data.loc[index, f'rsp_{i}'] = rsp
            score = parse_result(rsp)
            if  score is not None:
                scores.append(score)
        if len(scores) > 0:
            data.loc[index, 'score'] = round(sum(scores) / len(scores), 2)
        data.to_excel('../results/random_search_result_filtered_similar_score.xlsx', index=False)
        time.sleep(random.randint(2, 3))
    logger.info('Cost: ', all_cost)
    
if __name__ == '__main__':
    main()