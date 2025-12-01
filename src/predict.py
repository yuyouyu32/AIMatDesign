import os

import click

from config import N_Action, N_State, logging
from RLs.DDPG import DDPGAgent
from RLs.DQN import DQNAgent
from RLs.PPO import PPOAgent
from RLs.Predictor import Predictor
from RLs.SAC import SACAgent
from RLs.TD3 import TD3Agent
from RLs.Random import RandomAgent
from RLs.BaseAgent import BaseAgent

logger = logging.getLogger(__name__)

@click.command()
@click.option('--model', default='td3', help='RL model agent.')
@click.option('--c_pth', default='../ckpts/td3/', help='Critic model to predict.')
@click.option('--a_pth', default='../ckpts/td3/', help='Actor model to predict.')
@click.option('--episodes', default=1500, help='Number of episodes to predict.')
@click.option('--save_path', default='../designs/td3', help='Path to save the designed results.')
@click.option('--log_episodes', default=10, help='Log every n episodes.')
@click.option('--explore_base_index', default=None, help='Index of the base to explore.')

def predict(model, c_pth, a_pth, episodes, save_path, log_episodes, explore_base_index):
    """Train a RLs agent with given parameters."""
    if model == 'td3':
        agent = TD3Agent(N_State, N_Action)
    elif model == 'ppo':
        agent = PPOAgent(N_State, N_Action)
    elif model == 'dqn':
        agent = DQNAgent(N_State, N_Action)
    elif model == 'ddpg':
        agent = DDPGAgent(N_State, N_Action)
    elif model == 'sac':
        agent = SACAgent(N_State, N_Action)
    elif model == "random":
        agent = RandomAgent(N_State, N_Action)
    else:
        raise ValueError(f"Model {model} not supported.")
    predictor = Predictor(agent=agent, episodes=episodes, save_path=save_path, log_episodes=log_episodes)
    if model != 'random':
        predictor.load(c_pth, a_pth)
        predictor.agent.epsilon = predictor.agent.epsilon_min
    done_num = 0
    all_eposides = 0
    if explore_base_index is not None:
        if explore_base_index.isdigit():
            explore_base_index = int(explore_base_index)
            logger.info(f"Start predicting {model} agent with {episodes} episodes, env reset by explore base index: {explore_base_index}.")
            predictor.predict(explore_base_index=explore_base_index)
            done_num += len(predictor.env.new_bmgs)
            all_eposides += episodes
        elif explore_base_index.lower() == 'all':
            logger.info(f"Start predicting {model} agent with {episodes} episodes {len(predictor.env.init_base_matrix.keys())} bases, env reset by all explore bases.")
            for base_matrix in list(predictor.env.init_base_matrix.keys()):
                logger.info(f"Start predicting {model} agent with {episodes} episodes, env reset by explore base: {base_matrix}.")
                predictor.save_path = os.path.join(save_path, base_matrix)
                if not os.path.exists(predictor.save_path):
                    os.makedirs(predictor.save_path)
                predictor.predict(explore_base_index=base_matrix)
                done_ratio = len(predictor.env.new_bmgs) / episodes
                logger.info(f"Base Matrix: {base_matrix}, Done Ratio: {done_ratio}")
                done_num += len(predictor.env.new_bmgs)
                all_eposides += episodes
                predictor.env.new_bmgs = []
        else:
            logger.info(f"Start predicting {model} agent with {episodes} episodes, env reset by explore base index: {explore_base_index}.")
            predictor.predict(explore_base_index=explore_base_index)
            done_num += len(predictor.env.new_bmgs)
            all_eposides += episodes

    else:
        explore_base_index = None
        logger.info(f"Start predicting {model} agent with {episodes} episodes and random env reset method.")
        predictor.predict(explore_base_index=explore_base_index)
        done_num += len(predictor.env.new_bmgs)
        all_eposides += episodes
    logger.info(f"Predicting {model} agent done.")
    logger.info(f"Predicting {model} agent log sr percentile.")
    all_steps = predictor.env.log_sr_percentile()
    logger.info(f"Predicting {model} agent done eposode ratio: {round(100 * done_num / all_eposides, 2)}%")
    logger.info(f"Predicting {model} agent done step ratio: {round(100 * done_num / all_steps, 2)}%")

if __name__ == '__main__':
    predict()
