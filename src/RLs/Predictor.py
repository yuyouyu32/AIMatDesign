import os
import random
from collections import deque
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ExploreBases, MaxStep, N_Action, N_State, Seed, logging
from env.environment import Environment
from RLs.BaseAgent import BaseAgent
from RLs.PPO import PPOAgent
from RLs.TD3 import TD3Agent
from RLs.DDPG import DDPGAgent
from RLs.SAC import SACAgent
from RLs.DQN import DQNAgent
from RLs.Random import RandomAgent
from RLs.utils import moving_average

logger = logging.getLogger(__name__)
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

class Predictor:
    def __init__(self, agent: Union[BaseAgent, PPOAgent, TD3Agent, DDPGAgent, SACAgent, DQNAgent], episodes: int, save_path: str, log_episodes: int = 10):
        self.agent = agent
        self.env: Environment = self.agent.env
        self.episodes = episodes
        self.save_path = save_path
        self.log_episodes = log_episodes
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.log_dir = os.path.join(self.save_path, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.window_size = 10
        self.reward_record = []

    def reset_state(self, explore_base_index):
        if explore_base_index is None:
            random_base_element = np.random.choice(list(self.env.init_base_matrix.keys()))
            state = self.env.reset_by_constraint(*self.env.init_base_matrix[random_base_element])
        elif type(explore_base_index) == int:
            mandatory_elements, optional_elements, k = ExploreBases[explore_base_index]
            state = self.env.reset_by_constraint(mandatory_elements, optional_elements, k, replace_flag=False, min_optional_len=1)
        elif type(explore_base_index) == str and explore_base_index in self.env.init_base_matrix:
            if explore_base_index not in self.env.init_base_matrix:
                raise ValueError(f"Invalid explore base index: {explore_base_index}")
            state = self.env.reset_by_constraint(*self.env.init_base_matrix[explore_base_index])
        else:
            raise ValueError(f"Invalid explore base index: {explore_base_index}")
        return state


    def draw_reward_q_curve(self, reward_record, q_values, episode):
        # 给我绘制q值和reward的曲线,要求在同一张图上
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots()

        # Plot reward curve
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.plot(range(len(reward_record)), reward_record, label="Reward")
        ax1.tick_params(axis='y')

        # Plot q-value curve
        ax2 = ax1.twinx()
        ax2.set_ylabel("Q-Value")
        ax2.plot(range(len(q_values)), q_values, label="Q-Value", color='r')
        ax2.tick_params(axis='y')

        # Add legend and title
        fig.tight_layout()
        fig.legend(loc="upper right")
        plt.title("Reward and Q-Value Curve")

        # Show the plot
        plt.savefig(os.path.join(self.save_path, f"reward_q_curve_{episode}.png"))
        
        
        
    def predict(self, explore_base_index: Optional[int] = None):
        total_steps = 0
        all_rewards = []
        all_q_values = []
        for episode in tqdm(range(self.episodes)):
            state = self.reset_state(explore_base_index)
            done = False
            episode_step = 0
            episode_reward = 0
            q_values = []
            reward_record = []
            while not done and episode_step < MaxStep:
                action = self.agent.select_action(state, explore=False)
                next_state, reward, done = self.env.step(state, action)
                if self.agent.name == "TD3":
                    q_value = self.agent.get_q_values(state, action)
                else:
                    q_value = 0
                state = next_state
                episode_reward += reward
                total_steps += 1
                episode_step += 1
                reward_record.append(reward)
                q_values.append(q_value)
                if done:
                    if reward < 0:
                        state = self.reset_state(explore_base_index)
                        done = False
                    
            all_q_values.append(q_values)
            all_rewards.append(reward_record)
            # self.draw_reward_q_curve(reward_record, q_values, episode)
            all_rewards.append
            average_reward = episode_reward / episode_step
            self.reward_record.append(average_reward)

            self.writer.add_scalar("Reward", average_reward, episode)
            if episode % self.log_episodes == 0 and episode > 0:
                logger.info(f"Episode: {episode + 1}/{self.episodes}, Ave Reward: {round(average_reward, 2)}, Done nums: {len(self.env.new_bmgs)}")
        self.env.save_bmgs(os.path.join(self.save_path, "new_BMGs.xlsx"))
        self.save_logs(self.reward_record)
        self.writer.close()
        self.save_r_q_curve(all_rewards, all_q_values)

    def load(self, critic_path, actor_path):
        self.agent.load_model(self.agent.actor, actor_path)
        self.agent.load_model(self.agent.critic, critic_path)
        if hasattr(self.agent, "actor_target"):
            self.agent.load_model(self.agent.actor_target, actor_path)
            self.agent.load_model(self.agent.critic_target, critic_path)

    def save_logs(self, reward_record):
        torch.save(reward_record, os.path.join(self.save_path, "reward_record.pth"))

    def save_r_q_curve(self, reward_record, q_values):
        import pandas as pd
        df = pd.DataFrame()
        df["Episode"] = range(len(reward_record))
        df["Reward"] = [str(reward) for reward in reward_record]
        df["Q-Value"] = [str(q) for q in q_values]
        df.to_csv(os.path.join(self.save_path, "reward_q_curve.csv"), index=False)
    

