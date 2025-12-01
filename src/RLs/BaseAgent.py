from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from config import *
from env.environment import Environment
from exp.PERBuffer import PrioritizedReplayBuffer
from exp.ReplayBuffer import ReplayBuffer
from exp.TrustBuffer import TrustReplayBuffer
from RLs.NetWork import device
from RLs.utils import moving_average

class BaseAgent(ABC):
    def __init__(self, use_per: bool = False, use_trust: bool = False) -> None:
        super(BaseAgent, self).__init__()
        self.name = 'base_agent'
        self.use_per = use_per
        self.use_trust = use_trust
        if self.use_per:
            self.buffer = PrioritizedReplayBuffer(PoolSize)
        else:
            self.buffer = ReplayBuffer(PoolSize)
        self.env = Environment()
        if self.use_trust:
            self.trust_pool = TrustReplayBuffer(TrustPoolPath)
        self.device = device
        
        
    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        pass
    
    @abstractmethod
    def train_step(self, batch_size: int) -> float:
        pass
    
    def store_experience(self, s, action, reward, s_, done) -> None:
        """Store experience in the replay buffer."""
        self.buffer.add(s, action, reward, s_, done)
    
    def sample_experiences(self, batch_size: int):
        """Sample a batch of experiences from the replay buffer."""
        return self.buffer.sample(batch_size)
    
    def sample(self, batch_size: int):
        # Sample initial experiences
        if not self.use_per:
            states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)
            weights, batch_idxes = torch.FloatTensor(np.ones_like(rewards)).to(self.device), None
        else:
            states, actions, rewards, next_states, dones, weights, batch_idxes = self.sample_experiences(batch_size)
            weights = np.sqrt(weights)
            weights = torch.FloatTensor(weights).to(self.device)
        replace_indices = None
        if self.use_trust:
            # Calculate average reward of the sampled experiences
            avg_reward = np.mean(rewards)

            # Determine the proportion of replacements based on avg_reward
            replacement_ratio = max(min(0.05, (self.trust_pool.ave_reward - avg_reward) / 2), 0)  # Adjust the scaling factor as needed
            num_replacements = int(batch_size * replacement_ratio)

            # Sample replacements from the trust pool
            if num_replacements > 0:
                trust_states, trust_actions, trust_rewards, trust_next_states, trust_dones = self.trust_pool.sample(avg_reward + 1, num_replacements)
                replace_indices = np.random.choice(batch_size, num_replacements, replace=False)

                states[replace_indices] = trust_states
                actions[replace_indices] = trust_actions
                rewards[replace_indices] = trust_rewards
                next_states[replace_indices] = trust_next_states
                dones[replace_indices] = trust_dones

                if self.use_per:
                    # Update weights for the replaced experiences
                    weights[replace_indices] = torch.ones_like(weights[replace_indices]).to(self.device)
                    # Delete the replaced experiences idxes from the PER buffer
                    batch_idxes = np.delete(batch_idxes, replace_indices)

        # Normalizing the states and actions
        states = states / np.sum(states, axis=1, keepdims=True)
        next_states = next_states / np.sum(next_states, axis=1, keepdims=True)
        actions = actions / (A_Scale)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        return states, actions, rewards, next_states, dones, weights, batch_idxes, replace_indices
    
    
    def update_priorities(self, idxes, priorities):
        """Update priorities of experiences in the buffer."""
        if self.use_per:
            self.buffer.update_priorities(idxes, priorities)
    
    def save_model(self, model: nn.Module, path: str) -> None:
        """Save the model parameters."""
        torch.save(model.state_dict(), path)
    
    def load_model(self, model: nn.Module, path: str) -> None:
        """Load the model parameters."""
        model.load_state_dict(torch.load(path))
    
    def evaluate_policy(self, episodes: int = 5) -> float:
        """Evaluate the current policy."""
        total_reward = 0.0
        total_conv_reward = 0.0
        for _ in range(episodes):
            rewards = []
            state = self.env.reset()
            step = 0
            done = False
            while not done and step < MaxStep:
                action = self.select_action(state, explore=False)
                next_state, reward, done = self.env.step(state, action, step)
                state = next_state
                rewards.append(reward)
                step += 1
            total_reward += np.mean(rewards)
            total_conv_reward += moving_average(rewards, min(len(rewards), 10))[-1]

        return total_reward / episodes, total_conv_reward / episodes
        

    
def unit_test():
    agent = BaseAgent()

# python -m RLs.BaseAgent
if __name__ == '__main__':
    unit_test()