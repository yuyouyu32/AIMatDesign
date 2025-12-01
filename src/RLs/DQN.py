import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from RLs.BaseAgent import BaseAgent
from RLs.NetWork import DeterActorNet, QNetwork, device
from .utils import OrnsteinUhlenbeckNoise

class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, use_per=False, use_trust=False):
        super(DQNAgent, self).__init__(use_per, use_trust)
        self.name = 'dqn'

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.actor = DeterActorNet(state_dim, action_dim).to(device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.9, patience=20)
        self.discount = 0.99
        self.tau = 0.005
        
        # Noise config
        self.noise = OrnsteinUhlenbeckNoise(np.zeros(action_dim))
        self.noise.sigma = 0.2 # Initial noise level
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05



    def select_action(self, state: np.ndarray, explore: bool=True):
        if explore and np.random.rand() < self.epsilon:
            action = self.env.get_random_legal_action(state)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            state = state / sum(state)
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            k = torch.count_nonzero(state, dim=1).item()  # Count non-zero elements in the state tensor
            k_tensor = torch.tensor([k], dtype=torch.int).to(self.device)  # Convert k to a tensor and move to device
            action = self.actor(state, k_tensor).cpu().data.numpy().flatten()
            action *= self.action_scale
        return action

    def train_step(self, batch_size):
        states, actions, rewards, next_states, dones, weights, batch_idxes, replace_indices = self.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            next_q_max, _ = next_q_values.max(dim=1, keepdim=True)
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.discount * next_q_max

        # Get current Q values
        current_q = self.q_network(states).gather(1, actions)

        # Compute loss
        loss = (weights * (current_q - target_q).pow(2)).mean()

        # Compute actor loss
        actor_loss = -self.q_network(states).mean()

        # Optimize the Q network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Optimize the actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update the learning rate
        self.scheduler.step(loss.item())
        self.actor_scheduler.step(actor_loss.item())

        if self.use_per and batch_idxes is not None:
            new_priorities = (torch.abs(current_q - target_q).cpu().data.numpy() + 1e-6).squeeze()
            if replace_indices is not None:
                new_priorities = np.delete(new_priorities, replace_indices)
            self.buffer.update_priorities(batch_idxes, new_priorities)
        
        return loss.item(), actor_loss.item()

def unit_test():
    from config import N_Action, N_State
    agent = DQNAgent(N_State, N_Action)
    s = agent.env.reset()
    print(agent.select_action(s, explore=False))

# python -m RLs.DQN
if __name__ == '__main__':
    unit_test()
