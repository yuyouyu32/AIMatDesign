import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from RLs.BaseAgent import BaseAgent
from RLs.NetWork import StochasticActorNet, QNetwork, device
import torch.nn.functional as F
import copy

class SACAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, action_space, use_per=False, use_trust=False):
        super(SACAgent, self).__init__(use_per, use_trust)
        self.name = 'sac'

        # Actor Network (Stochastic Policy)
        self.actor = StochasticActorNet(state_dim, action_dim, action_space).to(device)

        # Critic Networks (Twin Q-networks)
        self.critic1 = QNetwork(state_dim + action_dim, 1).to(device)
        self.critic2 = QNetwork(state_dim + action_dim, 1).to(device)

        # Target Critic Networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # Scheduler
        self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.9, patience=20)
        self.critic1_scheduler = ReduceLROnPlateau(self.critic1_optimizer, mode='min', factor=0.9, patience=20)
        self.critic2_scheduler = ReduceLROnPlateau(self.critic2_optimizer, mode='min', factor=0.9, patience=20)

        # Hyperparameters
        self.discount = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Entropy regularization coefficient

        # Automatic Entropy Tuning
        self.target_entropy = -np.prod(action_space.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

    def select_action(self, state: np.ndarray, explore: bool=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if explore:
            action, _, _ = self.actor.sample(state)
        else:
            _, action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train_step(self, batch_size):
        states, actions, rewards, next_states, dones, weights, batch_idxes, replace_indices = self.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        weights = torch.FloatTensor(weights).to(device)

        # Sample actions from the policy for next states
        with torch.no_grad():
            next_state_actions, next_state_log_pis, _ = self.actor.sample(next_states)
            next_state_inputs = torch.cat([next_states, next_state_actions], dim=1)
            q1_next_target = self.critic1_target(next_state_inputs)
            q2_next_target = self.critic2_target(next_state_inputs)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pis
            next_q_value = rewards + (1 - dones) * self.discount * min_q_next_target

        # Critic losses
        state_actions = torch.cat([states, actions], dim=1)
        q1_current = self.critic1(state_actions)
        q2_current = self.critic2(state_actions)
        critic1_loss = F.mse_loss(q1_current, next_q_value)
        critic2_loss = F.mse_loss(q2_current, next_q_value)

        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor loss
        pi, log_pi, _ = self.actor.sample(states)
        state_pi = torch.cat([states, pi], dim=1)
        q1_pi = self.critic1(state_pi)
        q2_pi = self.critic2(state_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Entropy temperature adjustment
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # Soft update of target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update learning rates
        self.actor_scheduler.step(actor_loss.item())
        self.critic1_scheduler.step(critic1_loss.item())
        self.critic2_scheduler.step(critic2_loss.item())

        if self.use_per and batch_idxes is not None:
            # Update priorities in PER buffer
            with torch.no_grad():
                td_errors = (q1_current - next_q_value).abs().cpu().numpy() + 1e-6
                if replace_indices is not None:
                    td_errors = np.delete(td_errors, replace_indices)
                self.buffer.update_priorities(batch_idxes, td_errors.squeeze())
        critic_loss = (critic1_loss.item() + critic2_loss.item()) / 2
        return critic_loss, actor_loss.item()

def unit_test():
    from config import N_Action, N_State, Action_Space
    agent = SACAgent(N_State, N_Action, Action_Space)
    s = agent.env.reset()
    print(agent.select_action(s, explore=False))

# python -m RLs.SAC
if __name__ == '__main__':
    unit_test()
