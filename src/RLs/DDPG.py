import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import A_Scale, N_Action
from RLs.BaseAgent import BaseAgent
from RLs.NetWork import DeterActorNet, DoubleQNet, device, QNetwork

class DDPGAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, use_per: bool = False, use_trust: bool = False):
        super(DDPGAgent, self).__init__(use_per, use_trust)
        self.name = 'ddpg'
        self.actor = DeterActorNet(state_dim, action_dim).to(device)
        self.actor_target = DeterActorNet(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.9, patience=20)

        self.critic = QNetwork(state_dim, action_dim).to(device)
        self.critic_target = QNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-5)
        self.critic_scheduler = ReduceLROnPlateau(self.critic_optimizer, mode='min', factor=0.9, patience=20)
        
        self.action_scale = A_Scale
        self.discount = 0.99
        self.tau = 0.01
        # explore config
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.15


    def select_action(self, state: np.ndarray, explore: bool=True):
        if explore and np.random.rand() < self.epsilon:
            action = self.env.get_random_legal_action(state)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        else:
            state = state / sum(state)
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            k = torch.count_nonzero(state, dim=1).item()  # Count non-zero elements in the state tensor
            k_tensor = torch.tensor([k], dtype=torch.int).to(self.device)  # Convert k to a tensor and move to device
            with torch.no_grad():
                self.actor.eval()
                action = self.actor(state, k_tensor).cpu().data.numpy().flatten()
            action *= self.action_scale
        return action


    def train_step(self, batch_size):
        states, actions, rewards, next_states, dones, weights, batch_idxes, replace_indices = self.sample(batch_size)

        # Training logic
        with torch.no_grad():
            k = torch.count_nonzero(next_states, dim=1)  # Count non-zero elements in the state tensor
            next_actions = self.actor_target(next_states, k)
            target_q = self.critic_target(next_states, next_actions)
            rewards = rewards.view(-1, 1)
            dones = dones.view(-1, 1)
            target_q = rewards + (1 - dones) * self.discount * target_q
        self.actor.train()
        self.critic.train()
        current_q = self.critic(states, actions)
        TD_Error = current_q - target_q
        weighted_TD_errors= weights * TD_Error

        # Use Mean Squared Error Loss directly
        critic_loss = weighted_TD_errors.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # Gradient clipping
        self.critic_optimizer.step()

        # Update the actor network using the deterministic policy gradient
        k = torch.count_nonzero(states, dim=1)  # Count non-zero elements in the state tensor
        actor_loss = -self.critic(states, self.actor(states, k)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # Gradient clipping
        self.actor_optimizer.step()

        if self.use_per and batch_idxes is not None:
            new_priorities = (torch.abs(TD_Error)).cpu().data.numpy() + 1e-6
            if replace_indices is not None:
                new_priorities = np.delete(new_priorities, replace_indices)
            self.buffer.update_priorities(batch_idxes, new_priorities)

        # Update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        # Update the learning rate
        self.actor_scheduler.step(actor_loss.item())
        self.critic_scheduler.step(critic_loss.item())
        
        return critic_loss.item(), actor_loss.item()

def unit_test():
    from config import N_Action, N_State
    agent = DDPGAgent(N_State, N_Action, use_per=True)
    s = agent.env.reset()
    print(agent.select_action(s, explore=False))

# python -m RLs.DDPG
if __name__ == '__main__':
    unit_test()