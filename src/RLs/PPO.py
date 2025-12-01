import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau

from RLs.BaseAgent import BaseAgent
from RLs.NetWork import PPOActorNetwork, PPOCriticNetwork, device
from RLs.utils import activate_A_func
from config import A_Scale, N_Action, N_State


class PPOAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, gamma=0.99, lamda=0.95, clip_eps=0.2, update_epochs=10, entropy_coeff=0.01):
        super(PPOAgent, self).__init__(use_per=False, use_trust=False)
        self.name = 'PPO'
        self.actor = PPOActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = PPOCriticNetwork(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=5e-5)
        self.actor_scheduler = ReduceLROnPlateau(self.actor_optimizer, mode='min', factor=0.95, patience=20)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)
        self.critic_scheduler = ReduceLROnPlateau(self.critic_optimizer, mode='min', factor=0.95, patience=20)
        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.mse_loss = nn.MSELoss()
        self.memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_scale = A_Scale
        self.entropy_coeff = entropy_coeff


    def select_action(self, state, explore: bool=True):
        state = state / sum(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        k = torch.count_nonzero(state, dim=1).item()  # Count non-zero elements in the state tensor
        k_tensor = torch.tensor([k], dtype=torch.int).to(self.device)  # Convert k to a tensor and move to device
        with torch.no_grad():
            action_mean, action_std = self.actor(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = torch.clamp(action, -1, 1)
        action = activate_A_func(action, k_tensor)
        action *= self.action_scale
        return action.cpu().data.numpy().flatten(), action_log_prob.cpu().data.numpy().flatten()

    def store_experience(self, transition):
        self.memory.append(transition)

    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lamda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages
    
    def train_step(self, batch_size=64):
        
        states, actions, log_probs_old, rewards, dones, values, next_values = zip(*self.memory)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(self.device)
        next_values = torch.tensor(np.array(next_values), dtype=torch.float32).to(self.device)

        advantages = self.compute_advantages(rewards, values, next_values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + values

        # # Standardize advantages
        # if len(advantages) > 1:
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        critic_loss_record = []
        actor_loss_record = []
        train_step = 0
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                # Update Actor
                action_mean, action_std = self.actor(batch_states)
                dist = Normal(action_mean, action_std)
                log_probs = dist.log_prob(batch_actions).sum(axis=-1)
                batch_log_probs_old = batch_log_probs_old.sum(axis=-1)
                ratio = torch.exp(log_probs - batch_log_probs_old)

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.entropy_coeff * dist.entropy().mean()
                total_loss = policy_loss + entropy_loss

                self.actor_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # Update Critic
                state_values = self.critic(batch_states).squeeze()
                value_loss = self.mse_loss(state_values, batch_returns)

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(list(self.critic.parameters()), max_norm=0.5)
                self.critic_optimizer.step()
                
                actor_loss_record.append(total_loss.item())
                critic_loss_record.append(value_loss.item())
                train_step += 1
                
            
            # Update learning rate schedulers after each epoch
            self.critic_scheduler.step(np.mean(critic_loss_record))
            self.actor_scheduler.step(np.mean(actor_loss_record))

        self.memory = []
        return np.mean(critic_loss_record), np.mean(actor_loss_record), train_step
            

def unit_test():
    agent = PPOAgent(N_State, N_Action)
    agent.train(10)
    

# python -m RLs.PPO
if __name__ == '__main__':
    unit_test()