import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from RLs.utils import activate_A_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeterActorNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(DeterActorNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(s_shape, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU(),
            nn.Linear(128, a_shape),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, k):
        x = self.fc(x)
        a = activate_A_func(x, k)
        return a

class DoubleQNet(nn.Module):
    def __init__(self, s_shape, a_shape):
        super(DoubleQNet, self).__init__()
        self.fc_s = nn.Sequential(
            nn.Linear(s_shape, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),  
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU()
        )
        self.fc_a = nn.Sequential(
            nn.Linear(a_shape, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),  
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU()
        )
        self.output1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),  
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.output2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32),  
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, a):
        h1 = self.fc_s(x)
        h2 = self.fc_a(a)
        cat = torch.cat([h1, h2], dim=1)
        q1 = self.output1(cat)
        q2 = self.output2(cat)
        return q1, q2
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc_s = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.fc_a = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )
        self.fc_cat = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        self._initialize_weights()

    def forward(self, state, action):
        state_out = self.fc_s(state)
        action_out = self.fc_a(action)
        cat = torch.cat([state_out, action_out], dim=1)
        q_value = self.fc_cat(cat)
        return q_value
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
class PPOPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOPolicyNetwork, self).__init__()
        self.fc_s = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),  
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU()
        )
        self.mean_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.log_std_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        x = self.fc_s(state)
        action_mean = self.mean_head(x)
        action_log_std = self.log_std_head(x)
        action_std = torch.exp(action_log_std)
        state_value = self.value_head(x)
        return action_mean, action_std, state_value
    
class PPOActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),  
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.std_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.LogSigmoid()
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.fc(state)
        action_mean = self.mean_head(x)
        action_std = torch.exp(self.std_head(x))
        return action_mean, action_std

class PPOCriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(PPOCriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),  
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        state_value = self.fc(state)
        return state_value
    
class StochasticActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(StochasticActorNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),  
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.LeakyReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.log_std_head = nn.Linear(128, action_dim)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = self.fc(state)
        action_mean = self.mean_head(x)
        action_log_std = self.log_std_head(x)
        action_std = torch.exp(action_log_std)
        return action_mean, action_std, action_log_std

    def sample(self, state):
        action_mean, action_std, action_log_std = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, dist