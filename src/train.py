import click
from RLs.TD3 import TD3Agent
from RLs.PPO import PPOAgent
from RLs.DDPG import DDPGAgent
from RLs.SAC import SACAgent
from RLs.DQN import DQNAgent
from RLs.Trainer import Trainer
from config import logging, N_Action, N_State

logger = logging.getLogger(__name__)

@click.command()
@click.option('--model', default='td3', help='RL model to train.')
@click.option('--use_per', is_flag=True, default=False, help='Use Prioritized Experience Replay.')
@click.option('--use_trust', is_flag=True, default=False, help='Use Trusted Experience Replay.')
@click.option('--batch_size', default=256, help='Batch size for training.')
@click.option('--total_steps', default=1e5, help='Number of steps to train.')
@click.option('--save_path', default='../ckpts/TD3', help='Path to save the trained model.')
@click.option('--start_timesteps', default=500, help='Timesteps to start training.')
@click.option('--log_episodes', default=10, help='Log every n episodes.')
@click.option('--eval_steps', default=1000, help='Number of steps to evaluate the model.')

def train(model, use_per, use_trust, batch_size, total_steps, save_path, start_timesteps, log_episodes, eval_steps):
    """Train a RLs agent with given parameters."""
    if model == 'td3':
        agent = TD3Agent(N_State, N_Action, use_per=use_per, use_trust=use_trust)
    elif model == 'ppo':
        agent = PPOAgent(N_State, N_Action)
    elif model == 'dqn':
        agent = DQNAgent(N_State, N_Action, use_per=use_per, use_trust=use_trust)
    elif model == 'ddpg':
        agent = DDPGAgent(N_State, N_Action, use_per=use_per, use_trust=use_trust)
    elif model == 'sac':
        agent = SACAgent(N_State, N_Action, use_per=use_per, use_trust=use_trust)
    else:
        raise ValueError(f"Model {model} not supported.")
    trainer = Trainer(agent, batch_size=batch_size, total_steps=total_steps, save_path=save_path, start_timesteps=start_timesteps, log_episodes=log_episodes, eval_steps=eval_steps)
    trainer.train()
# 
if __name__ == '__main__':
    train()
