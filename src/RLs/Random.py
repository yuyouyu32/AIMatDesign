import numpy as np


from config import A_Scale, N_Action
from RLs.BaseAgent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, use_per: bool = False, use_trust: bool = False):
        super(RandomAgent, self).__init__(use_per, use_trust)
        self.name = 'Random'
        
    def select_action(self, state: np.ndarray, explore: bool=True):
        return self.env.get_random_action(state)

    def train_step(self, batch_size: int) -> float:
        pass
 
   
def unit_test():
    from config import N_Action, N_State
    agent = RandomAgent(N_State, N_Action, use_per=True)
    s = agent.env.reset()
    for i in range(100):
        print(sum(agent.select_action(s, explore=True)))

# python -m RLs.TD3
if __name__ == '__main__':
    unit_test()