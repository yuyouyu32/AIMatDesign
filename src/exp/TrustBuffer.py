import bisect
import json

from exp.ReplayBuffer import ReplayBuffer
import random


class TrustReplayBuffer(ReplayBuffer):
    def __init__(self, jsonl_file):
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()
        
        self._storage = []
        self._rewards = []
        for line in lines:
            data = json.loads(line)
            obs_t = (data["s"], data["a"], data["r"], data["s_"], data["done"])
            self._storage.append(obs_t)
            self._rewards.append(data["r"])
        
        self._maxsize = len(self._storage)
        self._next_idx = len(self._storage)
        
        # Sort by reward
        self._storage.sort(key=lambda x: x[2])
        self._rewards.sort()
        self.random_ratio = 0.2
        self.ave_reward = sum(self._rewards) / len(self._rewards)

    def add(self, obs_t, action, reward, obs_tp1, done):
        raise NotImplementedError("Add method is not supported for JSONLReplayBuffer")

    def sample(self, expected_reward, batch_size):
        """Sample a batch of experiences closest to the expected reward with randomness.
        Parameters
        ----------
        expected_reward: float
            The target reward to approximate.
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            Batch of observations
        act_batch: np.array
            Batch of actions executed given obs_batch
        rew_batch: np.array
            Rewards received as results of executing act_batch
        next_obs_batch: np.array
            Next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # Find the index of the closest reward using binary search
        idx = bisect.bisect_left(self._rewards, expected_reward)
        
        # Define a range around the closest index for random selection
        range_size = min(max(batch_size, 10), len(self._storage))
        start = max(0, idx - range_size // 2)
        end = min(len(self._storage), start + range_size)
        
        # Adjust range to ensure sufficient samples
        if end - start < batch_size:
            start = max(0, end - batch_size)
            end = start + batch_size

        # Ensure the final indices are within valid bounds
        if end > len(self._storage):
            end = len(self._storage)
            start = end - batch_size

        # Ensure at least 10% randomness within the range
        extra_randomness = max(1, int(self.random_ratio * range_size))
        start = max(0, start - extra_randomness)
        end = min(len(self._storage), end + extra_randomness)
        # Randomly select batch_size samples within the defined range
        idxes = random.sample(range(start, end), min(batch_size, end - start))
        return self._encode_sample(idxes)
    
def unit_test():
    from config import TrustPoolPath
    trust_buffer = TrustReplayBuffer(TrustPoolPath)
    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = trust_buffer.sample(1.3, 20)
    print(rew_batch, done_mask)

# python -m exp.TrustBuffer
if __name__ == "__main__":
    unit_test()