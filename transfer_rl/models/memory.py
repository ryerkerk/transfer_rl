import numpy as np


class Buffer():
    def __init__(self):
        self.actions = []
        self.obs = []
        self.logp = []
        self.rewards = []
        self.dones = []
        self.pt = 0

    def reset(self):
        self.pt = 0

    def push(self, action, obs, logp, reward, done):
        if len(self.actions) > self.pt:
            self.actions[self.pt] = action
            self.obs[self.pt] = obs
            self.logp[self.pt] = logp
            self.rewards[self.pt] = reward
            self.dones[self.pt] = done
        else:
            self.actions.append(action)
            self.obs.append(obs)
            self.logp.append(logp)
            self.rewards.append(reward)
            self.dones.append(done)

        self.pt += 1

    def get(self):
        return np.vstack(self.actions[:self.pt]).astype(np.float32), \
               np.vstack(self.obs[:self.pt]).astype(np.float32), \
               np.vstack(self.logp[:self.pt]).astype(np.float32), \
               np.vstack(self.rewards[:self.pt]).astype(np.float32), \
               np.vstack(self.dones[:self.pt])

    def __len__(self):
        return self.pt


