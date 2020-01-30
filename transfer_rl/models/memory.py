import numpy as np

class Buffer():
    def __init__(self, capacity=-1):
        """

        :param capacity: A capacity of -1 means no maximum memory length.
                         Instead the current memory index must be periodically
                         cleared with the reset() command
        """
        self.actions = []
        self.states = []
        self.next_states = []
        self.logp = []
        self.rewards = []
        self.dones = []
        self.pt = 0
        self.capacity = capacity

    def reset(self):
        """
        Reset memory by setting index back to 0.
        """
        self.pt = 0

    def push(self, action, states, logp, reward, done, next_states):
        if len(self.actions) > self.pt:
            self.actions[self.pt] = action
            self.states[self.pt] = states
            self.next_states[self.pt] = next_states
            self.logp[self.pt] = logp
            self.rewards[self.pt] = reward
            self.dones[self.pt] = done
        else:
            self.actions.append(action)
            self.states.append(states)
            self.next_states.append(next_states)
            self.logp.append(logp)
            self.rewards.append(reward)
            self.dones.append(done)

        self.pt += 1
        if self.capacity > 0:
            self.pt = self.pt % self.capacity

    def get_all(self):
        """
        :return: All memory in buffer up to current index
        """
        return self.actions[:self.pt], self.states[:self.pt], self.logp[:self.pt], \
               self.rewards[:self.pt], self.dones[:self.pt], self.next_states[:self.pt]

    def sample(self, batch_size):
        """
        Return a random sample of memories
        :param n: Number of samples to return
        """

        indices = np.random.choice(len(self.actions), batch_size)
        actions = [self.actions[idx] for idx in indices]
        states = [self.states[idx] for idx in indices]
        next_states = [self.next_states[idx] for idx in indices]
        logp = [self.logp[idx] for idx in indices]
        rewards = [self.rewards[idx] for idx in indices]
        dones = [self.dones[idx] for idx in indices]

        # print(actions)
        return actions, states, logp, rewards, dones, next_states

    def __len__(self):
        return self.pt


