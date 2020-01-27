import torch
from .controller import Controller
from .network import FeedForwardActorCritic
import numpy as np

class PPO(Controller):
    """
    """

    def __init__(self, device='cuda'):
        super(PPO, self).__init__()
        self.device = torch.device(device)
        self.train_steps = 0

        # These will be set when creating model
        self.model = None
        self.gamma = 0.99
        self.eps = 0.2
        self.optimizer = None
        self.learning_rate = 1e-2
        self.MSELoss = torch.nn.MSELoss()
        self.train_steps = 80

    def create_model(self, n_features, n_actions, hidden_layers=[10, 10, 10], action_std=0.1,
                     gamma=0.99, eps=0.2, learning_rate=1e-4, device = 'cuda', train_steps=80,
                     optimizer_type='adam'):

        self.device = device
        self.model = FeedForwardActorCritic(n_features, n_actions, hidden_layers,
                                               self.device, action_std)
        self.model.to(self.device)

        self.train_steps = train_steps  # Initialize number of training steps to 0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps = eps

        # Optimizer to be used
        if optimizer_type == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise Exception('Unrecognized optimizer')

        print(self.model.forward)

        pass

    def sample_action(self, state):
        """

        """
        with torch.no_grad():
            actions, logp = self.model.sample_action(state)

        return actions, logp

    def train(self, mem):
        batch_actions, batch_states, batch_logp, batch_rewards, batch_dones = mem.get_all()

        discounted_rewards = [0 for i in range(len(batch_rewards))]
        r = 0
        for i, (reward, done) in enumerate(zip(batch_rewards[::-1], batch_dones[::-1])):
            r = reward + self.gamma * r * (1. - done)  #
            discounted_rewards[-1 - i] = r

        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        batch_actions = torch.squeeze(torch.stack(batch_actions).to(self.device)).detach()
        batch_logp = torch.squeeze(torch.stack(batch_logp).to(self.device)).detach()
        batch_states = torch.squeeze(torch.stack(batch_states).to(self.device)).detach()

        self.model.train()

        for _ in range(self.train_steps):
            logp, values, entropy = self.model.get_logp_value_ent(batch_states, batch_actions)

            ratios = torch.exp(logp - batch_logp.squeeze().detach())

            # Finding Surrogate Loss:
            advantages = discounted_rewards.squeeze() - values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            l1 = -torch.min(surr1, surr2)
            #l2 = 0.5 * self.MSELoss(values, discounted_rewards)
            l2 = 0.5 * (values - discounted_rewards)**2
            l3 = - 0.01 * entropy
            loss = l1 + l2 + l3

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def save_model(self, path):
        """
        Save the current policy net model to the given path
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load the policy net model from the given path
        """
        self.model.load_state_dict(torch.load(path))
