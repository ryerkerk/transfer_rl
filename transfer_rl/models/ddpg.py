import torch
from .controller import Controller
from .network import FeedForwardDDPG
import numpy as np

class DDPG(Controller):
    """


    This implementation is based on the one found at:
    https://github.com/ghliu/pytorch-ddpg/blob/master/ddpg.py
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.target_model = None
        self.batch_size=128
        self.train_steps = 100
        self.learning_rate = 1e-4
        self.eps = 0.2
        self.gamma = 0.99
        self.action_std = .5
        self.action_std_decay = 0.999
        self.train_epochs_elapsed = 0
        self.frames_per_train = 1
        self.frames_elapsed = 0
        self.tau = 0.001

    def create_model(self, n_features, n_actions, hidden_layers=[10, 10, 10], action_std=0.5,
                     action_std_decay=0.999, gamma=0.99, eps=0.2, learning_rate=1e-4, train_steps=1,
                     optimizer_type='adam', batch_size=128, frames_per_train=1, tau=0.001):

        self.frames_per_train = frames_per_train
        self.frames_elapsed = 0
        self.train_epochs_elapsed = 0
        self.model = FeedForwardDDPG(n_features, n_actions, hidden_layers,
                                    self.device, action_std)

        self.target_model = FeedForwardDDPG(n_features, n_actions, hidden_layers,
                                           self.device, action_std)

        print(self.model)

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.batch_size=batch_size
        self.train_steps = train_steps  # Initialize number of training steps to 0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps = eps
        self.tau = tau

        self.action_std = action_std
        self.action_std_decay = action_std_decay
        self.crit_loss = torch.nn.MSELoss()

        if optimizer_type == 'adagrad':
            self.actor_optimizer = torch.optim.Adagrad(self.model.actor.parameters(), lr=learning_rate)
            self.critic_optimizer = torch.optim.Adagrad(self.model.critic.parameters(), lr=learning_rate*1)
        elif optimizer_type == 'adam':
            self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=learning_rate)
            self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=learning_rate*10, weight_decay=1e-2)
        elif optimizer_type == 'rmsprop':
            self.actor_optimizer = torch.optim.RMSprop(self.model.actor.parameters(), lr=learning_rate)
            self.critic_optimizer = torch.optim.RMSprop(self.model.critic.parameters(), lr=learning_rate*1)
        elif optimizer_type == 'sgd':
            self.actor_optimizer = torch.optim.SGD(self.model.actor.parameters(), lr=learning_rate)
            self.critic_optimizer = torch.optim.SGD(self.model.critic.parameters(), lr=learning_rate*1)
        else:
            raise Exception('Unrecognized optimizer')

    def sample_action(self, state):
        """

        """
        with torch.no_grad():
            actions, logp = self.model.sample_action(state, self.action_std)

        self.frames_elapsed += 1
        return actions, logp

    def check_train(self, mem):
        if len(mem) >= 100 and self.frames_elapsed % self.frames_per_train == 0:
            self.train(mem)

    def train(self, mem):
        self.train_epochs_elapsed += 1

        # self.action_std = self.action_std*self.action_std_decay


        for _ in range(self.train_steps):
            batch_actions, batch_states, batch_logp, batch_rewards, batch_dones, batch_next_state = mem.sample(
                self.batch_size)

            self.model.train()

            batch_actions = torch.squeeze(torch.stack(batch_actions).to(self.device)).detach()
            batch_states = torch.squeeze(torch.stack(batch_states).to(self.device)).detach()
            batch_next_state = torch.squeeze(torch.stack(batch_next_state).to(self.device)).detach()
            batch_rewards = torch.squeeze(torch.tensor(batch_rewards).to(self.device)).view(-1, 1).detach()
            batch_dones = torch.squeeze(torch.tensor(batch_dones).to(self.device)).view(-1, 1).detach()

            with torch.no_grad():
                target_actions = self.target_model.get_action(batch_next_state)
                next_q_values = self.target_model.get_value(batch_next_state, target_actions)
                target_q_batch = batch_rewards + self.gamma * (1 - batch_dones.float())

            # Critic update
            # self.model.critic.zero_grad()
            q_batch = self.model.get_value(batch_states, batch_actions)

            value_loss = self.crit_loss(q_batch, target_q_batch)
            # value_loss = value_loss.mean()
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            # Actor update
            # self.model.actor.zero_grad()

            pred_action = self.model.get_action(batch_states)
            policy_loss = -1 * self.model.get_value(batch_states, pred_action)

            policy_loss = policy_loss.mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.soft_update_target()

    def soft_update_target(self):
        for target_param, param in zip(self.target_model.critic.parameters(), self.model.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_model.actor.parameters(), self.model.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau)
