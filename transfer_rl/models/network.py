import torch.nn as nn
import torch


class FeedForwardActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_nodes, device, std=0.1):
        super(FeedForwardActorCritic, self).__init__()
        self.device = device

        self.actor = nn.ModuleList()
        self.actor.append(nn.Linear(num_inputs, hidden_nodes[0]))
        self.actor.append(nn.ReLU())
        for i in range(len(hidden_nodes) - 1):
            self.actor.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            self.actor.append(nn.ReLU())

        self.actor.append(nn.Linear(hidden_nodes[-1], num_outputs))

        self.critic = nn.ModuleList()
        self.critic.append(nn.Linear(num_inputs, hidden_nodes[0]))
        self.critic.append(nn.ReLU())
        for i in range(len(hidden_nodes) - 1):
            self.critic.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            self.critic.append(nn.ReLU())

        self.critic.append(nn.Linear(hidden_nodes[-1], 1))

        self.action_std = std
        self.action_var = std * std

    def get_action(self, state):
        """
        Get action.

        """
        # Run state through actor network
        actions = state
        for layer in self.actor:
            actions = layer(actions)

        return actions

    def sample_action(self, state):
        """
        Get action, and then apply some variance to better explore policy search space
        """
        actions = self.get_action(state)
        co_var_mat = torch.eye(actions.shape[1]).repeat(actions.shape[0], 1, 1) * self.action_var
        dist = torch.distributions.MultivariateNormal(actions.cpu(), co_var_mat)
        actions = dist.sample()
        logp = dist.log_prob(actions)  # Log probability of this

        return actions.detach(), logp.detach()

    def get_value(self, state):
        """
        Get state value from critic
        """

        # Run state through actor network
        values = state
        for layer in self.critic:
            values = layer(values)

        return values

    def get_logp_value_ent(self, state):
        """
        Used to evaluate previous states and actions
        """
        # Run state through actor network
        actions = self.get_action(state)
        values = self.get_value(state)

        co_var_mat = torch.eye(actions.shape[1]).repeat(actions.shape[0], 1, 1) * self.action_var
        dist = torch.distributions.MultivariateNormal(actions, co_var_mat)
        logp = dist.log_prob(actions)   # Log probability of this
        entropy = dist.entropy()

        return logp, values, entropy