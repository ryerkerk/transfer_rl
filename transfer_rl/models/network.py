import torch.nn as nn
import torch


class FeedForwardActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_nodes, device, std=0.1):
        super(FeedForwardActorCritic, self).__init__()
        self.device = device

        act_func = nn.ReLU
        self.actor = nn.ModuleList()
        self.actor.append(nn.Linear(num_inputs, hidden_nodes[0]))
        self.actor.append(act_func())
        for i in range(len(hidden_nodes) - 1):
            self.actor.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            self.actor.append(act_func())

        self.actor.append(nn.Linear(hidden_nodes[-1], num_outputs))
        self.actor.append(nn.Tanh())

        self.critic = nn.ModuleList()
        self.critic.append(nn.Linear(num_inputs, hidden_nodes[0]))
        self.critic.append(act_func())
        for i in range(len(hidden_nodes) - 1):
            self.critic.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            self.critic.append(act_func())

        self.critic.append(nn.Linear(hidden_nodes[-1], 1))

        self.actor.to(self.device)
        self.critic.to(self.device)

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

    def sample_action(self, state, std):
        """
        Get action, and then apply some variance to better explore policy search space
        """

        actions_mean = self.get_action(state)
        co_var_mat = torch.eye(actions_mean.shape[1]).repeat(actions_mean.shape[0], 1, 1) * std * std
        dist = torch.distributions.MultivariateNormal(actions_mean, co_var_mat)
        actions = dist.sample()
        # d = actions-actions_mean
        # d = torch.clamp(d, -self.action_var*2, self.action_var*2)
        # actions = actions_mean + d
        # actions = torch.clamp(actions, -1, 1)
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

    def get_logp_value_ent(self, state, actions_prev, std):
        """
        Used to evaluate previous states and actions

        """
        # Run state through actor network

        actions_cur_p = self.get_action(state) # Actions on current policy
        values = self.get_value(state)

        co_var_mat = torch.eye(actions_cur_p.shape[1]).repeat(actions_cur_p.shape[0], 1, 1) * std * std
        dist = torch.distributions.MultivariateNormal(actions_cur_p, co_var_mat)

        logp = dist.log_prob(actions_prev)   # Log probability of this
        entropy = dist.entropy()

        return logp, torch.squeeze(values), entropy
