import torch.nn as nn
import torch


class FeedForwardActorCritic(nn.Module):
    """
    This model maintains feed forward actor and critic network. Functons are provided to
    get deterministic action and value outputs, or to sample actions with a provided level
    of noise applied to the action state.
    """
    def __init__(self, num_inputs, num_outputs, hidden_nodes, device, std=0.1):
        """
        Initialize by creating the actor and critic networks

        :param num_inputs: Number of state variables provided to actor/critic networks
        :param num_outputs: Number of action outputs requried by actor network
        :param hidden_nodes: List containing number of nodes in each hidden layer
        :param device: Device on which to place networks
        :param std:
        """
        super(FeedForwardActorCritic, self).__init__()
        self.device = device

        # Activation function that will be used for each hidden laer
        act_func = nn.ReLU

        # Create actor network
        self.actor = nn.ModuleList()
        self.actor.append(nn.Linear(num_inputs, hidden_nodes[0]))
        self.actor.append(act_func())
        for i in range(len(hidden_nodes) - 1):
            # Create a new fully connected layer for each hidden node layer
            self.actor.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            self.actor.append(act_func())

        self.actor.append(nn.Linear(hidden_nodes[-1], num_outputs))
        self.actor.append(nn.Tanh())

        # Create the critic network
        self.critic = nn.ModuleList()
        self.critic.append(nn.Linear(num_inputs, hidden_nodes[0]))
        self.critic.append(act_func())
        for i in range(len(hidden_nodes) - 1):
            # Create a new fully connected layer for each hidden node layer
            self.critic.append(nn.Linear(hidden_nodes[i], hidden_nodes[i + 1]))
            self.critic.append(act_func())

        self.critic.append(nn.Linear(hidden_nodes[-1], 1))
        # No activation function used on final output of critic network

        self.actor.to(self.device)
        self.critic.to(self.device)

    def get_action(self, state):
        """
        Get deterministic action, with no noise applied.
        """

        actions = state
        for layer in self.actor:
            actions = layer(actions)

        return actions

    def sample_action(self, state, std):
        """

        :param state: Current envinronment state
        :param std: Standard deviation of noise to apply to action space
        """

        # Get deterministic actions
        actions_mean = self.get_action(state)

        # Apply noise to action space
        co_var_mat = torch.eye(actions_mean.shape[1]).repeat(actions_mean.shape[0], 1, 1) * std * std
        dist = torch.distributions.MultivariateNormal(actions_mean, co_var_mat)
        actions = dist.sample()

        # Get log probability of taking these actions
        logp = dist.log_prob(actions)

        return actions.detach(), logp.detach()

    def get_value(self, state):
        """
        Get state value from critic
        """

        values = state
        for layer in self.critic:
            values = layer(values)

        return values

    def get_logp_value_ent(self, state, actions_prev, std):
        """
        This function evaluates the current network on a previous state, and returns the log
        probability of having taken the action taken by the prevous network on this state, the
        value function of the current critic on this state, and the entropy of the noisy action space

        :param state: Previous state of environment from memory
        :param actions_prev: Action taken in this state
        :param std: Standard deviation of noise that was used to sample action in this state
        :return:
        """

        # Get deterministic actions and values using the current policy on prevous state
        actions_cur_p = self.get_action(state)
        values = self.get_value(state)

        # Get log probability of the current network taking the action that was taken previously
        co_var_mat = torch.eye(actions_cur_p.shape[1]).repeat(actions_cur_p.shape[0], 1, 1) * std * std
        dist = torch.distributions.MultivariateNormal(actions_cur_p, co_var_mat)
        logp = dist.log_prob(actions_prev)

        # Entropy of the noisy action space
        entropy = dist.entropy()

        return logp, torch.squeeze(values), entropy