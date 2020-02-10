import torch
import math
from .controller import Controller
from .network import FeedForwardPPO
from ..transfer_learning import TransferLearningInitializeOnly, TransferLearningFreezeNLayers, \
                                TransferLearningFreezeNLayersFullThaw
import numpy as np

class PPO(Controller):
    """
    The proximal policy optimization (PPO) algorithm is based on the paper
    "Proximal Policy Optimization Algorithms" by Schulman et al.


    """
    def __init__(self, device='cpu'):
        super(PPO, self).__init__()
        self.device = torch.device(device)
        self.train_steps = 0
        self.adaptive_action_std = False # Set to true by calling set_adaptive_action_std()

    def create_model(self, n_features, n_actions, hidden_layers=[10, 10, 10], action_std=0.1,
                     gamma=0.99, eps=0.2, learning_rate=1e-4, train_steps=80, batch_size=10000,
                     optimizer_type='adam', transfer_learning=None, total_frames=1e6,
                     tl_start = 0.1, tl_end=0.3):

        self.model = FeedForwardPPO(n_features, n_actions, hidden_layers,
                                    self.device, action_std)
        self.model.to(self.device)

        self.train_steps = train_steps  # Initialize number of training steps to 0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.eps = eps
        self.action_std = action_std
        self.batch_size = batch_size

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

        if transfer_learning == 'initialize' or transfer_learning == 'none':
            self.transfer_learning = TransferLearningInitializeOnly(optim = self.optimizer,
                                                                    models = [self.model.actor, self.model.critic],
                                                                    total_frames = total_frames)
        elif transfer_learning == 'freeze_all_but_final':
            self.transfer_learning = TransferLearningFreezeNLayers(optim=self.optimizer,
                                                                    models=[self.model.actor, self.model.critic],
                                                                    total_frames=total_frames)
        elif transfer_learning == 'freeze_then_thaw':
            self.transfer_learning = TransferLearningFreezeNLayersFullThaw(optim=self.optimizer,
                                                               models=[self.model.actor, self.model.critic],
                                                               tl_end=tl_end, total_frames=total_frames)
        else:
            raise Exception('Unrecognized transfer learning')

        print(self.optimizer)
        print(self.model.forward)

        pass


    def sample_action(self, state):
        """

        """
        with torch.no_grad():
            actions, logp = self.model.sample_action(state, self.action_std)

        return actions, logp

    def check_train(self, mem, frames):
        if len(mem) >= self.batch_size:
            self.transfer_learning.update_learning_rates(frames)
            self.train(mem)
            mem.reset()
            """ Debugging
            w = []
            for i in range(len(self.model.actor)):
                if hasattr(self.model.actor[i], 'weight'):
                    w.append(float(self.model.actor[i].weight.sum().detach()))
                    w.append(float(self.model.actor[i].bias.sum().detach()))
            print(w)
            """

        self.update_action_std(frames)

    def train(self, mem):
        """
        Train the PPO model using the given memory buffer.

        This function is based on the one provided by Nikhil Barhate at
        https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py
        """

        batch_actions, batch_states, batch_logp, batch_rewards, batch_dones, _ = mem.get_all()

        # Calculate discounted rewards from memory
        discounted_rewards = [0 for i in range(len(batch_rewards))]
        r = 0
        for i, (reward, done) in enumerate(zip(batch_rewards[::-1], batch_dones[::-1])):
            r = reward + self.gamma * r * (1. - done)  #
            discounted_rewards[-1 - i] = r

        # Normalize discounted rewards and move to a torch tensor
        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # Move actions, logp values, and states from memory into torch tensor
        batch_actions = torch.squeeze(torch.stack(batch_actions).to(self.device)).detach()
        batch_logp = torch.squeeze(torch.stack(batch_logp).to(self.device)).detach()
        batch_states = torch.squeeze(torch.stack(batch_states).to(self.device)).detach()

        # Train model for the a defined myber of steps.
        self.model.train()
        for _ in range(self.train_steps):

            # Get logp, critic values, and entropy for batch using CURRRENT model
            logp, values, entropy = self.model.get_logp_value_ent(batch_states, batch_actions, self.action_std)

            # Calculate ratio between logp values in CURRENT model and model before training
            # Eq (6) of Schulman et al.
            ratios = torch.exp(logp - batch_logp.squeeze().detach())

            # Calculate surrogate loss (l1)
            # This is the minimum of the surrogate objective used in TRPO (surr1), and the
            # clipped surrogate objected proposed in PPO (surr2).
            # Eq (6-7) of Schulman et al.
            advantages = discounted_rewards.squeeze() - values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            l1 = -torch.min(surr1, surr2)

            # Calculate loss on state-value function (critic)
            l2 = 0.5 * (values - discounted_rewards)**2

            # Entropy bonus to promote exploration
            l3 = - 0.01 * entropy

            # Final loss function, Eq (9) of Schulman et al.
            loss = l1 + l2 + l3

            # Take a gradient stop
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def reset_final_layer(self):
        """
        Reinitializes the weights for the final layer of the actor and model networks to random values.
        """

        models = [self.model.actor, self.model.critic]

        for i in range(len(models)):
            layer_count = 0
            layers = list(models[i].named_children())
            # Iterate over layers, starting from the final layers
            for _, cur_layer in layers[::-1]:
                # Only consider layers that define weights
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= 1:
                        # Reset weights of final layer
                        stdv = 1. / math.sqrt(cur_layer.weight.size(1))
                        cur_layer.weight.data.uniform_(-stdv, stdv)
                        if cur_layer.bias is not None:
                            cur_layer.bias.data.uniform_(-stdv, stdv)

    def add_noise_layers(self, n, alpha=1):
        """
        Add noise to the final n layers of both the actor and critic networks.

        :param n: Number of layers to add noise to, counting from final layer
        :param alpha: Noise added is alpha*std(layer weights)
        """
        models = [self.model.actor, self.model.critic]

        for i in range(len(models)):
            layer_count = 0
            layers = list(models[i].named_children())
            for _, cur_layer in layers[::-1]:
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= n:
                        with torch.no_grad():
                            cur_layer.weight.add_(torch.nn.Parameter(torch.randn(cur_layer.weight.size()) * alpha * cur_layer.weight.std()))

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
