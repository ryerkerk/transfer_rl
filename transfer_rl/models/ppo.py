import torch
import math
from .controller import Controller
from .network import FeedForwardPPO
from ..transfer_learning import TransferLearningInitializeOnly, TransferLearningFreezeNLayers, TransferLearningFreezeNLayersFullThaw
import numpy as np

class PPO(Controller):
    """
    """

    def __init__(self, device='cpu'):
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
        self.batch_size = 10000
        self.action_std = 0.5
        self.transfer_learning = None
        self.adaptive_action_std = False
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
        elif transfer_learning == 'freeze_first_layer':
            self.transfer_learning = TransferLearningFreezeNLayers(optim=self.optimizer,
                                                                    models=[self.model.actor, self.model.critic],
                                                                    total_frames=total_frames)
        elif transfer_learning == 'freeze_full_thaw':
            self.transfer_learning = TransferLearningFreezeNLayersFullThaw(optim=self.optimizer,
                                                               models=[self.model.actor, self.model.critic],
                                                               tl_start=tl_start, tl_end=tl_end,
                                                               total_frames=total_frames)
        else:
            raise Exception('Unrecognized transfer learning')

        print(self.optimizer)
        print(self.model.forward)

        pass

    def set_adaptive_action_std(self, action_std_start, action_std_final, action_std_end, total_frames):
        assert action_std_start >= 0, "Starting action std value needs to be greater than 0 when using adaptive action noise"
        assert action_std_final >= 0, "Ending action std value needs to be greater than 0 when using adaptive action noise"
        assert 0 <= action_std_end <= 1, "Transition period for adaptive action noise needs to be between 0 and 1"

        self.adaptive_action_std = True
        self.action_std_start = action_std_start
        self.action_std_final = action_std_final
        self.action_std_end_frame = action_std_end*total_frames
        self.action_std = self.action_std_start

    def update_action_std(self, frames):
        # Only update if using adaptive noise
        if self.adaptive_action_std == True:
            self.action_std = self.action_std_start \
                    + (self.action_std_final - self.action_std_start) * frames / self.action_std_end_frame
            if frames > self.action_std_end_frame:
                self.action_std = self.action_std_final


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
        batch_actions, batch_states, batch_logp, batch_rewards, batch_dones, _ = mem.get_all()

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
            logp, values, entropy = self.model.get_logp_value_ent(batch_states, batch_actions, self.action_std)

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

    def reset_final_layer(self):
        models = [self.model.actor, self.model.critic]

        for i in range(len(models)):
            layer_count = 0
            layers = list(models[i].named_children())
            for _, cur_layer in layers[::-1]:
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= 1:
                        stdv = 1. / math.sqrt(cur_layer.weight.size(1))
                        cur_layer.weight.data.uniform_(-stdv, stdv)
                        if cur_layer.bias is not None:
                            cur_layer.bias.data.uniform_(-stdv, stdv)

    def add_noise_layers(self, n, alpha=1):
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
