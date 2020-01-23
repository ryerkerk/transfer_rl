import torch
from .controller import Controller
from .network import FeedForwardActorCritic

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

    def create_model(self, n_features, n_actions, hidden_layers=[10, 10, 10], action_std=0.1,
                     gamma=0.99, eps=0.2, learning_rate=1e-2,
                     optimizer_type='adam'):

        self.model = FeedForwardActorCritic(n_features, n_actions, hidden_layers,
                                               self.device, action_std)
        self.model.to(self.device)
        self.model.eval()

        self.train_steps = 0  # Initialize number of training steps to 0
        self.gamma = gamma  # Assign invalue values
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

        state = torch.tensor(state, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            self.model.eval()
            actions, logp = self.model.sample_action(state)

        return actions, logp