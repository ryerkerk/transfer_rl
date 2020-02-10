from .transfer_learner import TransferLearner


class TransferLearningFreezeAllButNLayers(TransferLearner):
    """
    This transfer learning function freezes all but the final n layers of the model by setting their
    learning rates to 0. The final n layers will have the learning rate set in provided optimizer
    """

    def __init__(self, optim=None, models=None, total_frames=None, n_frozen_layers=1):
        super(TransferLearningFreezeAllButNLayers, self).__init__(optim=optim, models=models, total_frame=total_frames)

        self.optim = optim

        # Check lr defined in original optimizer
        self.lr = self.optim.param_groups[0]['lr']

        # Get rid of all parameter groups to start
        while len(self.optim.param_groups) > 0:
            del self.optim.param_groups[0]

        # Loop over models (e.g., actor and critic)
        for i in range(len(models)):
            layer_count = 0
            layers = list(models[i].named_children())
            # Loop over layers n model
            for _, cur_layer in layers[::-1]:
                # Only consider layers that define weights (i.e., ignore ReLu or Tanh layers)
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= n_frozen_layers:
                        # Train first n layers
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': self.lr})
                    else:
                        # Freeze all other layers
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': 0})



    def update_learning_rates(self, cur_frame=0):
        """
        Learning rates don't change in this version
        """
        pass

