from .transfer_learner import TransferLearner


class TransferLearningFreezeNLayers(TransferLearner):
    """
    This transfer learning only initialize the initial weights of the model, which should have been
    loaded earlier. All other function calls will just pass back to the caller.
    """

    def __init__(self, optim=None, models=None, total_frames=None, n_frozen_layers=1):
        super(TransferLearningFreezeNLayers, self).__init__(optim=optim, models=models, total_frame=total_frames)

        self.optim = optim

        # Check lr defined in original optimizer
        self.lr = self.optim.param_groups[0]['lr']

        # Get rid of all parameter groups to start
        while len(self.optim.param_groups) > 0:
            del self.optim.param_groups[0]

        for i in range(len(models)):
            layer_count = 0
            layers = list(models[i].named_children())
            for _, cur_layer in layers[::-1]:
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= n_frozen_layers:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': self.lr})
                    else:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': 0})



    def update_learning_rates(self, cur_frame=0):
        """
        Learning rates don't change in this version
        """
        pass

