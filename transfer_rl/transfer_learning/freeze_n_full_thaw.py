from .transfer_learner import TransferLearner


class TransferLearningFreezeNLayersFullThaw(TransferLearner):
    """
    This transfer learning only initialize the initial weights of the model, which should have been
    loaded earlier. All other function calls will just pass back to the caller.
    """

    def __init__(self, optim=None, models=None, total_frames=None, n_frozen_layers=1, tl_end=0.3):
        super(TransferLearningFreezeNLayersFullThaw, self).__init__(optim=optim, models=models, total_frame=total_frames)

        self.optim = optim

        self.tl_frame_end = tl_end*total_frames

        # Check lr defined in original optimizer
        self.lr = self.optim.param_groups[0]['lr']

        # Get rid of all parameter groups to start
        while len(self.optim.param_groups) > 0:
            del self.optim.param_groups[0]

        self.models = models
        for i in range(len(self.models)):
            layer_count = 0
            layers = list(self.models[i].named_children())
            for _, cur_layer in layers[::-1]:
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= n_frozen_layers:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': self.lr})
                    else:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': 0})

        self.layer_count = layer_count


    def update_learning_rates(self, cur_frame=0):

        n_layers = 1 + max(0, (self.layer_count-1) * cur_frame/self.tl_frame_end)
        # Get rid of all parameter groups to start
        while len(self.optim.param_groups) > 0:
            del self.optim.param_groups[0]

        for i in range(len(self.models)):
            layer_count = 0
            layers = list(self.models[i].named_children())
            for _, cur_layer in layers[::-1]:
                if hasattr(cur_layer, 'weight'):
                    layer_count += 1
                    if layer_count <= n_layers:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': self.lr})
                    else:
                        self.optim.add_param_group({'params': cur_layer.parameters(), 'lr': 0})

